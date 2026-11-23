"""Step8：组件与结构实验，固定样本划分，权重 ES-based NGCI 与混合模型。

所有路径相对于 trial；所有新输出限制在 step8。Python 不自动启动 R。
命令顺序：setup → train → prepare → 手动 R → finalize，各子实验依次完成。
训练支持 Ctrl+C 或 stop 命令，在批次边界保存可恢复状态。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import time
import warnings

sys.dont_write_bytecode = True
# 必须在第一次创建 CUDA context 前设置；与确定性矩阵乘法配套。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
import torch
from torch import nn
import statsmodels
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("step8")
CODE = ROOT / "code/step8_experiments.py"
MASTER_SEED, N_MODELS, N_UNITS = 629, 20, 64
MATRIX = Path("step1/core_preprocessed/log2_tpm_plus_1.npy")
EXPERIMENTS = ("bn", "activation_bn", "dropout_bn", "structure")
PARAMETERS = dict(minSize=15, maxSize=500, eps=0, scoreType="std", gseaParam=1,
                  sampleSize=101, nPermSimple=1000)
BASELINE = "baseline"
STOP = ROOT / "control/STOP_TRAINING"
_stop_requested = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def enter_trial():
    """从当前工作目录向上寻找 trial，数据文件中不保存机器专属绝对路径。"""
    here = Path.cwd()
    for folder in (here, *here.parents):
        if (folder / CODE).is_file() and (folder / MATRIX).is_file():
            os.chdir(folder)
            return
    raise ValueError("请从 trial 或其子目录运行。")


def out(path):
    path = Path(path)
    require(not path.is_absolute() and path.parts[0] == "step8" and ".." not in path.parts,
            f"写入路径必须位于 step8：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(paths):
    return {Path(p).as_posix(): sha(p) for p in paths}


def check_hashes(mapping):
    for name, expected in mapping.items():
        require(Path(name).is_file() and sha(name) == expected, f"来源或输出已改变：{name}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = out(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(frame, path):
    path = out(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g", na_rep="NA")
    os.replace(tmp, path)


def save_torch(value, path):
    """先写同目录临时文件，完成后原子替换；旧 checkpoint 不会先被删除。"""
    path = out(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def versions():
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, statsmodels=statsmodels.__version__)


def condition_catalog():
    """相同条件只有一个 ID；跨实验引用该 ID，不重复训练或复制模型文件。"""
    conditions = {}
    def add(name, activation="LeakyReLU", bn="linear_bn_activation", dropout=.3,
            scope="full", deep=False):
        if name not in conditions:
            conditions[name] = dict(condition_id=name, condition_index=len(conditions),
                activation=activation, bn_position=bn, dropout=dropout, input_scope=scope,
                hidden_dims=[15000, 4000, 1000, 256] if deep else [2048, 512],
                latent_dim=64, epochs=100, batch_size=64, learning_rate=.0002,
                initialization="kaiming_uniform_a_0.2_leaky_relu", final_epoch=100)
        return name
    add(BASELINE)
    experiments = {"bn": [BASELINE, add("no_bn", bn="none")]}
    acts = []
    for activation in ("LeakyReLU", "Tanh", "Sigmoid", "ReLU", "Swish"):
        for position in ("linear_bn_activation", "linear_activation_bn"):
            name = BASELINE if activation == "LeakyReLU" and position == "linear_bn_activation" else f"{activation.lower()}_{position}"
            acts.append(add(name, activation, position))
    experiments["activation_bn"] = acts
    drops = []
    for rate in (0., .2, .4, .6, .8):
        for bn in ("none", "linear_bn_activation"):
            drops.append(add(f"dropout_{int(rate*10):02d}_{'no_bn' if bn == 'none' else 'bn'}", bn=bn, dropout=rate))
    experiments["dropout_bn"] = drops
    experiments["structure"] = [BASELINE, add("deep", deep=True), add("small", scope="small")]
    return conditions, experiments


def activation_module(name):
    # SiLU 与 beta=1 的 Swish 完全同式；显示名称仍为 Swish。
    return {"LeakyReLU": lambda: nn.LeakyReLU(.2), "ReLU": nn.ReLU,
            "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid, "Swish": nn.SiLU}[name]()


class Autoencoder(nn.Module):
    """瓶颈和最终输出均为 Linear；每个隐藏块按配置放置 BN、激活和 Dropout。"""
    def __init__(self, condition, input_dim):
        super().__init__()
        dims = [input_dim, *condition["hidden_dims"], condition["latent_dim"]]
        def stack(widths):
            layers = []
            for i, (a, b) in enumerate(zip(widths, widths[1:])):
                layers.append(nn.Linear(a, b))
                if i == len(widths)-2:
                    continue
                bn, act = nn.BatchNorm1d(b), activation_module(condition["activation"])
                position = condition["bn_position"]
                if position == "linear_bn_activation":
                    layers.extend([bn, act])
                elif position == "linear_activation_bn":
                    layers.extend([act, bn])
                else:
                    require(position == "none", "未知 BN 位置。")
                    layers.append(act)
                layers.append(nn.Dropout(condition["dropout"]))
            return nn.Sequential(*layers)
        self.encoder = stack(dims)
        self.decoder = stack(dims[::-1])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=.2, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def training_seed(condition, model_number):
    # 基线真实种子为630..649；新增条件使用独立且不重叠的命名空间。
    return MASTER_SEED + model_number if condition["condition_id"] == BASELINE else MASTER_SEED + 8000000 + condition["condition_index"] * 100 + model_number


def gsea_seed(condition, model_number, unit):
    return MASTER_SEED + 9000000 + condition["condition_index"] * 100000 + (model_number-1)*64 + unit


def runtime(device_name):
    require(sklearn.__version__ == "1.9.0", "scikit-learn 必须为1.9.0。")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    require(device.type in ("cpu", "cuda"), "仅支持 cpu/cuda。")
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA 不可用；请检查 trial_1 内核。")
    if device.type=="cuda" and device.index is None:
        device=torch.device("cuda",torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(np_state[0],np_state[1].tolist(),*np_state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(saved):
    random.setstate(saved["python"])
    s=saved["numpy"]
    np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"]:
        require(torch.cuda.is_available(), "CUDA checkpoint 不能静默改用 CPU 续训。")
        torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda"]])


def sample_inputs():
    full=read_csv("step1/core_metadata/sample_order_full.csv",keep_default_na=False)
    manifest=read_csv("step1/core_metadata/sample_manifest.csv",keep_default_na=False)
    train=read_csv("step1/core_metadata/sample_order_train.csv",keep_default_na=False)
    test=read_csv("step1/core_metadata/sample_order_test.csv",keep_default_na=False)
    require(len(full)==7862 and full.sample_id.is_unique and list(full.sample_id)==list(manifest.Sample), "完整样本 ID 不一致。")
    tr=train.full_data_position.to_numpy(dtype=np.int64)
    te=test.full_data_position.to_numpy(dtype=np.int64)
    require(len(tr)==6289 and len(te)==1573 and np.array_equal(np.sort(np.r_[tr,te]),np.arange(len(full))), "划分覆盖或重叠错误。")
    require(list(full.sample_id.iloc[tr])==list(train.sample_id) and list(full.sample_id.iloc[te])==list(test.sample_id), "划分的样本 ID 错位。")
    return manifest, tr, te


def reference_snapshot(scope, selected):
    """小输入只与既定 Top100 取交集，不补足、不重新排序选基因。"""
    features=read_csv("step7/reference/feature_to_gene.csv",keep_default_na=False).iloc[selected].copy()
    features["matrix_column"]=np.arange(len(features))
    names=sorted(features.gene_name.unique(),key=lambda v:v.encode("utf-8"))
    lookup={v:i for i,v in enumerate(names)}
    features["merged_column"]=features.gene_name.map(lookup)
    counts=np.bincount(features.merged_column,minlength=len(names))
    genes=pd.DataFrame(dict(merged_column=np.arange(len(names)),gene_name=names,n_features=counts))
    paths=read_json("step7/reference/pathways.json")
    universe=set(names)
    paths={tid:[gene for gene in members if gene in universe] for tid,members in paths.items()}
    sets=read_csv("step7/reference/reference_sets.csv",keep_default_na=False)
    sets["n_before_intersection"]=sets.n_matched
    sets["n_matched"]=sets.reference_tissue_id.map(lambda tid:len(paths[tid]))
    sets["n_removed_by_input_intersection"]=sets.n_before_intersection-sets.n_matched
    sets["input_scope"]=scope
    sets["gsea_eligible"]=sets.n_matched.between(15,500)
    require(len(sets)==53 and sets.gsea_eligible.all(), "参考集交集不足15个基因：需先决定组织范围，不能静默更改53组织 NGCI。")
    folder=ROOT / "reference" / scope
    write_csv(features,folder/"feature_to_gene.csv")
    write_csv(genes,folder/"gene_map.csv")
    write_csv(sets,folder/"reference_sets.csv")
    write_json(folder/"pathways.json",paths)
    return dict(n_features=len(features),n_genes=len(genes),min_set_size=int(sets.n_matched.min()))


def setup():
    """冻结输入指纹、条件、种子、参考交集及训练映射；不启动训练。"""
    conditions,experiments=condition_catalog()
    manifest,tr,te=sample_inputs()
    gene=read_csv("step1/core_metadata/gene_order.csv",keep_default_na=False)
    requested=read_csv("input_data/gene_name_19594.csv",header=None,keep_default_na=False)[0]
    selected=np.flatnonzero(gene.gene_name.isin(set(requested)).to_numpy())
    require(len(requested)==19594 and requested.nunique()==19558 and len(selected)==19666, "小输入名单与已确认的19666特征方案不同。")
    require(set(requested)<=set(gene.gene_name), "小输入名单含未匹配名称。")
    features=read_csv("step7/reference/feature_to_gene.csv",keep_default_na=False)
    require(len(gene)==60498 and np.array_equal(features.matrix_column,np.arange(60498)) and
            list(features.gene_name)==list(gene.gene_name) and list(features.gene_id)==list(gene.gene_id),
            "Step1与Step7的特征顺序不一致。")
    sample_map=read_csv("step2/raw_outputs/model_01/sample_map.csv",keep_default_na=False)
    require(np.array_equal(sample_map.row_index,np.arange(7862)) and
            np.array_equal(sample_map.full_data_position,np.arange(7862)) and
            list(sample_map.sample_id)==list(manifest.Sample) and list(sample_map.tissue_name)==list(manifest.tissue_name),
            "逐样本映射与Step1的表达矩阵顺序不一致。")
    require(sample_map.iloc[tr].split.eq("train").all() and sample_map.iloc[te].split.eq("test").all() and
            np.array_equal(sample_map.iloc[tr].order_in_split,np.arange(len(tr))) and
            np.array_equal(sample_map.iloc[te].order_in_split,np.arange(len(te))),"样本映射的划分或顺序错误。")
    f7=read_json("step7/quality_control/finalize.json")
    require(f7["status"]=="complete" and f7["n_ngci"]==1280, "Step7 未完整完成。")
    d=f7["configuration"]["design"]
    require(d["propagation_entry"]=="decoder[1:]" and d["mode"]=="eval" and
            d["reference_selection"]["id"]=="tissue_top100_mean_tpm_v1", "Step7 传播/参考设计不同。")
    check_hashes({k:v for k,v in f7["output_sha256"].items() if "/statistics/" in k})
    check_hashes(f7["configuration"]["code_sha256"])
    upstream=[MATRIX,*[Path("step1/core_metadata")/name for name in
        ("sample_order_full.csv","sample_order_train.csv","sample_order_test.csv","sample_manifest.csv","gene_order.csv","model_architecture.json","experiment_config.json")],
        Path("input_data/gene_name_19594.csv"),Path("step7/quality_control/finalize.json"),
        Path("step2/raw_outputs/model_01/sample_map.csv"),
        *[Path("step7/reference")/name for name in ("feature_to_gene.csv","gene_map.csv","reference_sets.csv","pathways.json","tissue_map.csv")],
        Path("step7/environment/r_environment_lock.json"),Path("step7/statistics/ngci_per_neuron.csv")]
    source=hashes(upstream)
    target=ROOT/"metadata/design.json"
    if target.exists():
        previous=read_json(target)
        require(previous["source_sha256"]==source and previous["conditions"]==conditions and previous["experiments"]==experiments,
                "冻结设计或输入已改变，不能覆盖既有实验。")
        verify_setup()
        print("setup 已完成；输入指纹一致。",flush=True)
        return
    refs={"full":reference_snapshot("full",np.arange(60498)),"small":reference_snapshot("small",selected)}
    write_csv(pd.DataFrame(dict(small_column=np.arange(len(selected)),full_column=selected,
                 gene_id=gene.gene_id.iloc[selected].to_numpy(),gene_name=gene.gene_name.iloc[selected].to_numpy())),ROOT/"metadata/small_feature_map.csv")
    write_csv(sample_map.drop(columns=["model_number","model_seed"]),ROOT/"metadata/sample_map.csv")
    out(ROOT/"reference/tissue_map.csv").write_bytes(Path("step7/reference/tissue_map.csv").read_bytes())
    out(ROOT/"environment/r_environment_lock.json").write_bytes(Path("step7/environment/r_environment_lock.json").read_bytes())
    seed_rows=[dict(condition_id=c,model_number=m,train_seed=training_seed(v,m),
        first_gsea_seed=(629+7000000+(m-1)*64 if c==BASELINE else gsea_seed(v,m,0)),
        last_gsea_seed=(629+7000000+(m-1)*64+63 if c==BASELINE else gsea_seed(v,m,63)),reused=c==BASELINE)
        for c,v in conditions.items() for m in range(1,21)]
    write_csv(pd.DataFrame(seed_rows),ROOT/"metadata/seeds.csv")
    write_json(ROOT/"environment/python_environment.json",versions())
    resource_report(conditions,refs)
    design=dict(schema=1,conditions=conditions,experiments=experiments,references=refs,
        source_sha256=source,parameters=PARAMETERS,small_input_rule="all exact-name matched features, expression order; 19666 features",
        ngci="signed Pearson r; weight ES; all 53 tissues equally weighted; two-sided; no p selection",
        bn_rule="two hidden BN blocks in encoder AND decoder for shallow; one per hidden block for deep",
        initialization="all Linear: kaiming_uniform_(a=0.2,nonlinearity='leaky_relu')",
        statistical_policy=dict(bn_reml=True,activation_bn_reml=False,dropout_bn_reml=False,structure_reml=True,
            activation_missing="only missing NGCI converted to zero for activation_bn LMM, raw values retained",
            activation_fisher_clip=[-.99999,.99999],fixed_effect_BH="within each experiment; non-intercept coefficients; structure pools two contrasts",
            gsea_BH="separate per condition across 20x64x53; baseline reuses Step7 family",alpha=.05),
        checkpoint_policy="batch-boundary stop; epoch and 300-second rolling saves; hard kill restarts last atomic save",
        baseline_reuse="Step1 epoch100 models; Step7 ES/NGCI; no baseline retraining",
        execution_order=list(EXPERIMENTS),n_unique_conditions=len(conditions),n_new_models=(len(conditions)-1)*20)
    write_json(target,design)
    snapshot=[p for folder in (ROOT/"reference",ROOT/"metadata",ROOT/"environment") for p in folder.rglob("*") if p.is_file()]
    write_json(ROOT/"quality_control/setup.json",dict(status="complete",output_sha256=hashes(snapshot)))
    print(f"setup 完成：{len(conditions)} 条件，440 个新模型；small={refs['small']}。",flush=True)


def verify_setup():
    report=read_json(ROOT/"quality_control/setup.json")
    require(report["status"]=="complete","请先运行 setup。")
    check_hashes(report["output_sha256"])
    design=read_json(ROOT/"metadata/design.json")
    check_hashes(design["source_sha256"])
    require(design["conditions"]==condition_catalog()[0],"代码条件与冻结设计不一致。")
    return design


def stop_handler(signum, frame):
    global _stop_requested
    _stop_requested=True
    print("已收到停止请求；当前批次结束后保存 checkpoint，请勿强制结束进程。",flush=True)


@contextlib.contextmanager
def training_lock():
    """OS 文件锁在进程结束后自动释放，不需要删除未知的活动锁。"""
    path=out(ROOT/"control/train.lock")
    stream=path.open("a+b")
    if path.stat().st_size==0:
        stream.write(b"0");stream.flush()
    stream.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("已有 Step8 训练进程；不要并发写入 checkpoint。")
    try:
        yield
    finally:
        stream.seek(0)
        if os.name=="nt":
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()


def memory_preflight(condition, input_dim, device, training=True):
    """只创建 meta 形状计算内存下限，禁止因 OOM 静默改维度/精度/批量。"""
    with torch.device("meta"):
        model=Autoencoder(condition,input_dim)
    count=sum(p.numel() for p in model.parameters())
    if condition['condition_id']=='deep' and device.type=='cuda':
        from streamed_training import preflight
        preflight(count,device,training=training,
                  largest_linear_elements=max(m.weight.numel() for m in model.modules() if isinstance(m,nn.Linear)))
        return count
    estimated=count*(16 if training else 4)*1.15 + 512*2**20
    if device.type=="cuda":
        free,total=torch.cuda.mem_get_info(device)
        require(estimated<free, f"{condition['condition_id']} 预计至少需 {estimated/2**30:.1f} GiB 可用显存，当前 {free/2**30:.1f} GiB。请使用更大显存设备，或内存足够时显式 --device cpu；不改变结构/float32/batch size。")
    elif training:
        import psutil
        require(estimated<psutil.virtual_memory().available,
                f"{condition['condition_id']} 预计至少需 {estimated/2**30:.1f} GiB 可用内存；当前不足，不能启动深层训练。")
    # 活动 checkpoint 与其临时副本，最终权重及日志均预留空间。
    needed=count*(28 if training else 4)+2**30
    require(shutil.disk_usage(ROOT).free>needed,"磁盘可用空间不足以原子保存当前模型。")
    return count


def resource_report(conditions,references):
    """记录最低参数存储量与保守运行估算；不要求一次腾出全部实验的空间。"""
    rows=[];architectures={}
    for cid,c in conditions.items():
        dim=references[c["input_scope"]]["n_features"]
        with torch.device("meta"):
            model=Autoencoder(c,dim)
        parameters=sum(p.numel() for p in model.parameters())
        buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers())
        state_bytes=parameters*4+buffer_bytes
        architectures[cid]=dict(input_dim=dim,latent_dim=64,
            encoder_layers=[str(layer) for layer in model.encoder],
            decoder_layers=[str(layer) for layer in model.decoder],
            initialization=c["initialization"],bn_count_encoder=sum(isinstance(x,nn.BatchNorm1d) for x in model.encoder),
            bn_count_decoder=sum(isinstance(x,nn.BatchNorm1d) for x in model.decoder))
        rows.append(dict(condition_id=cid,input_dim=dim,n_parameters=parameters,
            state_dict_min_gib=state_bytes/2**30,
            float32_training_state_min_gib=parameters*16/2**30,
            training_preflight_estimate_gib=(parameters*16*1.15+512*2**20)/2**30,
            new_model_count=0 if cid==BASELINE else 20,
            new_final_state_dict_min_gib=0 if cid==BASELINE else 20*state_bytes/2**30))
    frame=pd.DataFrame(rows)
    write_csv(frame,ROOT/"metadata/resource_estimates.csv")
    write_json(ROOT/"metadata/architectures.json",architectures)
    minimum=float(frame.new_final_state_dict_min_gib.sum())
    free=shutil.disk_usage(ROOT).free/2**30
    write_json(ROOT/"quality_control/resources_at_setup.json",dict(
        free_disk_gib=free,new_final_state_dict_min_gib=minimum,
        enough_disk_for_all_final_weights=free>minimum,
        note="Final weights only; scores, logs and active atomic Adam checkpoint require additional space."))
    print(f"全部新增模型的最终权重至少 {minimum:.1f} GiB；当前可用磁盘 {free:.1f} GiB。分析输出和活动断点另占空间。",flush=True)


def initial_progress():
    return dict(epoch=1,phase="train",cursor=0,order=None,train_total=0.,test_total=0.,history=[])


def train_loop(model,optimizer,matrix,train_indices,test_indices,progress,generator,device,
               save,should_stop,epochs=100,batch_size=64,save_seconds=300.,after_batch=None):
    """游标指向下一批；所有停点均在完整的 forward/backward/Adam step 之后。

    测试阶段也记录游标和累计损失。恢复时不重复已提交批次，也不丢弃 Dropout
    RNG 状态。强制终止最多回退到最近一次完整 checkpoint，而非整个模型。
    """
    last_save=time.monotonic()
    while progress["epoch"]<=epochs:
        if should_stop():
            save(progress,generator);return False
        if progress["order"] is None:
            # 保留 DataLoader 的独立打乱 RNG；一次生成并保存整个 epoch 顺序。
            progress["order"]=torch.randperm(len(train_indices),generator=generator)
        training=progress["phase"]=="train"
        model.train(training)
        indices=train_indices[progress["order"].cpu().numpy()] if training else test_indices
        for start in range(progress["cursor"],len(indices),batch_size):
            batch_started=time.monotonic()
            positions=indices[start:start+batch_size]
            x=torch.from_numpy(np.asarray(matrix[positions],dtype=np.float32)).to(device)
            with torch.set_grad_enabled(training):
                latent,recon=model(x)
                loss=nn.functional.mse_loss(recon,x)
                require(bool(torch.isfinite(loss)),f"非有限损失：epoch={progress['epoch']}, phase={progress['phase']}")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"梯度包含 NaN/Inf，停止且保留上一次完整 checkpoint。")
                    optimizer.step()
            progress["train_total" if training else "test_total"]+=float(loss.detach().cpu())*len(positions)
            progress["cursor"]=start+len(positions)
            if getattr(optimizer,'streamed',False):
                print(f"stream epoch={progress['epoch']:03d} {progress['phase']} "
                      f"rows={progress['cursor']}/{len(indices)} loss={float(loss.detach().cpu()):.8f} "
                      f"seconds={time.monotonic()-batch_started:.1f}",flush=True)
            if after_batch:
                after_batch(progress)
            del x,latent,recon,loss
            if should_stop():
                save(progress,generator);return False
            if time.monotonic()-last_save>=save_seconds:
                save(progress,generator);last_save=time.monotonic()
        if training:
            progress.update(phase="test",cursor=0)
        else:
            progress["history"].append(dict(epoch=progress["epoch"],train_loss=progress["train_total"]/len(train_indices),
                test_loss=progress["test_total"]/len(test_indices)))
            row=progress["history"][-1]
            print(f"epoch={row['epoch']:03d}/{epochs} train={row['train_loss']:.8f} test={row['test_loss']:.8f}",flush=True)
            progress.update(epoch=progress["epoch"]+1,phase="train",cursor=0,order=None,train_total=0.,test_total=0.)
            save(progress,generator);last_save=time.monotonic()
    return True


def train_one(condition,number,matrix,tr,te,device,design,save_seconds):
    cid=condition["condition_id"]
    folder=ROOT/"models"/cid/f"model_{number:02d}"
    completion=folder/"complete.json"
    if completion.exists():
        report=read_json(completion)
        require(report["status"]=="complete","模型完成记录无效。")
        check_hashes(report["output_sha256"])
        require(report["condition"]==condition,"已完成模型条件改变。")
        require(report["configuration"]["design_sha256"]==sha(ROOT/"metadata/design.json") and
                report["configuration"]["seed"]==training_seed(condition,number),"已完成模型的设计或种子不同。")
        print(f"跳过已完成 {cid}/model_{number:02d}",flush=True);return True
    input_dim=matrix.shape[1]
    memory_preflight(condition,input_dim,device)
    seed=training_seed(condition,number)
    seed_all(seed)
    configuration=dict(condition=condition,input_dim=input_dim,model_number=number,seed=seed,
        design_sha256=sha(ROOT/"metadata/design.json"),code_sha256=sha(CODE),versions=versions(),
        device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
        precision="float32; no AMP; no TF32",deterministic=True)
    streamed=cid=='deep' and device.type=='cuda'
    if streamed:
        from streamed_training import (BACKEND,CHUNK_ELEMENTS,DiskState,StreamedAdam,install_streaming,
                                       checkpoint_metadata,load_model_checkpoint,materialize_metadata)
        configuration['execution_backend']=dict(id=BACKEND,chunk_elements=CHUNK_ELEMENTS,
            code_sha256=sha(ROOT/'code/streamed_training.py'),batch_partitioned=False,
            parameters='CPU float32',gradients_and_moments='disk float32',compute='CUDA float32')
    resume=folder/"resume.pt"
    generator=torch.Generator().manual_seed(seed)
    model=Autoencoder(condition,input_dim)
    if streamed:
        disk=DiskState(folder/'stream_work',model.named_parameters())
        install_streaming(model,device,disk)
        optimizer=StreamedAdam(model,disk,device,lr=condition['learning_rate'])
    else:
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=condition["learning_rate"],foreach=False)
    progress=initial_progress()
    if resume.exists():
        # 深层只读张量元数据，再逐片恢复；不整体mmap巨型文件，避免Windows提交额度耗尽。
        # 常规模型仍使用普通加载，防止CPU Adam标量引用映射而阻止原子替换。
        saved=checkpoint_metadata(resume) if streamed else torch.load(resume,map_location="cpu",weights_only=True)
        require(saved["configuration"]==configuration,"checkpoint 的输入/代码/环境/设备与当前不同，拒绝无提示续训。")
        if streamed:
            load_model_checkpoint(model,resume,saved['model_state_dict'])
            optimizer.load_state_dict(saved['optimizer_state_dict'],resume)
            for key in ('progress','shuffle_rng','rng'):
                saved[key]=materialize_metadata(resume,saved[key])
        else:
            model.load_state_dict(saved["model_state_dict"],strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        progress=saved["progress"]
        generator.set_state(saved["shuffle_rng"].cpu())
        restore_rng(saved["rng"])
        del saved
        if streamed:
            import gc
            gc.collect()
        print(f"恢复 {cid}/{number}: epoch={progress['epoch']}, {progress['phase']}, next_row={progress['cursor']}",flush=True)
    def save(p,g):
        if device.type=="cuda":torch.cuda.synchronize(device)
        if streamed:
            with optimizer.checkpoint_state() as state:
                save_torch(dict(configuration=configuration,model_state_dict=model.state_dict(),optimizer_state_dict=state,
                    progress=copy.deepcopy(p),shuffle_rng=g.get_state(),rng=rng_state()),resume)
        else:
            save_torch(dict(configuration=configuration,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                progress=copy.deepcopy(p),shuffle_rng=g.get_state(),rng=rng_state()),resume)
        history=pd.DataFrame(p["history"],columns=["epoch","train_loss","test_loss"])
        history["model_number"],history["model_seed"]=number,seed
        write_csv(history,folder/"loss.csv")
        write_json(folder/"progress.json",dict(epoch=p["epoch"],phase=p["phase"],next_row=p["cursor"],
            completed_epochs=len(p["history"]),status="paused" if _stop_requested or STOP.exists() else "running",
            checkpoint=resume.as_posix(),configuration=configuration))
    print(f"训练 {cid}/model_{number:02d}, seed={seed}, device={device}",flush=True)
    finished=train_loop(model,optimizer,matrix,tr,te,progress,generator,device,save,
        lambda:_stop_requested or STOP.exists(),epochs=100,batch_size=64,save_seconds=save_seconds)
    if finished:
        final=folder/"epoch_100.pt"
        save_torch(dict(model_state_dict=model.state_dict(),condition=condition,configuration=configuration,
            input_dim=input_dim,model_number=number,model_seed=seed,final_epoch=100),final)
        write_json(completion,dict(status="complete",condition=condition,configuration=configuration,
            output_sha256=hashes([final,folder/"loss.csv"])))
        # 最终 state_dict 已完成并有指纹；不保留440份巨大的 Adam 状态。
        resume.unlink()
        if streamed:
            disk.remove_work_files()
        write_json(folder/"progress.json",dict(status="complete",completed_epochs=100,checkpoint=final.as_posix()))
    del optimizer,model
    if device.type=="cuda":torch.cuda.empty_cache()
    return finished


def selected_conditions(design,experiment,condition=None):
    names=design["experiments"][experiment]
    if condition:
        require(condition in names,f"{condition} 不属于 {experiment}")
        names=[condition]
    return [design["conditions"][name] for name in names]


def train(args):
    global _stop_requested
    design=verify_setup()
    device=runtime(args.device)
    conditions=selected_conditions(design,args.experiment,args.condition)
    _,tr,te=sample_inputs()
    with training_lock():
        # 新的一次显式 train 命令表示继续；只在获得独占锁后清除旧停止标志。
        if STOP.exists():STOP.unlink()
        _stop_requested=False
        old=signal.signal(signal.SIGINT,stop_handler)
        matrix=None
        write_json(ROOT/"control/active_train.json",dict(pid=os.getpid(),status="running",experiment=args.experiment,
            condition=args.condition,model_start=args.model_start,model_end=args.model_end,device=str(device)))
        outcome="complete"
        error_message=None
        try:
            for c in conditions:
                if c["condition_id"]==BASELINE:
                    print("baseline 直接引用 Step1/Step7，不训练。",flush=True);continue
                if matrix is None:
                    matrix=np.load(MATRIX,allow_pickle=False)
                    require(matrix.shape==(7862,60498) and matrix.dtype==np.float16,"预处理矩阵不匹配。")
                current=matrix
                if c["input_scope"]=="small":
                    columns=read_csv(ROOT/"metadata/small_feature_map.csv").full_column.to_numpy()
                    current=matrix[:,columns]
                for number in range(args.model_start,args.model_end+1):
                    if not train_one(c,number,current,tr,te,device,design,args.save_seconds):
                        outcome="paused"
                        print("训练已安全暂停；再次运行同一 train 命令即可续训。",flush=True);return
        except BaseException as error:
            outcome="failed"
            error_message=f"{type(error).__name__}: {error}"
            raise
        finally:
            signal.signal(signal.SIGINT,old)
            write_json(ROOT/"control/active_train.json",dict(pid=os.getpid(),status=outcome,
                experiment=args.experiment,stopped=_stop_requested or STOP.exists(),error=error_message))


def model_checkpoint(condition,number):
    """基线文件保持只读；新条件只能使用本步骤的 epoch100 完成记录。"""
    cid=condition["condition_id"]
    if cid==BASELINE:
        path=Path(f"step1/core_models/model_{number:02d}.pt")
        prepared=read_json(f"step7/quality_control/prepared/model_{number:02d}.json")
        require(prepared["status"]=="complete" and sha(path)==prepared["audit"]["checkpoint_sha256"],
                "基线 checkpoint 与 Step7 来源不符。")
    else:
        folder=ROOT/"models"/cid/f"model_{number:02d}"
        report=read_json(folder/"complete.json")
        require(report["status"]=="complete" and report["condition"]==condition,"模型未完成或条件改变。")
        check_hashes(report["output_sha256"])
        path=folder/"epoch_100.pt"
    if cid=='deep':
        from streamed_training import checkpoint_metadata
        saved=checkpoint_metadata(path)
    else:
        saved=torch.load(path,map_location="cpu",mmap=True,weights_only=True)
    require(saved["final_epoch"]==100 and saved["model_number"]==number and
            saved["model_seed"]==training_seed(condition,number),"checkpoint 的 epoch/模型/种子错误。")
    return path,saved


def load_analysis_model(condition,number,input_dim,device):
    path,saved=model_checkpoint(condition,number)
    with torch.device("meta"):
        model=Autoencoder(condition,input_dim)
    if condition['condition_id']=='deep':
        from streamed_training import load_model_checkpoint
        model.to_empty(device='cpu')
        load_model_checkpoint(model,path,saved['model_state_dict'])
    else:
        # clone使分析副本和只读磁盘存储独立，归零偏置不能修改checkpoint。
        weights={k:v.clone() for k,v in saved["model_state_dict"].items()}
        require(all(bool(torch.isfinite(v).all()) for v in weights.values()),"模型参数含非有限值。")
        model.load_state_dict(weights,strict=True,assign=True)
    if condition['condition_id']=='deep' and device.type=='cuda':
        from streamed_training import install_streaming
        install_streaming(model,device)
    else:
        model.to(device)
    model.eval().requires_grad_(False)
    return model,path


def propagate_weight(decoder):
    """一次传播全部64行带符号入口；自动适配无BN/BN位置/深度，不硬编码隐藏索引。"""
    require(not decoder.training and isinstance(decoder[0],nn.Linear) and
            decoder[0].in_features==64,"Decoder入口必须以64维瓶颈为输入。")
    before={name:buffer.detach().clone() for name,buffer in decoder.named_buffers()}
    gamma={name:layer.weight.detach().clone() for name,layer in decoder.named_modules()
           if isinstance(layer,nn.BatchNorm1d)}
    calls=[0]*len(decoder)
    def hook(index):
        def mark(module,inputs,output):calls[index]+=1
        return mark
    hooks=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(decoder)]
    try:
        with torch.no_grad():
            for layer in decoder.modules():
                if isinstance(layer,(nn.Linear,nn.BatchNorm1d)) and layer.bias is not None:
                    layer.bias.zero_()
            entry=10000*decoder[0].weight.T.contiguous()
            if hasattr(decoder[0],'compute_device'):
                entry=entry.to(decoder[0].compute_device)
            scores=decoder[1:](entry)
    finally:
        for h in hooks:h.remove()
    require(calls==[0]+[1]*(len(decoder)-1),"传播路径重复Linear或跳过了BN/激活。")
    require(all(torch.equal(before[name],buffer) for name,buffer in decoder.named_buffers()),"BN缓冲区在eval传播中改变。")
    require(all(torch.equal(gamma[name],layer.weight) for name,layer in decoder.named_modules()
                if isinstance(layer,nn.BatchNorm1d)),"BN gamma 被修改。")
    require(scores.shape[0]==64 and bool(torch.isfinite(scores).all()),"权重评分无效。")
    return entry.cpu().numpy(),scores.cpu().numpy(),dict(module_call_counts=calls,bn_buffers_unchanged=True,
                bn_gamma_unchanged=True,signed_entry=True,batch_size=64,mode="eval",gradient_recording=False)


def merge_gene_scores(raw,features,genes):
    """精确名称合并只在评分阶段进行，按float64求和和除法。"""
    columns=features.merged_column.to_numpy(dtype=np.int64)
    counts=genes.n_features.to_numpy(dtype=np.int64)
    require(raw.shape[1]==len(features),"基因特征数不一致。")
    return np.stack([np.bincount(columns,weights=np.asarray(row,dtype=np.float64),minlength=len(genes))/counts for row in raw])


def prepare_one(condition,number,matrix,manifest,device):
    cid=condition["condition_id"]
    if cid==BASELINE:
        model_checkpoint(condition,number)
        print(f"复用 baseline/model_{number:02d} 的 Step7 ES/NGCI。",flush=True);return
    dim=matrix.shape[1]
    path,_=model_checkpoint(condition,number)
    configuration=dict(condition=condition,number=number,checkpoint_sha256=sha(path),
        design_sha256=sha(ROOT/"metadata/design.json"),code_sha256=sha(CODE),versions=versions(),device=str(device))
    if cid=='deep' and device.type=='cuda':
        from streamed_training import BACKEND,CHUNK_ELEMENTS
        configuration['execution_backend']=dict(id=BACKEND,chunk_elements=CHUNK_ELEMENTS,
            code_sha256=sha(ROOT/'code/streamed_training.py'))
    folder=ROOT/"rank_inputs"/cid/f"model_{number:02d}"
    report=ROOT/"quality_control/prepared"/cid/f"model_{number:02d}.json"
    if report.exists():
        previous=read_json(report)
        require(previous["configuration"]==configuration,"prepare输入/实现/设备改变，禁止混合新旧结果。")
        if previous["status"]=="complete":
            check_hashes(previous["output_sha256"])
            print(f"跳过已完成 prepare {cid}/{number}",flush=True);return
    memory_preflight(condition,dim,device,training=False)
    write_json(report,dict(status="running",configuration=configuration))
    model,_=load_analysis_model(condition,number,dim,device)
    # 先用完整训练参数提取激活，再对独立分析副本的Decoder偏置归零。
    latent=np.empty((len(matrix),64),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(matrix),64):
            x=torch.from_numpy(np.asarray(matrix[start:start+64],dtype=np.float32)).to(device)
            latent[start:start+len(x)]=model.encoder(x).cpu().numpy()
    require(np.isfinite(latent).all(),"瓶颈输出含非有限值。")
    tissue_map=read_csv(ROOT/"reference/tissue_map.csv",keep_default_na=False)
    medians=[]
    for row in tissue_map.itertuples():
        indices=np.flatnonzero(manifest.tissue_name.to_numpy()==row.tissue_name)
        require(len(indices)==row.n_samples,"组织样本数不一致。")
        medians.append(dict(tissue_id=row.tissue_id,tissue_name=row.tissue_name,n_samples=len(indices),
                       **{f"unit_{u:05d}":float(x) for u,x in enumerate(np.median(latent[indices],axis=0))}))
    entry,raw,audit=propagate_weight(model.decoder)
    reference=ROOT/"reference"/condition["input_scope"]
    genes=read_csv(reference/"gene_map.csv",keep_default_na=False)
    features=read_csv(reference/"feature_to_gene.csv",keep_default_na=False)
    merged=merge_gene_scores(raw,features,genes)
    require(np.isfinite(merged).all(),"合并评分含非有限值。")
    outputs=[folder/name for name in ("bottleneck.npy","tissue_medians.csv","entry.npy","scores.f64.bin","jobs.csv","score_qc.csv")]
    np.save(out(outputs[0]),latent,allow_pickle=False)
    write_csv(pd.DataFrame(medians),outputs[1])
    np.save(out(outputs[2]),entry,allow_pickle=False)
    merged.astype("<f8").tofile(out(outputs[3]))
    jobs=pd.DataFrame(dict(condition_id=cid,model_number=number,matrix_row=np.arange(64),neuron_index=np.arange(64),
        unit_id=[f"unit_{u:05d}" for u in range(64)],seed=[gsea_seed(condition,number,u) for u in range(64)]))
    write_csv(jobs,outputs[4])
    qc=jobs[["unit_id"]].copy()
    qc["min"],qc["max"]=merged.min(1),merged.max(1)
    qc["n_positive"],qc["n_negative"],qc["n_zero"]=(merged>0).sum(1),(merged<0).sum(1),(merged==0).sum(1)
    qc["n_unique"]=[len(np.unique(row)) for row in merged]
    write_csv(qc,outputs[5])
    sources=hashes([path,ROOT/"metadata/design.json",ROOT/"metadata/sample_map.csv",CODE,
                   *[reference/name for name in ("gene_map.csv","feature_to_gene.csv","reference_sets.csv","pathways.json")]])
    write_json(report,dict(status="complete",configuration=configuration,condition_id=cid,model_number=number,
        n_genes=len(genes),n_features=dim,n_neurons=64,binary_dtype="little-endian float64",
        sample_map="step8/metadata/sample_map.csv",source_sha256=sources,output_sha256=hashes(outputs),audit=audit))
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    print(f"prepare 完成 {cid}/model_{number:02d}",flush=True)


def prepare(args):
    design=verify_setup()
    device=runtime(args.device)
    manifest,_,_=sample_inputs()
    # 只读映射避免推理复制完整矩阵到内存；训练阶段使用一次性内存加载。
    matrix=np.load(MATRIX,mmap_mode="r",allow_pickle=False)
    for c in selected_conditions(design,args.experiment,args.condition):
        current=matrix
        if c["input_scope"]=="small":
            current=matrix[:,read_csv(ROOT/"metadata/small_feature_map.csv").full_column.to_numpy()]
        for number in range(args.model_start,args.model_end+1):
            prepare_one(c,number,current,manifest,device)


def bh(pvalues):
    """计划检验数包含缺失值；缺失p保持NA，绝不以NGCI缺失处理规则替换。"""
    values=np.asarray(pvalues,dtype=float)
    finite=np.flatnonzero(np.isfinite(values))
    require(np.all((values[finite]>=0)&(values[finite]<=1)),"p值超出[0,1]。")
    result=np.full(len(values),np.nan)
    if len(finite):
        order=finite[np.argsort(values[finite],kind="stable")]
        ranked=values[order]*len(values)/np.arange(1,len(order)+1)
        result[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1])
    return result


def ngci_result(activation,es):
    """n固定53；只要ES有效，即使GSEA的p/NES缺失仍保留组织。"""
    x,y=np.asarray(activation,dtype=float),np.asarray(es,dtype=float)
    require(len(x)==len(y)==53,"NGCI组织数必须为53。")
    result=dict(n_tissues=53,ngci=np.nan,pearson_r=np.nan,pearson_p=np.nan,status="ok")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        result["status"]="missing_activation_or_ES";return result
    if np.ptp(x)==0 or np.ptp(y)==0:
        result["status"]="constant_activation_or_ES";return result
    fitted=stats.pearsonr(x,y,alternative="two-sided")
    result.update(ngci=float(fitted.statistic),pearson_r=float(fitted.statistic),pearson_p=float(fitted.pvalue))
    return result


def finalize_condition(condition):
    cid=condition["condition_id"]
    folder=ROOT/"statistics/conditions"/cid
    sources={}
    if cid==BASELINE:
        final=read_json("step7/quality_control/finalize.json")
        names=["step7/statistics/matched_gsea.csv","step7/statistics/ngci_per_neuron.csv"]
        check_hashes({name:final["output_sha256"][name] for name in names})
        table=read_csv(names[0]);neurons=read_csv(names[1])
        require(neurons.shape[0]==1280 and neurons.n_tissues.eq(53).all(),"基线NGCI范围错误。")
        table=table.loc[table.reference_available].copy()
        require(len(table)==67840,"基线GSEA缺失。")
        sources.update(hashes(names+["step7/quality_control/finalize.json"]))
    else:
        tables=[];neuron_rows=[]
        refs=read_csv(ROOT/"reference"/condition["input_scope"]/"reference_sets.csv",keep_default_na=False)
        for number in range(1,21):
            report_path=ROOT/"quality_control/fgsea"/cid/f"model_{number:02d}.json"
            report=read_json(report_path)
            require(report["status"]=="complete" and report["n_tests"]==64*53,"GSEA模型未完整完成。")
            check_hashes(report["source_sha256"]);check_hashes(report["output_sha256"])
            prepared_path=ROOT/"quality_control/prepared"/cid/f"model_{number:02d}.json"
            prepared=read_json(prepared_path)
            check_hashes(prepared["source_sha256"]);check_hashes(prepared["output_sha256"])
            result_path=ROOT/"gsea_results"/cid/f"model_{number:02d}.csv.gz"
            t=read_csv(result_path)
            require(len(t)==3392 and t.model_number.eq(number).all() and t.condition_id.eq(cid).all() and
                not t.duplicated(["neuron_index","reference_tissue_id"]).any() and
                set(t.neuron_index)==set(range(64)),"GSEA主键范围错误。")
            require((t.unit_id==t.neuron_index.map(lambda u:f"unit_{u:05d}")).all() and
                    t.score_source.eq("weight").all(),"GSEA神经元编号或路线来源不一致。")
            med_path=ROOT/"rank_inputs"/cid/f"model_{number:02d}"/"tissue_medians.csv"
            med=read_csv(med_path).set_index("tissue_id")
            pairs=[]
            for unit,part in t.groupby("neuron_index",sort=True):
                require(set(part.reference_tissue_id)==set(refs.reference_tissue_id),"NGCI参考组织不完整。")
                part=part.set_index("reference_tissue_id").loc[refs.reference_tissue_id].reset_index()
                activation=med.loc[part.reference_tissue_id,f"unit_{unit:05d}"].to_numpy()
                part["activation_median"]=activation
                part["n_samples"]=med.loc[part.reference_tissue_id,"n_samples"].to_numpy()
                pairs.append(part)
                neuron_rows.append(dict(model_number=number,neuron_index=int(unit),unit_id=f"unit_{unit:05d}",
                    score_source="weight",gsea_score="ES",**ngci_result(activation,part.ES)))
            tables.append(pd.concat(pairs,ignore_index=True))
            sources.update(hashes([report_path,result_path,med_path]))
        table=pd.concat(tables,ignore_index=True)
        neurons=pd.DataFrame(neuron_rows)
        table=table.rename(columns={"padj":"q_within_ranking"})
        table["q_matched_global"]=bh(table.pval)
    table["condition_id"],neurons["condition_id"]=cid,cid
    table["correction_family"]=f"weight_only_{cid}_20x64x53"
    table["significant"]=table.q_matched_global.lt(.05)
    neurons["p_value_used_for_selection"]=False
    neurons["p_lt_0_05"]=neurons.pearson_p.lt(.05)
    write_csv(table,folder/"matched_gsea.csv")
    write_csv(neurons,folder/"ngci_per_neuron.csv")
    # 每个条件独立汇总，跨模型不假定神经元编号具有同一生物学含义。
    coverage=table.groupby(["model_number","reference_tissue_id"],sort=True).agg(
        n_significant=("significant","sum"),n_tested=("significant","size"),n_finite_p=("pval","count")).reset_index()
    coverage["fraction_significant"]=coverage.n_significant/coverage.n_tested
    coverage["detected"]=coverage.n_significant.gt(0)
    stable=coverage.groupby("reference_tissue_id",sort=True).agg(n_models=("model_number","nunique"),
        n_models_detected=("detected","sum"),mean_fraction_significant=("fraction_significant","mean")).reset_index()
    stable["fraction_models_detected"]=stable.n_models_detected/stable.n_models
    write_csv(coverage,folder/"stability_per_model.csv")
    write_csv(stable,folder/"stability_across_models.csv")
    outputs=[folder/name for name in ("matched_gsea.csv","ngci_per_neuron.csv","stability_per_model.csv","stability_across_models.csv")]
    write_json(ROOT/"quality_control/finalize_conditions"/f"{cid}.json",dict(status="complete",condition=condition,
        n_primary_tests=len(table),n_ngci=len(neurons),n_valid_ngci=int(neurons.ngci.notna().sum()),
        n_missing_p=int(table.pval.isna().sum()),source_sha256=sources,output_sha256=hashes(outputs)))
    return neurons


def statistical_input(neurons,experiment,conditions):
    """保留原始r、缺失、替换和Fisher边界处理列，确保每次替换可审计。"""
    table=neurons.copy()
    table["model_id"]=table.condition_id+"/model_"+table.model_number.astype(str).str.zfill(2)
    table["activation"]=table.condition_id.map(lambda c:conditions[c]["activation"])
    table["bn_position"]=table.condition_id.map(lambda c:conditions[c]["bn_position"])
    table["bn"]=table.bn_position.ne("none").astype(int)
    table["dropout_factor"]=table.condition_id.map(lambda c:str(conditions[c]["dropout"]))
    table["ngci_raw"]=table.ngci
    table["ngci_for_lmm"]=table.ngci
    table["missing_ngci_replaced_with_zero"]=False
    table["fisher_boundary_clipped"]=False
    if experiment=="activation_bn":
        missing=table.ngci.isna()
        table.loc[missing,"ngci_for_lmm"]=0.
        table.loc[missing,"missing_ngci_replaced_with_zero"]=True
        table["fisher_boundary_clipped"]=table.ngci_for_lmm.abs().gt(.99999)
        table["ngci_for_lmm"]=table.ngci_for_lmm.clip(-.99999,.99999)
    valid=table.ngci_for_lmm.notna()
    require(table.loc[valid,"ngci_for_lmm"].between(-1,1,inclusive="neither").all(),
            "Fisher z 遇到±1；该子实验没有授权边界替换，停止统计。")
    table["z_coef"]=np.arctanh(table.ngci_for_lmm)
    table["included_in_lmm"]=np.isfinite(table.z_coef)
    # 未定义相关只在激活函数子实验按批准规则置0；其他子实验显式记录排除。
    return table


def fit_lmm(data,formula,reml,comparison):
    """神经元嵌套在模型内；模型为独立训练重复，不对不同模型的同编号神经元配对。"""
    used=data.loc[data.included_in_lmm].copy()
    require(used.model_id.nunique()>=2 and len(used)>used.model_id.nunique(),"LMM有效层级数据不足。")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted=smf.mixedlm(formula,used,groups=used.model_id,re_formula="1").fit(reml=reml)
    ci=fitted.conf_int()
    coefficients=[]
    for name,beta in fitted.fe_params.items():
        coefficients.append(dict(comparison=comparison,term=name,estimate=float(beta),
            standard_error=float(fitted.bse_fe[name]),wald_z=float(fitted.tvalues[name]),p_value=float(fitted.pvalues[name]),
            ci95_low=float(ci.loc[name,0]),ci95_high=float(ci.loc[name,1]),scale="Fisher_z",
            n_observations=int(fitted.nobs),n_models=int(used.model_id.nunique()),estimation="REML" if reml else "ML"))
    variance=float(fitted.cov_re.iloc[0,0]);residual=float(fitted.scale)
    diagnostic=dict(comparison=comparison,formula=formula,reml=reml,converged=bool(fitted.converged),
        n_observations=int(fitted.nobs),n_models=int(used.model_id.nunique()),model_random_intercept_variance=variance,
        residual_variance=residual,ICC=variance/(variance+residual),
        warnings=[str(w.message) for w in caught],log_likelihood=float(fitted.llf),
        note="Coefficients and CI are on Fisher-z scale; not Pearson-r increments or Cohen d.")
    if not fitted.converged:
        error=RuntimeError(f"{comparison} LMM未收敛；诊断已记录，不输出完成的推断。")
        error.diagnostic=diagnostic
        raise error
    frame=pd.DataFrame(coefficients)
    require(np.isfinite(frame[["estimate","standard_error","p_value","ci95_low","ci95_high"]]).all().all(),
            f"{comparison} LMM固定效应包含非有限值。")
    return frame,diagnostic,fitted


def interaction_means(fitted,table):
    """固定效应的条件均值和Wald区间；使用完整协方差，不相加各系数CI。

    random intercept 设为0。tanh(z)是变换后的拟合位置，不是原始r的算术均值。
    这些区间不是新模型或单个神经元的预测区间，不增加显著性检验。
    """
    # 直接取已拟合模型中每个条件的第一行设计矩阵，保持其精确编码与列顺序。
    # 不重新解释公式，避免statsmodels公式后端/版本改变导致design_info接口不兼容。
    used=table.loc[table.included_in_lmm]
    require(list(fitted.model.data.row_labels)==list(used.index),"LMM设计矩阵的行与统计输入不一致。")
    positions=np.flatnonzero(~used.condition_id.duplicated().to_numpy())
    cells=used.iloc[positions][["condition_id","activation","bn_position","bn","dropout_factor"]].copy()
    matrix=np.asarray(fitted.model.exog)[positions]
    names=fitted.fe_params.index
    covariance=fitted.cov_params().loc[names,names].to_numpy()
    estimate=matrix@fitted.fe_params.to_numpy()
    variance=np.einsum("ij,jk,ik->i",matrix,covariance,matrix)
    require(np.all(variance>=-1e-12),"固定效应条件均值的方差为负。")
    se=np.sqrt(np.maximum(variance,0))
    cells["estimate_z"],cells["se_z"]=estimate,se
    cells["ci95_low_z"],cells["ci95_high_z"]=estimate-1.959963984540054*se,estimate+1.959963984540054*se
    cells["fitted_r"]=np.tanh(estimate)
    cells["ci95_low_r"],cells["ci95_high_r"]=np.tanh(cells.ci95_low_z),np.tanh(cells.ci95_high_z)
    return cells


def plot_interaction(cells,experiment,folder):
    """交互图用固定效应拟合位置与95%CI；x为离散实验条件。"""
    fig,ax=plt.subplots(figsize=(7,4.5))
    if experiment=="activation_bn":
        field="activation";order=["LeakyReLU","Tanh","Sigmoid","ReLU","Swish"]
        groups=[("linear_bn_activation","BN before activation"),("linear_activation_bn","BN after activation")]
        group_field="bn_position";xlabel="Activation"
    else:
        field="dropout_factor";order=["0.0","0.2","0.4","0.6","0.8"]
        groups=[(0,"Without BN"),(1,"With BN")]
        group_field="bn";xlabel="Dropout probability"
    for offset,(key,label),color in zip((-.05,.05),groups,("#316a96","#b56840")):
        rows=cells.loc[cells[group_field].eq(key)].set_index(field).loc[order]
        center=rows.fitted_r.to_numpy()
        ax.errorbar(np.arange(len(order))+offset,center,
            yerr=np.stack([center-rows.ci95_low_r.to_numpy(),rows.ci95_high_r.to_numpy()-center]),
            fmt="o-",color=color,label=label,capsize=3)
    ax.set_xticks(np.arange(len(order)),order)
    ax.set_xlabel(xlabel);ax.set_ylabel("NGCI: tanh(fitted Fisher z), 95% CI")
    ax.legend(frameon=False);ax.spines[["top","right"]].set_visible(False)
    save_figure(fig,folder/"interaction")


def summarize_models(table):
    rows=[]
    for (cid,number),part in table.groupby(["condition_id","model_number"],sort=False):
        r=part.ngci_raw.dropna()
        q=r.quantile([.25,.5,.75])
        rows.append(dict(condition_id=cid,model_number=int(number),n_neurons=len(part),n_valid=len(r),
            n_missing=int(part.ngci_raw.isna().sum()),ngci_median=q.loc[.5],q1=q.loc[.25],q3=q.loc[.75],
            iqr=q.loc[.75]-q.loc[.25],fraction_r_gt_0_5=float((r>.5).mean()) if len(r) else np.nan,
            n_lmm_zero_imputed=int(part.missing_ngci_replaced_with_zero.sum())))
    return pd.DataFrame(rows)


def save_figure(fig,path):
    if "self_test" in Path(path).parts:
        fig.suptitle("Synthetic validation only",fontsize=10)
    for suffix in (".png",".pdf"):
        fig.savefig(out(Path(str(path)+suffix)),dpi=200,bbox_inches="tight")
    plt.close(fig)


def plot_experiment(table,summary,effects,folder,order):
    """神经元分布仅作描述；模型级点明确表示独立训练重复，不绘制t-SNE。"""
    labels=[]
    conditions=condition_catalog()[0]
    for cid in order:
        c=conditions[cid]
        if cid=="baseline":label="Baseline"
        elif cid=="small":label="19,666 features"
        elif cid=="deep":label="Deep"
        elif cid=="no_bn":label="Without BN"
        elif cid.startswith("dropout_"):label=f"Dropout {c['dropout']:.1f}\n{'BN' if c['bn_position']!='none' else 'No BN'}"
        else:label=c["activation"]+"\n"+("BN → activation" if c["bn_position"]=="linear_bn_activation" else "Activation → BN")
        labels.append(label)
    fig,ax=plt.subplots(figsize=(max(6,len(order)*1.1),4.5))
    vals=[table.loc[table.condition_id.eq(c),"ngci_raw"].dropna().to_numpy() for c in order]
    ax.boxplot(vals,tick_labels=labels,showfliers=True,widths=.5)
    ax.axhline(0,color=".65",lw=.6);ax.set_ylabel("Weight NGCI (Pearson r)")
    ax.tick_params(axis="x",labelrotation=30)
    ax.spines[["top","right"]].set_visible(False)
    save_figure(fig,folder/"neuron_ngci")
    for field,ylabel in (("ngci_median","Median NGCI per model"),("fraction_r_gt_0_5","Fraction with NGCI > 0.5 (valid neurons)")):
        fig,ax=plt.subplots(figsize=(max(6,len(order)*1.1),4.5))
        values=[summary.loc[summary.condition_id.eq(c),field].dropna().to_numpy() for c in order]
        ax.boxplot(values,tick_labels=labels,showfliers=False,widths=.5)
        for i,array in enumerate(values,1):
            ax.scatter(i+np.linspace(-.12,.12,len(array)),array,s=13,color="#316a96",alpha=.65)
        ax.set_ylabel(ylabel);ax.tick_params(axis="x",labelrotation=30)
        ax.spines[["top","right"]].set_visible(False)
        save_figure(fig,folder/field)
    main=effects.loc[effects.term.ne("Intercept")].copy()
    fig,ax=plt.subplots(figsize=(9,max(3,len(main)*.45)))
    x=main.estimate.to_numpy();low=main.ci95_low.to_numpy();high=main.ci95_high.to_numpy()
    ax.errorbar(x,np.arange(len(main)),xerr=np.stack([x-low,high-x]),fmt="o",color="#316a96",capsize=3)
    ax.set_yticks(np.arange(len(main)),[f"{r.comparison}: {r.term}" for r in main.itertuples()],fontsize=8)
    ax.invert_yaxis();ax.axvline(0,color=".5",lw=.8)
    ax.set_xlabel("Fixed effect (Fisher z), 95% CI");ax.spines[["top","right"]].set_visible(False)
    save_figure(fig,folder/"fixed_effects")


def lmm_specifications(table,experiment):
    """显式指定参考水平和对比方向，训练条件名称不参与隐式字母排序。"""
    table=table.copy()
    if experiment=="bn":
        specifications=[("bn",table,"z_coef ~ bn",True)]
    elif experiment=="activation_bn":
        table["activation"]=pd.Categorical(table.activation,categories=["LeakyReLU","ReLU","Sigmoid","Swish","Tanh"])
        table["bn_position"]=pd.Categorical(table.bn_position,categories=["linear_bn_activation","linear_activation_bn"])
        specifications=[("activation_bn",table,"z_coef ~ activation * bn_position",False)]
    elif experiment=="dropout_bn":
        table["dropout_factor"]=pd.Categorical(table.dropout_factor,categories=["0.0","0.2","0.4","0.6","0.8"])
        specifications=[("dropout_bn",table,"z_coef ~ C(dropout_factor) * bn",False)]
    else:
        require(experiment=="structure","未知统计子实验。")
        specifications=[]
        for other,label in (("deep","deep_minus_shallow"),("small","full_minus_small")):
            pair=table.loc[table.condition_id.isin([BASELINE,other])].copy()
            pair["contrast"]=pair.condition_id.eq(other if other=="deep" else BASELINE).astype(int)
            specifications.append((label,pair,"z_coef ~ contrast",True))
    return specifications


def finalize(args):
    design=verify_setup()
    cs=selected_conditions(design,args.experiment)
    neurons=pd.concat([finalize_condition(c) for c in cs],ignore_index=True)
    table=statistical_input(neurons,args.experiment,design["conditions"])
    folder=ROOT/"statistics"/args.experiment
    write_csv(table,folder/"lmm_input.csv")
    effects=[];diagnostics=[];texts=[]
    specifications=lmm_specifications(table,args.experiment)
    try:
        for label,data,formula,reml in specifications:
            coef,diag,fitted=fit_lmm(data,formula,reml,label)
            effects.append(coef);diagnostics.append(diag);texts.append(fitted.summary().as_text())
            if args.experiment in ("activation_bn","dropout_bn"):
                cells=interaction_means(fitted,data)
                write_csv(cells,folder/"interaction_means.csv")
                plot_interaction(cells,args.experiment,ROOT/"figures"/args.experiment)
    except Exception as error:
        write_json(ROOT/"quality_control"/f"{args.experiment}_lmm.json",dict(status="failed",error=str(error),
            failed_fit=getattr(error,"diagnostic",None),completed_fits=diagnostics))
        raise
    effects=pd.concat(effects,ignore_index=True)
    main=effects.term.ne("Intercept")
    effects["q_fixed_effects"]=np.nan
    effects.loc[main,"q_fixed_effects"]=bh(effects.loc[main,"p_value"])
    effects["correction_family"]=args.experiment+"_nonintercept_fixed_effects"
    effects["significant_q_lt_0_05"]=effects.q_fixed_effects.lt(.05)
    write_csv(effects,folder/"fixed_effects.csv")
    summary=summarize_models(table)
    write_csv(summary,folder/"model_summaries.csv")
    out(folder/"lmm_summary.txt").write_text("\n\n".join(texts),encoding="utf-8")
    write_json(folder/"lmm_diagnostics.json",dict(fits=diagnostics))
    plot_experiment(table,summary,effects,ROOT/"figures"/args.experiment,[c["condition_id"] for c in cs])
    outputs=[*folder.glob("*"),*(ROOT/"figures"/args.experiment).glob("*")]
    write_json(ROOT/"quality_control"/f"{args.experiment}_lmm.json",dict(status="complete",experiment=args.experiment,
        n_models=int(table.model_id.nunique()),n_neurons=len(table),n_imputed=int(table.missing_ngci_replaced_with_zero.sum()),
        fits=diagnostics,source_sha256=hashes([CODE,ROOT/"metadata/design.json",*[ROOT/"quality_control/finalize_conditions"/f"{c['condition_id']}.json" for c in cs]]),
        output_sha256=hashes(outputs)))
    print(f"{args.experiment} 汇总完成；效应单位为 Fisher z，原始NGCI分布保留。",flush=True)


def self_test(device_name):
    """真实恢复轨迹检查：含Dropout和BN；中断训练/测试/epoch边界后逐张量比对。

    仅在 quality_control/self_test 生成小型测试 checkpoint，不启动正式训练。
    另核验 Swish、缺失值政策、BH、完整配置形状和真实基线的评分方向。
    """
    device=runtime(device_name)
    folder=ROOT/"quality_control/self_test"
    conditions,experiments=condition_catalog()
    require(len(conditions)==23 and len(set(training_seed(c,m) for c in conditions.values()
        if c["condition_id"]!=BASELINE for m in range(1,21)))==440,"条件去重或种子冲突。")
    for name,c in conditions.items():
        with torch.device("meta"):
            model=Autoencoder(c,19666 if name=="small" else 60498)
        expected=0 if c["bn_position"]=="none" else 2*len(c["hidden_dims"])
        require(sum(isinstance(m,nn.BatchNorm1d) for m in model.modules())==expected,"BN数量错误。")
        require(model.decoder[0].in_features==64,"瓶颈维度错误。")
        dims=[19666 if name=="small" else 60498,*c["hidden_dims"],64]
        for block,widths in ((model.encoder,dims),(model.decoder,dims[::-1])):
            require([(m.in_features,m.out_features) for m in block if isinstance(m,nn.Linear)]==list(zip(widths,widths[1:])),
                    "Linear输入/输出方向错误。")
            if c["bn_position"]=="linear_bn_activation":
                require(isinstance(block[1],nn.BatchNorm1d) and isinstance(block[2],type(activation_module(c["activation"]))),"BN位置错误。")
            elif c["bn_position"]=="linear_activation_bn":
                require(isinstance(block[2],nn.BatchNorm1d) and isinstance(block[1],type(activation_module(c["activation"]))),"BN位置错误。")
    swish=activation_module("Swish")
    x=torch.linspace(-10,10,101,device=device,requires_grad=True)
    expected=x*torch.sigmoid(x)
    require(torch.allclose(swish(x),expected,atol=2e-6,rtol=2e-6),"Swish数值定义不符。")
    grad=torch.autograd.grad(swish(x).sum(),x)[0]
    reference_grad=torch.sigmoid(x)+x*torch.sigmoid(x)*(1-torch.sigmoid(x))
    require(torch.allclose(grad,reference_grad,atol=2e-6,rtol=2e-6),"Swish导数不符。")
    tiny=copy.deepcopy(conditions[BASELINE]);tiny.update(hidden_dims=[7,5],latent_dim=3)
    data=np.random.default_rng(629).normal(size=(28,11)).astype(np.float16)
    tr,te=np.arange(20),np.arange(20,28)
    def new_model():
        seed_all(931)
        m=Autoencoder(tiny,11).to(device)
        o=torch.optim.Adam(m.parameters(),lr=.0002,foreach=False)
        g=torch.Generator().manual_seed(940)
        return m,o,g
    uninterrupted,optim_ref,gen=new_model()
    progress_ref=initial_progress()
    train_loop(uninterrupted,optim_ref,data,tr,te,progress_ref,gen,device,lambda p,g:None,
               lambda:False,epochs=3,batch_size=4)
    target_state={k:v.detach().cpu().clone() for k,v in uninterrupted.state_dict().items()}
    target_optim=copy.deepcopy(optim_ref.state_dict())
    target_rng=rng_state();target_shuffle=gen.get_state().clone()
    def same(a,b):
        if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    results=[]
    for label,stop_after in (("train_batch",2),("test_batch",6),("last_test_batch",7),("epoch_boundary",None)):
        model,optimizer,generator=new_model()
        progress=initial_progress();counter=[0]
        path=folder/f"{label}.pt"
        def saved(p,g):
            save_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),progress=copy.deepcopy(p),
                            shuffle=g.get_state(),rng=rng_state()),path)
        def count(p):counter[0]+=1
        completed=train_loop(model,optimizer,data,tr,te,progress,generator,device,saved,
            lambda:progress["epoch"]==2 if stop_after is None else counter[0]>=stop_after,
            epochs=3,batch_size=4,after_batch=count)
        require(not completed and path.is_file(),"暂停未保存 checkpoint。")
        restored=torch.load(path,map_location="cpu",weights_only=True)
        resumed,resume_optimizer,resume_generator=new_model()
        resumed.load_state_dict(restored["model"])
        resume_optimizer.load_state_dict(restored["optimizer"])
        resume_generator.set_state(restored["shuffle"])
        restore_rng(restored["rng"])
        p=restored["progress"]
        train_loop(resumed,resume_optimizer,data,tr,te,p,resume_generator,device,lambda p,g:None,
                   lambda:False,epochs=3,batch_size=4)
        require(same(target_state,resumed.state_dict()),f"{label}: 恢复参数/BN状态不等同连续训练。")
        require(same(target_optim,resume_optimizer.state_dict()),f"{label}: Adam状态不一致。")
        require(p["history"]==progress_ref["history"],f"{label}: 每epoch损失不一致。")
        require(same(rng_state(),target_rng) and torch.equal(resume_generator.get_state(),target_shuffle),f"{label}: RNG状态不一致。")
        results.append(dict(stop=label,exact_model_match=True,exact_optimizer_match=True,exact_loss_match=True,exact_rng_match=True))
        path.unlink()
        del model,optimizer,resumed,resume_optimizer,restored
    del uninterrupted,optim_ref,target_optim
    if device.type=="cuda":torch.cuda.empty_cache()
    p=np.array([.01,np.nan,.04,.2])
    require(np.allclose(bh(p)[[0,2,3]],[.04,.08,.2666666666666667]),"计划分母 BH 错误。")
    toy=pd.DataFrame(dict(condition_id=[BASELINE]*3,model_number=[1]*3,neuron_index=[0,1,2],ngci=[np.nan,.5,1.]))
    converted=statistical_input(toy,"activation_bn",conditions)
    require(converted.ngci_raw.isna().sum()==1 and converted.missing_ngci_replaced_with_zero.sum()==1 and
            converted.fisher_boundary_clipped.sum()==1 and converted.ngci_for_lmm.iloc[0]==0.,"NA转换政策错误。")
    ordinary=statistical_input(toy.iloc[:2],"bn",conditions)
    require(not ordinary.included_in_lmm.iloc[0] and ordinary.ngci_raw.isna().iloc[0],"BN实验错误地替换缺失值。")
    require(np.isnan(ngci_result(np.arange(53),np.ones(53))["ngci"]),"常量向量未保留NA。")
    # 从真实基线输入检查权重列方向与层次；不调用Jacobian，也不重新计算GSEA。
    model,cp=load_analysis_model(conditions[BASELINE],1,60498,device)
    checkpoint_hash=sha(cp)
    entry,raw,audit=propagate_weight(model.decoder)
    saved_raw=np.load("step7/rank_inputs/model_01/gene_scores.npy",mmap_mode="r")
    require(np.allclose(raw,saved_raw,rtol=3e-5,atol=.03),"基线权重评分与Step7不同。")
    maximum=float(np.max(np.abs(raw-saved_raw)))
    require(sha(cp)==checkpoint_hash,"分析意外改变训练checkpoint。")
    require(np.any(entry<0) and np.any(entry>0),"入口未保留正负号。")
    del model,raw,saved_raw
    if device.type=="cuda":torch.cuda.empty_cache()
    # 混合模型使用合成数据只检查接口、随机截距与返回表的单位；不作为实验结果。
    rng=np.random.default_rng(108629)
    rows=[]
    for bn in (0,1):
        for m in range(20):
            intercept=rng.normal(0,.13)
            for unit in range(64):
                z=.4+.2*bn+intercept+rng.normal(0,.2)
                rows.append(dict(model_id=f"{bn}_{m}",bn=bn,z_coef=z,included_in_lmm=True))
    fitted,diag,_=fit_lmm(pd.DataFrame(rows),"z_coef ~ bn",True,"synthetic_bn")
    require(diag["n_models"]==40 and diag["n_observations"]==2560 and
            set(fitted.term)=={"Intercept","bn"},"LMM模型分组错误。")
    # 含交互项的两套统计设计和结构对比也用合成数据检查；不写入正式statistics。
    synthetic=[]
    for cid,c in conditions.items():
        effect=.12*(c["bn_position"]!="none")+.08*(c["bn_position"]=="linear_activation_bn")
        effect+=.05*["LeakyReLU","ReLU","Tanh","Sigmoid","Swish"].index(c["activation"])-.2*c["dropout"]
        effect+=-.18*(cid=="deep")-.11*(cid=="small")
        for m in range(1,21):
            intercept=rng.normal(0,.10)
            for unit in range(64):
                synthetic.append(dict(condition_id=cid,model_number=m,neuron_index=unit,
                    ngci=np.tanh(.4+effect+intercept+rng.normal(0,.16))))
    synthetic=pd.DataFrame(synthetic)
    tested_fits=[]
    for experiment in EXPERIMENTS:
        data=statistical_input(synthetic.loc[synthetic.condition_id.isin(experiments[experiment])],experiment,conditions)
        for label,part,formula,reml in lmm_specifications(data,experiment):
            coefs,diagnostic,result=fit_lmm(part,formula,reml,label)
            expected_models=200 if experiment in ("activation_bn","dropout_bn") else 40
            require(diagnostic["n_models"]==expected_models and diagnostic["n_observations"]==expected_models*64,
                    "合成LMM遗漏了模型或神经元。")
            if experiment in ("activation_bn","dropout_bn"):
                cells=interaction_means(result,part)
                require(len(cells)==10 and len(coefs)==10 and np.isfinite(cells.fitted_r).all(),"交互条件数量或固定效应设计错误。")
                require(np.allclose(cells.estimate_z,result.predict(cells),atol=1e-12),"条件均值与LMM固定效应预测不一致。")
                plot_interaction(cells,experiment,folder/"synthetic_figures"/experiment)
            tested_fits.append(dict(comparison=label,n_models=expected_models,reml=reml,converged=True))
    write_json(folder/"report.json",dict(status="passed",device=str(device),versions=versions(),
        resume_tests=results,baseline_max_abs_score_difference=maximum,propagation_audit=audit,
        swish_function_and_gradient=True,missing_policy=True,bh_with_missing=True,
        lmm_random_intercept=True,lmm_design_checks=tested_fits,code_sha256=sha(CODE)))
    print("self-test 通过：断点恢复、Swish、权重传播、NA规则、BH及LMM。",flush=True)


def status():
    """状态命令只读，不生成停止标志或启动工作。"""
    active=ROOT/"control/active_train.json"
    if active.exists():print(json.dumps(read_json(active),ensure_ascii=False,indent=2))
    for path in sorted((ROOT/"models").glob("*")) if (ROOT/"models").exists() else []:
        complete=list(path.glob("model_*/complete.json"))
        resumes=list(path.glob("model_*/resume.pt"))
        print(f"{path.name}: completed={len(complete)}/20, resumable={len(resumes)}")
        for resume in resumes:
            progress=resume.parent/"progress.json"
            if progress.exists():
                row=read_json(progress)
                print(resume.parent.name,{k:v for k,v in row.items() if k!="configuration"})


def main():
    # 子进程日志统一UTF-8，避免Windows终端/Notebook对中文使用不同编码。
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="backslashreplace")
    parser=argparse.ArgumentParser(description="Step8 component experiments and weight-based NGCI")
    parser.add_argument("stage",choices=["setup","self-test","train","prepare","finalize","stop","status"])
    parser.add_argument("--experiment",choices=EXPERIMENTS,default="bn")
    parser.add_argument("--condition",default=None)
    parser.add_argument("--model-start",type=int,default=1)
    parser.add_argument("--model-end",type=int,default=20)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--save-seconds",type=float,default=300.)
    args=parser.parse_args()
    enter_trial()
    require(1<=args.model_start<=args.model_end<=20,"模型范围必须在1..20。")
    require(args.save_seconds>=1,"checkpoint间隔至少1秒。")
    if args.stage=="setup":setup()
    elif args.stage=="self-test":self_test(args.device)
    elif args.stage=="train":train(args)
    elif args.stage=="prepare":prepare(args)
    elif args.stage=="finalize":
        require(args.condition is None,"finalize 必须包括该子实验的所有条件。")
        finalize(args)
    elif args.stage=="stop":
        out(STOP).write_text("Stop at next batch boundary.\n",encoding="utf-8")
        print("停止请求已写入；等待训练进程保存完成。",flush=True)
    else:status()


if __name__=="__main__":
    main()

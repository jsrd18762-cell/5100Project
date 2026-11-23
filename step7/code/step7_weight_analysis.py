"""Step7：瓶颈权重传播、完整排序 GSEA 的准备及结果整理。

从 trial 或其子目录运行：check -> self-test -> prepare -> compare。
之后在 R 中手动运行 01_run_fgsea_parallel.R，再运行本文件的 finalize，
随后运行 display_muscle.py，再手动运行 03_plot_muscle_enrichment.R。Python 不启动 R。
仅写 step7；所有模型、样本中位数、参考库及 Jacobian 输入均只读。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

# 在导入其他步骤的纯函数前禁用字节码，避免在上游目录创建 __pycache__。
sys.dont_write_bytecode = True
TRIAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRIAL_ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
import torch
from torch import nn
from step5.code.step5_gsea import REFERENCE_SELECTION, bh_planned, merge_scores, stability_tables
from step6.code.step6_ngci import pearson_result

ROOT = Path("step7")
CODE = ROOT / "code/step7_weight_analysis.py"
LOGGER = logging.getLogger("step7")
N_MODELS, N_UNITS, N_FEATURES, N_GENES, N_TISSUES = 20, 64, 60498, 58581, 53
REFERENCE_FILES = ("gene_map.csv", "feature_to_gene.csv", "reference_sets.csv", "pathways.json",
                   "tissue_map.csv", "display_selection.csv", "reference_selection.json", "top100_candidates.csv")
PARAMETERS = {"minSize": 15, "maxSize": 500, "eps": 0, "scoreType": "std", "gseaParam": 1,
              "sampleSize": 101, "nPermSimple": 1000}
DESIGN = {
    "score_source": "weight", "n_models": 20, "n_neurons_per_model": 64,
    "reference_selection": REFERENCE_SELECTION,
    "weight_module": "decoder.0", "weight_shape": [512, 64], "neuron_axis": "weight column",
    "entry_formula": "10000 * decoder.0.weight.T", "propagation_entry": "decoder[1:]",
    "mode": "eval", "grad_enabled": False, "linear_bias": "zero in analysis copy",
    "bn_beta": "zero in analysis copy", "bn_gamma": "trained value",
    "bn_running_mean": "trained value; unchanged during propagation",
    "bn_running_var": "trained value; unchanged during propagation", "bn_eps": "unchanged (1e-5)",
    "bn_statistics": "trained running statistics; decoder.1 and decoder.5 each called once",
    "dropout": "disabled by eval; p=0.3 modules retained",
    "propagation_batch_size": 64, "propagation_row_order": "unit_00000 through unit_00063",
    "propagation_passes": 1, "propagation_seed_formula": "629 + 7100000 + (model_number - 1)",
    "dtype": "float32 propagation; float64 exact-name mean and rank analysis", "TF32": False,
    "ranking_scope": "one tissue-independent vector per model and neuron",
    "gene_matching": "exact names; arithmetic mean; same 58581-gene universe as Step5",
    "rank_ties": "descending signed score, stable UTF-8 gene order; no jitter",
    "gsea_method": "fgseaMultilevel", "parameters": PARAMETERS,
    "primary_family": "weight only; all 20 models x 64 neurons x 53 reference tissues",
    "primary_family_size": 67840, "BH": "single weight-only family; separate from Jacobian",
    "missing_p": "retain NA; include in planned BH denominator", "significance": "q_matched_global < 0.05",
    "stability": "tissue detection and fractions across models; no neuron matching",
    "seed_formula": "629 + 7000000 + (model_number - 1) * 64 + neuron_index",
    "ngci": "signed Pearson r; ES; 53 equally weighted tissues; two-sided H0:rho=0",
    "ngci_p": "actual p and p<0.05 flag; no correction or selection",
    "spearman": "weight score vs tissue mean Jacobian score; all 58581 genes after same-name merging",
    "spearman_ties": "average ranks", "spearman_scope": "all 54 tissues with mean Jacobian scores, including CML",
    "spearman_inference": "coefficient and gene count only; no p threshold or selection",
    "display_model": 1, "display_unit": 0, "display_tissues": "fixed Step5 high/low three each",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(path: Path) -> None:
    """所有写入入口限制在 step7；拒绝绝对路径或向上跳转。"""
    require(not path.is_absolute() and path.parts[0] == "step7" and ".." not in path.parts,
            f"输出必须位于 step7：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: dict) -> None:
    output_path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    output_path(path)
    frame.to_csv(path, index=False, float_format="%.17g", na_rep="NA")


def hashes(paths) -> dict:
    return {Path(p).as_posix(): sha(Path(p)) for p in paths}


def check_hashes(values: dict) -> None:
    for file, expected in values.items():
        require(Path(file).is_file() and sha(Path(file)) == expected, f"文件缺失或指纹改变：{file}")


def record(path: Path, sources: bool = False) -> dict:
    saved = read_json(path)
    require(saved.get("status") == "complete", f"阶段尚未完成：{path}")
    check_hashes(saved["output_sha256"])
    if sources:
        check_hashes(saved["source_sha256"])
    return saved


def require_current_design(saved: dict) -> None:
    """旧传播模式或旧参考库的完成标记不能作为当前分析的输入。"""
    require(saved.get("configuration", {}).get("design") == DESIGN,
            "Step7参考库或传播设计已改变：请按Notebook从prepare重建输入及后续结果。")


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "torch": torch.__version__, "matplotlib": matplotlib.__version__}


def config(sources: dict, **extra) -> dict:
    # 只导入上游纯计算函数，不调用其入口、绘图或文件写入函数。
    return {"design": DESIGN, "source_sha256": sources, "versions": versions(),
            "code_sha256": hashes([CODE, Path("step5/code/step5_gsea.py"), Path("step6/code/step6_ngci.py")]),
            **extra}


def begin(path: Path, cfg: dict, outputs: list[Path], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        saved = record(path)
        require(saved["configuration"] == cfg and set(saved["output_sha256"]) == {p.as_posix() for p in outputs},
                f"配置改变，请检查后使用 --overwrite：{path}")
        LOGGER.info("Skip complete: %s", path)
        return False
    require(overwrite or not any(p.exists() for p in outputs), "存在未完成输出，请核对后使用 --overwrite。")
    for p in outputs:
        output_path(p)
    write_json(path, {"status": "running", "configuration": cfg})
    return True


def finish(path: Path, cfg: dict, outputs: list[Path], **details) -> None:
    write_json(path, {"status": "complete", "configuration": cfg, "source_sha256": cfg["source_sha256"],
                      "output_sha256": hashes(outputs), **details})


def seed_for(number: int, unit: int) -> int:
    return 629 + 7000000 + (number - 1) * 64 + unit


def propagation_seed_for(number: int) -> int:
    """保留每模型传播种子的记录；eval 下无随机掩码，输出不依赖此种子。"""
    return 629 + 7100000 + (number - 1)


def upstream() -> dict:
    """核对 Step5 参考库和 Step6 完成状态，不重算任一上游结果。"""
    reference = record(Path("step5/quality_control/reference.json"))
    require(reference["configuration"]["design"].get("reference_selection") == REFERENCE_SELECTION,
            "Step5尚未构建当前平均TPM Top100参考库。")
    six = record(Path("step6/quality_control/run.json"))
    check_hashes(six["configuration"]["source_sha256"])
    require(six["n_neurons"] == 1280 and six["configuration"]["design"]["gsea_score"] == "ES",
            "Step6 对象或 GSEA 得分不符合当前设计。")
    require(six["configuration"]["design"].get("reference_id") == REFERENCE_SELECTION["id"],
            "请先使用Top100的Step5结果重新运行Step6。")
    ref_paths = [Path("step5/reference") / name for name in REFERENCE_FILES]
    lock = Path("step5/environment/r_environment_lock.json")
    lock_data = read_json(lock)
    require(lock_data["R"] == "4.6.1" and lock_data["packages"]["fgsea"] == "1.38.0", "R/fgsea 版本锁不匹配。")
    return hashes([*ref_paths, lock, Path("step5/quality_control/reference.json"),
                   Path("step6/quality_control/run.json"), Path("step6/statistics/ngci_per_neuron.csv"),
                   Path("step6/statistics/ngci_tissue_pairs.csv"), Path("step1/core_metadata/model_architecture.json")])


def prepare_reference(sources: dict, overwrite: bool) -> None:
    """保存分析所用参考快照；原始来源保持只读，基因和组织顺序不重建。"""
    outputs = [ROOT / "reference" / name for name in REFERENCE_FILES]
    outputs += [ROOT / "environment/r_environment_lock.json", ROOT / "reference/analysis_design.json"]
    cfg = config(sources)
    path = ROOT / "quality_control/reference.json"
    if not begin(path, cfg, outputs, overwrite):
        return
    for name, dest in zip(REFERENCE_FILES, outputs):
        dest.write_bytes((Path("step5/reference") / name).read_bytes())
    outputs[-2].write_bytes(Path("step5/environment/r_environment_lock.json").read_bytes())
    write_json(outputs[-1], DESIGN)
    finish(path, cfg, outputs)


def reference_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = ROOT / "quality_control/reference.json"
    require_current_design(read_json(path))
    record(path, sources=True)
    require(read_json(ROOT / "reference/analysis_design.json") == DESIGN, "参考快照中的传播设计不匹配。")
    require(read_json(ROOT / "reference/reference_selection.json")["specification"] == REFERENCE_SELECTION,
            "Step7参考快照不是当前平均TPM Top100集合。")
    genes = pd.read_csv(ROOT / "reference/gene_map.csv", keep_default_na=False)
    features = pd.read_csv(ROOT / "reference/feature_to_gene.csv", keep_default_na=False)
    refs = pd.read_csv(ROOT / "reference/reference_sets.csv", keep_default_na=False)
    require(len(genes) == N_GENES and genes.gene_name.is_unique
            and np.array_equal(genes.merged_column, np.arange(N_GENES)), "合并基因顺序错误。")
    require(len(features) == N_FEATURES and np.array_equal(features.matrix_column, np.arange(N_FEATURES))
            and genes.gene_name.iloc[features.merged_column].tolist() == features.gene_name.tolist(), "特征到基因名的映射错误。")
    counts = np.bincount(features.merged_column, minlength=N_GENES)
    require(np.array_equal(counts, genes.n_features) and len(refs) == 53, "基因合并计数或参考集合数错误。")
    pathways = read_json(ROOT / "reference/pathways.json")
    require(refs.n_matched.eq(np.minimum(refs.n_candidates_matched, 100)).all()
            and refs.reference_tissue_id.map(lambda tid: len(pathways[tid])).eq(refs.n_matched).all(),
            "Top100参考成员数不一致。")
    return genes, features, refs


def load_decoder(number: int, device: torch.device) -> tuple[nn.Sequential, dict, dict]:
    """复制训练参数后加载 Decoder，绝不对磁盘映射的 checkpoint 张量原地归零。"""
    path = Path(f"step1/core_models/model_{number:02d}.pt")
    step4 = Path(f"step4/quality_control/model_{number:02d}/model.json")
    fourth = read_json(step4)
    digest = sha(path)
    require(fourth["status"] == "complete" and digest == fourth["configuration"]["checkpoint_sha256"],
            f"model_{number:02d}: checkpoint 与 Jacobian 来源不一致。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(checkpoint["model_number"] == number and checkpoint["model_seed"] == 629 + number
            and checkpoint["final_epoch"] == 100
            and checkpoint["architecture_config"] == read_json(Path("step1/core_metadata/model_architecture.json")),
            "模型编号、种子、epoch 或结构错误。")
    # meta 仅建立形状；clone 使分析副本和检查点不共享存储。
    with torch.device("meta"):
        decoder = nn.Sequential(nn.Linear(64, 512), nn.BatchNorm1d(512), nn.LeakyReLU(.2), nn.Dropout(.3),
                                nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(.2), nn.Dropout(.3),
                                nn.Linear(2048, 60498))
    state = {k.removeprefix("decoder."): v.clone() for k, v in checkpoint["model_state_dict"].items()
             if k.startswith("decoder.")}
    expected = decoder.state_dict()
    require(set(state) == set(expected) and all(state[k].shape == v.shape and state[k].dtype == v.dtype
            for k, v in expected.items()), "Decoder 参数名称、形状或精度错误。")
    require(all(torch.isfinite(v).all().item() for v in state.values()), "模型包含非有限参数。")
    decoder.load_state_dict(state, strict=True, assign=True)
    # eval() 让 BN 使用训练期间的 running 统计量，并关闭两个 Dropout。
    # 这是独立分析副本，不执行反向传播或优化器更新。
    decoder.to(device).eval().requires_grad_(False)
    modified = []
    with torch.no_grad():
        for name, layer in decoder.named_modules():
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                modified.append({"parameter": f"decoder.{name}.bias", "n_nonzero_before": int(layer.bias.count_nonzero())})
                layer.bias.zero_()
            if isinstance(layer, nn.BatchNorm1d):
                require(layer.eps == 1e-5 and layer.track_running_stats, "BN 设置不匹配。")
    audit = {"zeroed_biases": modified, "retained_at_load": ["weights", "BN gamma", "BN running_mean", "BN running_var", "BN num_batches_tracked", "BN eps"],
             "training_mode": decoder.training, "checkpoint_sha256": digest,
             "model_seed": 629 + number, "epoch": 100}
    return decoder, {path.as_posix(): digest, step4.as_posix(): sha(step4)}, audit


def propagate(decoder: nn.Sequential, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """将全部 64 个权重列按固定顺序作为一个批次，仅前向传播一次。

    入口是第一层 Linear 的权重列，维度已为512，不重复执行该Linear。
    从索引1开始依次经过两组BN/LeakyReLU及后续Linear；两组BN都使用训练
    期间的running统计量，beta按既定设置为零。Dropout关闭，入口保留正负号。
    """
    require(all(not module.training for module in decoder.modules()) and decoder[0].weight.shape == (512, 64),
            "传播入口或评估模式错误。")
    before = {name: {key: value.detach().clone() for key, value in layer.named_buffers()}
              for name, layer in decoder.named_modules() if isinstance(layer, nn.BatchNorm1d)}
    # 保留既定派生种子及记录；当前传播不使用随机算子，自检验证种子不影响输出。
    torch.manual_seed(seed)
    # 逐层记录实际调用，防止配置声明包含第一组BN/激活而执行路径仍跳过它们。
    call_counts = {f"decoder.{index}": 0 for index in range(len(decoder))}
    def count_call(index):
        def hook(module, arguments, output):
            call_counts[f"decoder.{index}"] += 1
        return hook
    hooks = [layer.register_forward_hook(count_call(index)) for index, layer in enumerate(decoder)]
    try:
        with torch.no_grad():
            entry = decoder[0].weight.T.contiguous() * 10000
            scores = decoder[1:](entry)
    finally:
        for hook in hooks:
            hook.remove()
    require(call_counts == {f"decoder.{index}": int(index > 0) for index in range(9)},
            "传播必须仅执行decoder[1:]，各模块一次，不重复第一层Linear。")
    require(scores.shape == (64, 60498) and torch.isfinite(scores).all().item(), "权重基因评分无效。")
    updates = {}
    for name, initial in before.items():
        layer = decoder[int(name)]
        first, last = int(initial["num_batches_tracked"].item()), int(layer.num_batches_tracked.item())
        require(all(torch.equal(value, getattr(layer, key)) for key, value in initial.items()),
                "评估模式传播不应修改任何 BN 缓冲区。")
        updates[f"decoder.{name}"] = {"batches_before": first, "batches_after": last,
            "running_mean_max_abs_change": float((layer.running_mean - initial["running_mean"]).abs().max().item()),
            "running_var_max_abs_change": float((layer.running_var - initial["running_var"]).abs().max().item())}
    audit = {"seed": seed, "batch_size": 64, "row_order": DESIGN["propagation_row_order"],
             "forward_passes": 1, "grad_enabled": False, "module_call_counts": call_counts,
             "bn_buffer_updates": updates}
    return entry.cpu().numpy(), scores.cpu().numpy(), audit


def prepare_model(number: int, device: torch.device, genes, features, overwrite: bool) -> None:
    decoder, sources, audit = load_decoder(number, device)
    sources.update(hashes([ROOT / "quality_control/reference.json", Path("step1/core_metadata/model_architecture.json")]))
    seed = propagation_seed_for(number)
    cfg = config(sources, model_number=number, device=str(device), propagation_seed=seed)
    folder = ROOT / f"rank_inputs/model_{number:02d}"
    outputs = [folder / name for name in ("entry.npy", "gene_scores.npy", "scores.f64.bin", "jobs.csv", "score_qc.csv")]
    report = ROOT / f"quality_control/prepared/model_{number:02d}.json"
    if not begin(report, cfg, outputs, overwrite):
        return
    started = time.perf_counter()
    entry, raw, audit["propagation"] = propagate(decoder, seed)
    merged = merge_scores(raw, features, genes.n_features.to_numpy())
    np.save(outputs[0], entry, allow_pickle=False)
    np.save(outputs[1], raw, allow_pickle=False)
    merged.astype("<f8").tofile(outputs[2])
    jobs = pd.DataFrame({"model_number": number, "matrix_row": np.arange(64), "neuron_index": np.arange(64),
                         "unit_id": [f"unit_{u:05d}" for u in range(64)], "seed": [seed_for(number, u) for u in range(64)]})
    write_csv(jobs, outputs[3])
    qc = jobs[["neuron_index", "unit_id"]].copy()
    qc["score_min"], qc["score_max"] = merged.min(1), merged.max(1)
    qc["n_positive"], qc["n_negative"], qc["n_zero"] = (merged > 0).sum(1), (merged < 0).sum(1), (merged == 0).sum(1)
    qc["n_unique_scores"] = [len(np.unique(row)) for row in merged]
    write_csv(qc, outputs[4])
    finish(report, cfg, outputs, model_number=number, n_neurons=64, n_features=60498, n_genes=58581,
           binary_dtype="little-endian float64", binary_order="row-major", audit=audit,
           elapsed_seconds=time.perf_counter() - started)
    LOGGER.info("Prepared model_%02d: 64 x 58581", number)


def centered_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman 使用平均并列秩；返回中心化的单位范数秩，常量行保留NaN。"""
    require(values.ndim == 2 and np.isfinite(values).all(), "排序评分必须为有限二维矩阵。")
    ranks = stats.rankdata(values, axis=1, method="average")
    ranks -= ranks.mean(1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1)
    normalized = np.full_like(ranks, np.nan)
    np.divide(ranks, norms[:, None], out=normalized, where=norms[:, None] > 0)
    return normalized, norms > 0


def compare_model(number: int, tissues: pd.DataFrame, overwrite: bool) -> None:
    """复用 Step5 已合并的平均梯度，不使用中位梯度，也不再次平均模型。"""
    tag = f"model_{number:02d}"
    prepared_path = ROOT / f"quality_control/prepared/{tag}.json"
    require_current_design(read_json(prepared_path))
    prepared = record(prepared_path, sources=True)
    sources = hashes([prepared_path, ROOT / "quality_control/reference.json"])
    folder = ROOT / f"rank_inputs/{tag}"
    jobs = pd.read_csv(folder / "jobs.csv")
    require(jobs.neuron_index.tolist() == list(range(64)), "权重评分行序错误。")
    weight = np.fromfile(folder / "scores.f64.bin", dtype="<f8").reshape(64, 58581)
    wranks, wvalid = centered_ranks(weight)
    # 先验证每组织输入，避免部分文件改变后仍跳过一个旧的完成结果。
    for tid in tissues.tissue_id:
        path = Path(f"step5/quality_control/prepared/{tag}/{tid}.json")
        p = record(path)
        require(p["configuration"]["model_number"] == number and p["configuration"]["source_tissue_id"] == tid
                and p["configuration"]["n_candidates"] == 64 and p["configuration"]["n_genes"] == 58581,
                "Jacobian 排序任务配置错误。")
        require(p["source_sha256"][f"step4/quality_control/{tag}/model.json"] ==
                prepared["source_sha256"][f"step4/quality_control/{tag}/model.json"], "两路线不是同一模型来源。")
        require(p["source_sha256"]["step5/quality_control/reference.json"] ==
                read_json(ROOT / "quality_control/reference.json")["source_sha256"]["step5/quality_control/reference.json"],
                "两路线不是同一参考库。")
        sources.update(hashes([path]))
        sources.update(p["output_sha256"])
    cfg = config(sources, model_number=number)
    output = ROOT / f"statistics/rank_agreement/{tag}.csv"
    report = ROOT / f"quality_control/compare/{tag}.json"
    if not begin(report, cfg, [output], overwrite):
        return
    pieces = []
    for tissue in tissues.itertuples(index=False):
        directory = Path(f"step5/rank_inputs/{tag}/{tissue.tissue_id}")
        jjobs = pd.read_csv(directory / "jobs.csv")
        require(jjobs.neuron_index.tolist() == jobs.neuron_index.tolist()
                and jjobs.matrix_row.tolist() == list(range(64)), "两路线神经元行映射不同。")
        jac = np.fromfile(directory / "scores.f64.bin", dtype="<f8").reshape(64, 58581)
        jranks, jvalid = centered_ranks(jac)
        rho = np.clip(np.einsum("ij,ij->i", wranks, jranks), -1, 1)
        pieces.append(pd.DataFrame({"model_number": number, "neuron_index": np.arange(64), "unit_id": jobs.unit_id,
            "source_tissue_id": tissue.tissue_id, "source_tissue": tissue.tissue_name,
            "score_source_a": "weight", "score_source_b": "jacobian_signed_mean", "n_genes": 58581,
            "spearman_rho": rho, "status": np.where(wvalid & jvalid, "ok", "undefined_constant_scores")}))
        LOGGER.info("Rank agreement: %s / %s", tag, tissue.tissue_id)
    table = pd.concat(pieces, ignore_index=True)
    write_csv(table, output)
    finish(report, cfg, [output], n_comparisons=len(table), n_valid=int(table.spearman_rho.notna().sum()))


def weight_statistics(raw: pd.DataFrame, tissue_map: pd.DataFrame):
    """权重法的组织是激活配对语境；每个神经元只有一份与组织无关的排序。

    加入CML结构性缺失行，使组织覆盖与Step5一致。BH仅覆盖53个参考组织，
    且仅使用本路线的p值；同名神经元跨模型不配对。
    """
    grid = pd.MultiIndex.from_product([range(1, 21), range(64), tissue_map.tissue_id],
                                     names=["model_number", "neuron_index", "activation_tissue_id"]).to_frame(index=False)
    names = tissue_map.set_index("tissue_id").tissue_name
    grid["activation_tissue"] = grid.activation_tissue_id.map(names)
    matched = grid.merge(raw, left_on=["model_number", "neuron_index", "activation_tissue_id"],
                         right_on=["model_number", "neuron_index", "reference_tissue_id"], how="left", validate="one_to_one")
    matched["reference_available"] = matched.reference_tissue_id.notna()
    require(matched.reference_available.sum() == 67840, "权重主要检验族必须恰为 67840。")
    matched["unit_id"] = matched.neuron_index.map(lambda u: f"unit_{u:05d}")
    matched["score_source"] = "weight"
    matched["ranking_scope"] = "tissue_independent"
    matched.loc[~matched.reference_available, "estimate_status"] = "no_matching_reference"
    matched = matched.rename(columns={"padj": "q_within_ranking"})
    matched["q_matched_global"] = bh_planned(matched.pval.to_numpy(), matched.reference_available.to_numpy())
    matched["significant"] = matched.q_matched_global.lt(.05)
    matched["primary_family_size"], matched["correction_family"] = 67840, "weight_only"
    # 使用相同的稳定性定义；临时列名只用于调用纯统计函数。
    compatible = matched.rename(columns={"activation_tissue_id": "source_tissue_id", "activation_tissue": "source_tissue"})
    per_model, across = stability_tables(compatible)
    for table in (per_model, across):
        table.rename(columns={"source_tissue_id": "activation_tissue_id", "source_tissue": "activation_tissue"}, inplace=True)
        table["score_source"], table["correction_family"] = "weight", "weight_only"
    return matched, per_model, across


def ngci_tables(matched: pd.DataFrame, activations: pd.DataFrame):
    """激活来源为已核对的Step6组织配对表；仅取中位数、样本数和标识。"""
    left = matched.loc[matched.reference_available].copy()
    med = activations[["model_number", "neuron_index", "source_tissue_id", "source_tissue", "n_samples", "activation_median", "display_group"]]
    pairs = left.merge(med, left_on=["model_number", "neuron_index", "activation_tissue_id"],
                       right_on=["model_number", "neuron_index", "source_tissue_id"], how="left", validate="one_to_one")
    require(len(pairs) == 67840 and pairs.activation_tissue.eq(pairs.source_tissue).all()
            and np.isfinite(pairs[["ES", "activation_median"]].to_numpy()).all(),
            "53组织配对不完整或ES无效；不会按缺失p/NES删除组织。")
    pairs = pairs.drop(columns=["source_tissue_id", "source_tissue"])
    neurons = []
    for (model, unit), part in pairs.groupby(["model_number", "neuron_index"], sort=True):
        require(len(part) == 53 and part.activation_tissue_id.nunique() == 53, "Pearson需要完整53个唯一组织。")
        neurons.append({"model_number": int(model), "neuron_index": int(unit), "unit_id": f"unit_{unit:05d}",
            "score_source": "weight", "gsea_score": "ES", "n_tissues": 53, "degrees_of_freedom": 51,
            **pearson_result(part.activation_median.to_numpy(), part.ES.to_numpy()),
            "alternative": "two-sided", "multiple_testing_correction": "none", "p_value_used_for_selection": False})
    neurons = pd.DataFrame(neurons)
    neurons["p_lt_0_05"] = neurons.p_lt_0_05.astype("boolean")
    summaries = []
    for model, part in neurons.groupby("model_number", sort=True):
        r = part.ngci.dropna()
        q1, median, q3 = r.quantile([.25, .5, .75], interpolation="linear")
        summaries.append({"model_number": model, "score_source": "weight", "n_neurons": len(part), "n_valid": len(r),
            "n_undefined": len(part) - len(r), "ngci_median": median, "ngci_q1": q1, "ngci_q3": q3, "ngci_iqr": q3 - q1,
            "ngci_min": r.min(), "ngci_max": r.max(), "n_p_lt_0_05": int(part.p_lt_0_05.sum()),
            "fraction_p_lt_0_05_among_valid": float(part.p_lt_0_05.sum()) / len(r) if len(r) else np.nan})
    return pairs, neurons, pd.DataFrame(summaries)


def plot_ngci(pairs, neurons, path: Path, qa: bool = False) -> None:
    """简洁主图：全部53组织，固定六组织用颜色和标识突出，不按p选择。"""
    group = pairs.loc[pairs.model_number.eq(1) & pairs.neuron_index.eq(0)]
    result = neurons.loc[neurons.model_number.eq(1) & neurons.neuron_index.eq(0)].iloc[0]
    selection = pd.read_csv(ROOT / "reference/display_selection.csv")
    fig, (ax, text_ax) = plt.subplots(1, 2, figsize=(10, 5.6), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    colors = {"other": "#939b92", "high": "#ba4b36", "low": "#287d8e"}
    for category in ("other", "high", "low"):
        part = group.loc[group.display_group.eq(category)]
        ax.scatter(part.ES, part.activation_median, color=colors[category], s=35, zorder=3)
    ax.set(xlabel="Tissue GSEA ES (weight)", ylabel="Median neuron activation", xlim=(-1.08, 1.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("QA SYNTHETIC" if qa else "Activation and weight-based enrichment", fontsize=13)
    text_ax.axis("off")
    text_ax.text(0, .98, f"Pearson r = {result.pearson_r:.6f}\np = {result.pearson_p:.6g}\nn = 53 tissues",
                 va="top", transform=text_ax.transAxes, fontsize=11, linespacing=1.6)
    for category, top in (("high", .65), ("low", .34)):
        text_ax.text(0, top, category.capitalize() + " activation", color=colors[category], transform=text_ax.transAxes)
        for i, row in enumerate(selection.loc[selection.group.eq(category)].sort_values("rank_in_group").itertuples(index=False)):
            point = group.loc[group.activation_tissue_id.eq(row.source_tissue_id)].iloc[0]
            label = f"{category[0].upper()}{row.rank_in_group}"
            # 标签沿竖向错开，长组织名只写在图旁，避免遮挡散点。
            offset = (6, 7 if i % 2 == 0 else -13)
            ax.annotate(label, (point.ES, point.activation_median), xytext=offset, textcoords="offset points", fontsize=8, color=colors[category])
            text_ax.text(0, top - .055 - i * .073, label + "  " + row.source_tissue,
                         fontsize=8, va="top", transform=text_ax.transAxes)
    output_path(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    if not qa:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def require_finalize_inputs() -> None:
    """先检查正式结果是否齐全；R自检输出不能代替20模型GSEA结果。

    此处只检查文件存在及完成状态，让缺少前置阶段时立即给出操作提示。
    通过后，finalize仍逐项核对指纹、模型/组织映射及统计量，不放宽校验。
    """
    pending = []
    for stage, label, output_folder, suffix in (
        ("fgsea", "正式 R GSEA", "gsea_results", ".csv.gz"),
        ("compare", "Python compare", "statistics/rank_agreement", ".csv"),
    ):
        missing = []
        for number in range(1, N_MODELS + 1):
            tag = f"model_{number:02d}"
            report = ROOT / f"quality_control/{stage}/{tag}.json"
            result = ROOT / output_folder / f"{tag}{suffix}"
            if not report.is_file() or read_json(report).get("status") != "complete" or not result.is_file():
                missing.append(tag)
        if missing:
            pending.append(f"{label}：{N_MODELS - len(missing)}/{N_MODELS} 个模型就绪；未就绪：" + ", ".join(missing))
    require(not pending, "finalize 尚不能运行，前置结果未完成。\n" + "\n".join(pending) +
            '\n若缺少正式 R GSEA，请在 R 中设置 options(step7.mode="run", step7.model_start=1L, '
            'step7.model_end=20L, step7.overwrite=FALSE)，然后运行 '
            'source("step7/code/01_run_fgsea_parallel.R", encoding="UTF-8")。'
            '\nself-test 只生成自检结果；等待正式 R 计算及 Python compare 全部完成后，再运行 finalize。'
            '\n--overwrite 不会生成缺少的前置结果，也无需因此重跑已完成的 prepare 或 compare。')


def finalize(overwrite: bool) -> None:
    """全部R结果完成后才进行权重独立BH；Step5校正结果仅只读配对。"""
    _, _, refs = reference_data()
    require_finalize_inputs()
    sources = upstream()
    tissues = pd.read_csv(ROOT / "reference/tissue_map.csv")
    lock_hash = sha(ROOT / "environment/r_environment_lock.json")
    raw_parts, comparisons = [], []
    for number in range(1, 21):
        tag = f"model_{number:02d}"
        prepared_path = ROOT / f"quality_control/prepared/{tag}.json"
        require_current_design(read_json(prepared_path))
        record(prepared_path, sources=True)
        rpath = ROOT / f"quality_control/fgsea/{tag}.json"
        saved = record(rpath, sources=True)
        require(saved["model_number"] == number and saved["environment_lock_sha256"] == lock_hash,
                "R来源模型或版本不匹配。")
        frame = pd.read_csv(ROOT / f"gsea_results/{tag}.csv.gz", float_precision="round_trip")
        expected = {(u, t) for u in range(64) for t in refs.reference_tissue_id}
        actual = list(zip(frame.neuron_index, frame.reference_tissue_id))
        require(len(actual) == 3392 and set(actual) == expected and frame.model_number.eq(number).all(), "R结果覆盖错误。")
        require(frame.unit_id.eq(frame.neuron_index.map(lambda u: f"unit_{u:05d}")).all()
                and frame.seed.eq(frame.neuron_index.map(lambda u: seed_for(number, u))).all(), "R神经元或种子映射错误。")
        require(frame.reference_tissue.eq(frame.reference_tissue_id.map(refs.set_index("reference_tissue_id").reference_tissue)).all()
                and frame["size"].eq(frame.reference_tissue_id.map(refs.set_index("reference_tissue_id").n_matched)).all(), "R参考组织或集合大小错误。")
        require(frame.score_source.eq("weight").all(), "结果不是权重来源。")
        for column in ("ES", "NES", "pval", "padj", "log2err"):
            require(not np.isinf(frame[column]).any(), f"{column}包含无穷值。")
        for column in ("pval", "padj"):
            require(frame[column].dropna().between(0, 1).all(), f"{column}越界。")
        require(frame.ES.dropna().abs().le(1 + 1e-10).all(), "ES越界。")
        raw_parts.append(frame)
        cpath = ROOT / f"quality_control/compare/{tag}.json"
        require_current_design(read_json(cpath))
        comp = record(cpath, sources=True)
        cf = pd.read_csv(ROOT / f"statistics/rank_agreement/{tag}.csv", float_precision="round_trip")
        expected_comparisons = {(u, t) for u in range(64) for t in tissues.tissue_id}
        require(len(cf) == 3456 and cf.n_genes.eq(58581).all() and cf.model_number.eq(number).all()
                and set(zip(cf.neuron_index, cf.source_tissue_id)) == expected_comparisons
                and cf.unit_id.eq(cf.neuron_index.map(lambda u: f"unit_{u:05d}")).all(), "排序一致性结果映射错误。")
        require(cf.source_tissue.eq(cf.source_tissue_id.map(tissues.set_index("tissue_id").tissue_name)).all(),
                "排序一致性组织名称错误。")
        require(cf.spearman_rho.dropna().between(-1, 1).all()
                and cf.status.eq("ok").equals(cf.spearman_rho.notna()), "排序一致性数值或状态错误。")
        comparisons.append(cf)
        sources.update(hashes([rpath, cpath]))
        sources.update(saved["output_sha256"])
        sources.update(comp["output_sha256"])
    sources.update(hashes([ROOT / "quality_control/reference.json"]))
    matched, stability_model, stability_across = weight_statistics(pd.concat(raw_parts, ignore_index=True),
        tissues)
    pairs, neurons, summaries = ngci_tables(matched, pd.read_csv("step6/statistics/ngci_tissue_pairs.csv", float_precision="round_trip"))
    jac_ngci = pd.read_csv("step6/statistics/ngci_per_neuron.csv", float_precision="round_trip")
    comparison = neurons[["model_number", "neuron_index", "unit_id", "ngci", "pearson_p", "status"]].merge(
        jac_ngci[["model_number", "neuron_index", "unit_id", "ngci", "pearson_p", "status"]],
        on=["model_number", "neuron_index", "unit_id"], validate="one_to_one", suffixes=("_weight", "_jacobian"))
    require(len(comparison) == 1280, "NGCI两路线未完整配对。")
    comparison["delta_ngci_weight_minus_jacobian"] = comparison.ngci_weight - comparison.ngci_jacobian
    agreement = pd.concat(comparisons, ignore_index=True)
    agg = agreement.groupby(["model_number", "neuron_index", "unit_id"], sort=True).spearman_rho
    agreement_summary = agg.agg(n_valid="count", median="median", minimum="min", maximum="max").reset_index()
    agreement_summary["n_tissues_planned"] = 54
    cfg = config(sources)
    frames = {"matched_gsea": matched, "stability_per_model": stability_model, "stability_across_models": stability_across,
              "ngci_tissue_pairs": pairs, "ngci_per_neuron": neurons, "ngci_model_summary": summaries,
              "ngci_method_comparison": comparison, "rank_agreement_all": agreement, "rank_agreement_per_neuron": agreement_summary}
    outputs = [ROOT / f"statistics/{name}.csv" for name in frames]
    base = ROOT / "figures/model_01/unit_00000_weight_ngci"
    outputs += [base.with_suffix(".png"), base.with_suffix(".pdf")]
    path = ROOT / "quality_control/finalize.json"
    if not begin(path, cfg, outputs, overwrite):
        return
    for frame, dest in zip(frames.values(), outputs):
        write_csv(frame, dest)
    plot_ngci(pairs, neurons, base)
    finish(path, cfg, outputs, n_primary_tests=67840, n_weight_rankings=1280,
           n_missing_primary_pvalues=int(matched.loc[matched.reference_available, "pval"].isna().sum()),
           n_ngci=len(neurons), n_rank_comparisons=len(agreement), correction_family="weight_only")
    LOGGER.info("Finalized weight-only BH, stability, NGCI and method comparisons.")


def self_test(device: torch.device) -> None:
    """真实模型检查带符号入口、评估模式与确定性；统计检查只使用合成数据。"""
    decoder, sources, audit = load_decoder(1, device)
    cp = torch.load("step1/core_models/model_01.pt", map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
    # 评估模式传播前后，全部非偏置参数和BN缓冲区都必须与检查点一致。
    for key, value in decoder.state_dict().items():
        if not key.endswith(".bias"):
            np.testing.assert_array_equal(value.cpu().numpy(), cp["decoder." + key].numpy())
    captured = {}

    def capture(name):
        def hook(module, arguments, output):
            require(not torch.is_grad_enabled() and not output.requires_grad, "传播记录了梯度。")
            captured[name] = (arguments[0].detach().cpu().numpy().copy(), output.detach().cpu().numpy().copy())
        return hook

    hooks = [decoder[index].register_forward_hook(capture(index)) for index in (1, 2, 3, 5, 6, 7)]
    try:
        entry, raw, audit["propagation"] = propagate(decoder, propagation_seed_for(1))
    finally:
        for hook in hooks:
            hook.remove()
    expected_entry = (cp["decoder.0.weight"].T * 10000).numpy()
    np.testing.assert_array_equal(entry, expected_entry)
    require(np.any(entry < 0) and np.any(entry > 0), "真实模型入口应保留正、负权重。")
    require(all(not module.training for module in decoder.modules()), "模型未完整进入eval。")
    require(all(module.bias.count_nonzero() == 0 for module in decoder.modules() if isinstance(module, (nn.Linear, nn.BatchNorm1d))), "偏置归零失败。")
    require(all(parameter.grad is None and not parameter.requires_grad for parameter in decoder.parameters()), "参数梯度状态错误。")
    for key, value in decoder.state_dict().items():
        if not key.endswith(".bias"):
            np.testing.assert_array_equal(value.cpu().numpy(), cp["decoder." + key].numpy())
    require(cp["decoder.8.bias"].count_nonzero() > 0, "检查点偏置被修改。")
    # 两个Dropout都必须是恒等映射，既不丢弃元素，也不进行额外缩放。
    for index in (3, 7):
        incoming, outgoing = captured[index]
        np.testing.assert_array_equal(outgoing, incoming)
    require(audit["propagation"]["module_call_counts"] ==
            {f"decoder.{index}": int(index > 0) for index in range(9)}, "实际传播层次错误。")
    # 第一组BN必须直接接收带符号512维权重列；第一组激活输出接到Dropout。
    np.testing.assert_array_equal(captured[1][0], entry)
    for bn_index, activation_index, dropout_index in ((1, 2, 3), (5, 6, 7)):
        incoming, outgoing = (a.astype(np.float64) for a in captured[bn_index])
        prefix = f"decoder.{bn_index}"
        running_mean = cp[prefix + ".running_mean"].numpy().astype(np.float64)
        running_var = cp[prefix + ".running_var"].numpy().astype(np.float64)
        expected_bn = (incoming - running_mean) / np.sqrt(running_var + decoder[bn_index].eps)
        expected_bn *= cp[prefix + ".weight"].numpy()
        np.testing.assert_allclose(outgoing, expected_bn, rtol=3e-5, atol=1e-5)
        np.testing.assert_array_equal(captured[activation_index][0], captured[bn_index][1])
        expected_activation = np.where(outgoing >= 0, outgoing, .2 * outgoing)
        np.testing.assert_allclose(captured[activation_index][1], expected_activation, rtol=2e-7, atol=1e-7)
        np.testing.assert_array_equal(captured[dropout_index][0], captured[activation_index][1])
    expected_linear = captured[3][1].astype(np.float64) @ cp["decoder.4.weight"].numpy().astype(np.float64).T
    np.testing.assert_allclose(captured[5][0], expected_linear, rtol=2e-5, atol=.005)
    expected_output = captured[7][1][:2].astype(np.float64) @ cp["decoder.8.weight"][:17].numpy().astype(np.float64).T
    np.testing.assert_allclose(raw[:2, :17], expected_output, rtol=2e-5, atol=1e-4)
    del decoder, cp, captured
    # 每轮重新加载分析副本并额外消耗RNG；相同或不同种子均须逐元素相等。
    for seed in (propagation_seed_for(1), propagation_seed_for(1) + 1):
        copy, _, _ = load_decoder(1, device)
        torch.rand(37, device=device)
        repeat_entry, repeated, _ = propagate(copy, seed)
        np.testing.assert_array_equal(entry, repeat_entry)
        np.testing.assert_array_equal(raw, repeated)
        del copy, repeated, repeat_entry
    check_hashes(sources)  # 多次传播后，模型文件及其来源记录必须未改变。
    features = pd.read_csv("step5/reference/feature_to_gene.csv", keep_default_na=False)
    genes = pd.read_csv("step5/reference/gene_map.csv", keep_default_na=False)
    merged = merge_scores(raw, features, genes.n_features.to_numpy())
    for index in np.flatnonzero(genes.n_features.to_numpy() > 1)[:10]:
        np.testing.assert_allclose(merged[:, index], raw[:, features.merged_column.eq(index)].mean(1, dtype=np.float64), rtol=1e-14)
    test_x = np.array([[3., 1, 1, 9, -2], [1, 1, 1, 1, 1]])
    test_y = np.array([[1., 3, 3, 4, -1], [0, 1, 2, 3, 4]])
    x, valid = centered_ranks(test_x)
    y, _ = centered_ranks(test_y)
    np.testing.assert_allclose(x[0] @ y[0], stats.spearmanr(test_x[0], test_y[0]).statistic, atol=1e-14)
    require(not valid[1] and np.isnan(x[1]).all(), "常量排序处理错误。")
    np.testing.assert_allclose(bh_planned(np.array([.01, .04, .03, .002]), np.ones(4, dtype=bool)), [.02, .04, .04, .008])
    missing = bh_planned(np.array([.01, np.nan, .03]), np.ones(3, dtype=bool))
    np.testing.assert_allclose(missing[[0, 2]], [.03, .045])
    require(np.isnan(missing[1]), "缺失p被错误填充。")
    # 用完整的合成任务网格验证表连接、独立BH族和NGCI，数值不会写入正式统计目录。
    tissue_map = pd.read_csv("step5/reference/tissue_map.csv")
    refs = pd.read_csv("step5/reference/reference_sets.csv")
    synthetic = pd.MultiIndex.from_product([range(1, 21), range(64), refs.reference_tissue_id],
        names=["model_number", "neuron_index", "reference_tissue_id"]).to_frame(index=False)
    tissue_index = synthetic.reference_tissue_id.map(dict(zip(refs.reference_tissue_id, range(53)))).to_numpy()
    x = np.linspace(-1, 1, 53)
    x /= np.linalg.norm(x)
    residual = x ** 2 - np.mean(x ** 2)
    residual -= np.dot(x, residual) * x
    residual /= np.linalg.norm(residual)
    # 正、负神经元的目标相关系数分别为+0.6/-0.6，避免只验证正相关。
    direction = np.where(synthetic.neuron_index.to_numpy() % 2 == 0, 1., -1.)
    synthetic["ES"] = direction * .6 * x[tissue_index] + .8 * residual[tissue_index]
    synthetic["NES"] = 2 * synthetic.ES
    synthetic["reference_tissue"] = synthetic.reference_tissue_id.map(refs.set_index("reference_tissue_id").reference_tissue)
    synthetic["pval"] = np.tile([.0001, .02, .3, .9], len(synthetic) // 4)
    synthetic.loc[0, ["pval", "NES"]] = np.nan
    synthetic["padj"], synthetic["estimate_status"] = np.nan, "synthetic"
    matched, stability, across = weight_statistics(synthetic, tissue_map)
    require(len(matched) == 69120 and len(stability) == 1080 and len(across) == 54, "合成结果表数量错误。")
    eligible = matched.reference_available.to_numpy()
    require((~eligible).sum() == 1280 and matched.loc[~eligible, "q_matched_global"].isna().all(),
            "缺少参考集合的组织进入了校正族。")
    # SciPy提供独立BH实现；缺失p以1参与分母，但报告时仍保留NA。
    weight_p = matched.loc[eligible, "pval"].to_numpy()
    expected_q = stats.false_discovery_control(np.nan_to_num(weight_p, nan=1.), method="bh")
    expected_q[~np.isfinite(weight_p)] = np.nan
    np.testing.assert_allclose(matched.loc[eligible, "q_matched_global"], expected_q, equal_nan=True)
    pooled_q = stats.false_discovery_control(np.r_[np.nan_to_num(weight_p, nan=1.), np.zeros(len(weight_p))])[:len(weight_p)]
    require(np.nanmax(np.abs(pooled_q - expected_q)) > .01, "测试数据不能区分独立族和错误的混合族。")
    # 在每个模型内重标神经元不会改变跨模型稳定性；禁止依赖跨模型同号神经元。
    permuted = synthetic.copy()
    permuted["neuron_index"] = (permuted.neuron_index + 3 * permuted.model_number) % 64
    _, permuted_stability, permuted_across = weight_statistics(permuted, tissue_map)
    pd.testing.assert_frame_equal(stability, permuted_stability)
    pd.testing.assert_frame_equal(across, permuted_across)
    activations = synthetic[["model_number", "neuron_index", "reference_tissue_id", "reference_tissue"]].rename(
        columns={"reference_tissue_id": "source_tissue_id", "reference_tissue": "source_tissue"})
    activations["activation_median"], activations["n_samples"], activations["display_group"] = x[tissue_index], 10, "other"
    # 打乱输入行顺序，检查按组织ID连接，而非按表格位置连接。
    pairs, neurons, summaries = ngci_tables(matched, activations.iloc[::-1])
    require(len(pairs) == 67840 and len(neurons) == 1280 and len(summaries) == 20
            and neurons.n_tissues.eq(53).all(), "NGCI配对数或结果表数量错误。")
    np.testing.assert_allclose(neurons.ngci, np.where(neurons.neuron_index % 2 == 0, .6, -.6), atol=1e-14)
    # 独立用t分布检验双侧p；缺失GSEA p/NES不能删除仍有有效ES的组织。
    expected_p = 2 * stats.t.sf(.6 * np.sqrt(51 / (1 - .6 ** 2)), df=51)
    np.testing.assert_allclose(neurons.pearson_p, expected_p, rtol=1e-12)
    require(pairs.pval.isna().sum() == 1 and pairs.NES.isna().sum() == 1, "NGCI误删了p/NES缺失行。")
    require(pearson_result(np.ones(53), x)["status"] == "undefined_constant_activation", "常量激活处理错误。")
    seeds = [seed_for(m, u) for m in range(1, 21) for u in range(64)]
    require(len(set(seeds)) == 1280 and min(seeds) > 629 + 5000000 + 20 * 54 * 64, "派生种子重复。")
    propagation_seeds = [propagation_seed_for(m) for m in range(1, 21)]
    require(len(set(propagation_seeds)) == 20 and not set(propagation_seeds).intersection(seeds), "传播种子与GSEA种子冲突。")
    sources.update(hashes([Path("step5/reference") / name for name in ("gene_map.csv", "feature_to_gene.csv", "tissue_map.csv", "reference_sets.csv")]))
    write_json(ROOT / "quality_control/self_test.json", {"status": "passed", "configuration": config(sources, device=str(device)),
        "checks": ["signed weight column orientation", "checkpoint unmodified", "zero linear bias and BN beta",
        "all trained BN buffers unchanged", "eval mode and no_grad",
        "decoder[1:] only; each BN and LeakyReLU called once; first Linear not repeated",
        "both Dropout modules are identity", "both BN formulas use trained running statistics",
        "float64 independent layer formulas", "fresh-copy equality despite prior RNG use",
        "same and different seeds give identical output; one full64 batch per pass",
                   "same-name mean", "Spearman average ties and constants", "BH planned denominator with missing p",
                   "weight-only BH vs independent SciPy and pooled-family counterexample", "structural missing reference excluded",
                   "stability invariant to within-model neuron permutation", "53-tissue ID join with shuffled rows",
                   "signed NGCI and two-sided t-test equivalence", "missing GSEA p/NES retained for NGCI", "unique derived seeds"],
        "max_absolute_formula_difference": float(np.max(np.abs(raw[:2, :17] - expected_output))), "audit": audit})
    LOGGER.info("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step7 weight propagation, NGCI and rank agreement")
    parser.add_argument("stage", choices=("check", "self-test", "prepare", "compare", "finalize"))
    parser.add_argument("--model-start", type=int, default=1)
    parser.add_argument("--model-end", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.chdir(TRIAL_ROOT)
    output_path(ROOT / "logs" / f"{args.stage}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / f"logs/{args.stage}.log", encoding="utf-8"), logging.StreamHandler()])
    require(1 <= args.model_start <= args.model_end <= 20, "模型范围必须在1-20。")
    require(sklearn.__version__ == "1.9.0", "需要 scikit-learn==1.9.0。")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.stage == "self-test":
        self_test(device)
    elif args.stage == "finalize":
        require(args.model_start == 1 and args.model_end == 20, "统一BH必须覆盖20模型。")
        finalize(args.overwrite)
    elif args.stage == "compare":
        reference_data()
        tissues = pd.read_csv(ROOT / "reference/tissue_map.csv")
        require(len(tissues) == 54 and tissues.tissue_id.is_unique, "需要54个唯一组织。")
        for number in range(args.model_start, args.model_end + 1):
            compare_model(number, tissues, args.overwrite)
    else:
        sources = upstream()
        if args.stage == "check":
            for number in range(args.model_start, args.model_end + 1):
                decoder, more, _ = load_decoder(number, device)
                sources.update(more)
                del decoder
                LOGGER.info("Check model_%02d passed", number)
            write_json(ROOT / "quality_control/input_check.json", {"status": "passed", "configuration": config(sources),
                "model_start": args.model_start, "model_end": args.model_end})
        else:
            prepare_reference(sources, args.overwrite)
            genes, features, _ = reference_data()
            for number in range(args.model_start, args.model_end + 1):
                prepare_model(number, device, genes, features, args.overwrite)


if __name__ == "__main__":
    main()

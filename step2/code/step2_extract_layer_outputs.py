"""Step2：逐模型提取线性层和审计层输出，并计算完整样本的组织中位数。

从 trial 目录执行（所有数据路径相对于 trial）：
    python step2/code/step2_extract_layer_outputs.py check
    python step2/code/step2_extract_layer_outputs.py smoke-test
    python step2/code/step2_extract_layer_outputs.py extract

六个主层均取 nn.Linear 的直接输出，隐藏层尚未经过 BN 或 LeakyReLU。
审计层独立保存 BN/LeakyReLU 输出，名称和模块位置都写入 layer_definitions.csv。
前五个主层及五个审计层保存 float32 NPY；最后的 60498 维主层仅保存组织
中位数 CSV。每个组织的全部样本参加一次精确中位数计算，不能用 batch
中位数的中位数代替。20 个模型分别处理，输入数据不再进行尺度变换。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
from pathlib import Path
import os
import platform
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import nn


# 路径从脚本所在目录推导，文件中不写机器专属路径。复用 Step1 的模型类，
# 使网络结构与训练时保持一致；导入模块不会触发其 main() 或训练过程。
TRIAL_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(TRIAL_ROOT))
from step1.code.step1_core_experiment import Autoencoder  # noqa: E402

STEP1_META = Path("step1/core_metadata")
MATRIX_PATH = Path("step1/core_preprocessed/log2_tpm_plus_1.npy")
STEP2_ROOT = Path("step2")
N_SAMPLES, N_GENES, N_TISSUES = 7862, 60498, 54
MASTER_SEED, N_MODELS, FINAL_EPOCH = 629, 20, 100


@dataclass(frozen=True)
class LayerSpec:
    """同时记录线性变换形状与实际捕获模块，避免将两种编号混用。

    in_features/out_features 描述所属 Linear 的输入/输出维度；审计模块的
    输入和输出宽度均为 out_features。capture_module 为零基模块路径。
    """

    name: str
    linear_module: str
    capture_module: str
    in_features: int
    out_features: int
    module_type: str
    role: str
    save_raw: bool = True


LAYERS = (
    LayerSpec("encoder_linear_60498_to_2048", "encoder.0", "encoder.0", 60498, 2048, "Linear", "primary"),
    LayerSpec("encoder_linear_2048_to_512", "encoder.4", "encoder.4", 2048, 512, "Linear", "primary"),
    LayerSpec("encoder_linear_512_to_64", "encoder.8", "encoder.8", 512, 64, "Linear", "primary"),
    LayerSpec("decoder_linear_64_to_512", "decoder.0", "decoder.0", 64, 512, "Linear", "primary"),
    LayerSpec("decoder_linear_512_to_2048", "decoder.4", "decoder.4", 512, 2048, "Linear", "primary"),
    LayerSpec("decoder_linear_2048_to_60498", "decoder.8", "decoder.8", 2048, 60498, "Linear", "primary", False),
    # 全局一基索引 3、6、7、12、15 对应下面五个实际模块。
    # 它们用于检查 BN/激活函数前后的数值，不替代上面的线性层主结果。
    LayerSpec("audit_encoder_leakyrelu_2048", "encoder.0", "encoder.2", 60498, 2048, "LeakyReLU", "audit"),
    LayerSpec("audit_encoder_batchnorm_512", "encoder.4", "encoder.5", 2048, 512, "BatchNorm1d", "audit"),
    LayerSpec("audit_encoder_leakyrelu_512", "encoder.4", "encoder.6", 2048, 512, "LeakyReLU", "audit"),
    LayerSpec("audit_decoder_leakyrelu_512", "decoder.0", "decoder.2", 64, 512, "LeakyReLU", "audit"),
    LayerSpec("audit_decoder_batchnorm_2048", "decoder.4", "decoder.5", 512, 2048, "BatchNorm1d", "audit"),
)
CAPTURE_LAYERS = {spec.capture_module: spec for spec in LAYERS}
FINAL_LAYER = "decoder_linear_2048_to_60498"
LATENT_LAYER = "encoder_linear_512_to_64"


@dataclass
class Inputs:
    """matrix 保持磁盘映射；sample_map 的行与 NPY 输出行一一对应。"""

    matrix: np.ndarray
    sample_map: pd.DataFrame
    gene_order: pd.DataFrame
    architecture: dict


def require(condition: bool, message: str) -> None:
    """一致性检查不能使用会被 python -O 关闭的 assert。"""

    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    """先写临时文件再替换；完成标记不会留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """分块计算来源指纹，避免把模型文件整体读入额外内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> Inputs:
    """按样本 ID 连接标签和划分，同时校验位置列；不再生成标签或划分。"""

    matrix = np.load(MATRIX_PATH, mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (N_SAMPLES, N_GENES) and matrix.dtype == np.float16,
            f"输入矩阵应为 {(N_SAMPLES, N_GENES)} float16，实际为 {matrix.shape} {matrix.dtype}")
    full = pd.read_csv(STEP1_META / "sample_order_full.csv", keep_default_na=False)
    manifest = pd.read_csv(STEP1_META / "sample_manifest.csv", keep_default_na=False)
    require(len(full) == N_SAMPLES and full["sample_id"].is_unique, "完整样本列表数量错误或 ID 重复。")
    require(np.array_equal(full["order"], np.arange(N_SAMPLES)), "完整样本位置必须连续且从 0 开始。")
    require(manifest["Sample"].is_unique and set(manifest["Sample"]) == set(full["sample_id"]),
            "sample_manifest 的样本 ID 与完整样本列表不一致。")
    # reindex 的依据是样本 ID，即使 manifest 的物理行顺序改变，也不会错配。
    aligned = manifest.set_index("Sample").loc[full["sample_id"]].reset_index()
    require(np.array_equal(aligned["expression_column_position"], full["order"]), "样本 ID 与表达矩阵位置不匹配。")
    require(aligned["tissue_name"].nunique() == N_TISSUES and aligned["tissue_name"].str.strip().ne("").all(),
            "组织标签为空或数量不是 54。")
    sample_map = pd.DataFrame({
        "row_index": np.arange(N_SAMPLES),
        "full_data_position": full["order"],
        "sample_id": full["sample_id"],
        "tissue_name": aligned["tissue_name"],
    })
    split_parts = []
    split_config = json.loads((STEP1_META / "sample_split.json").read_text(encoding="utf-8"))
    require(split_config["split_seed"] == MASTER_SEED, "样本划分种子不是 629。")
    for split_name, expected_count in (("train", 6289), ("test", 1573)):
        frame = pd.read_csv(STEP1_META / f"sample_order_{split_name}.csv", keep_default_na=False)
        require(len(frame) == expected_count and frame["sample_id"].is_unique, f"{split_name} 样本数量错误或 ID 重复。")
        require(np.array_equal(frame["order_in_split"], np.arange(expected_count)), f"{split_name} 内部顺序不连续。")
        require(frame["full_data_position"].between(0, N_SAMPLES - 1).all(), f"{split_name} 的完整位置越界。")
        expected_ids = full.iloc[frame["full_data_position"].to_numpy()]["sample_id"].tolist()
        require(frame["sample_id"].tolist() == expected_ids == split_config[f"{split_name}_sample_ids"],
                f"{split_name} 的样本 ID、位置和 JSON 记录不一致。")
        split_parts.append(frame[["sample_id", "order_in_split"]].assign(split=split_name))
    split_map = pd.concat(split_parts, ignore_index=True)
    require(split_map["sample_id"].is_unique and set(split_map["sample_id"]) == set(full["sample_id"]),
            "训练/测试样本存在重叠或未覆盖完整样本。")
    sample_map = sample_map.merge(split_map, on="sample_id", how="left", validate="one_to_one", sort=False)
    require(sample_map["sample_id"].tolist() == full["sample_id"].tolist(), "连接标签后样本顺序改变。")

    genes = pd.read_csv(STEP1_META / "gene_order.csv", keep_default_na=False)
    require(len(genes) == N_GENES and genes["gene_id"].is_unique, "基因 ID 数量错误或重复。")
    require(np.array_equal(genes["expression_row_position"], np.arange(N_GENES)), "基因顺序不连续。")
    architecture = json.loads((STEP1_META / "model_architecture.json").read_text(encoding="utf-8"))
    require(architecture["encoder_dims"] == [60498, 2048, 512, 64]
            and architecture["decoder_dims"] == [64, 512, 2048, 60498], "线性层形状不匹配。")
    require(architecture["leaky_relu_negative_slope"] == 0.2 and architecture["dropout_rate"] == 0.3
            and architecture["batch_norm_position"] == "after_linear_before_activation", "BN、激活函数或 Dropout 配置不匹配。")
    return Inputs(matrix, sample_map, genes, architecture)


def validate_layer_definitions(model: nn.Module) -> None:
    """通过模块类型、顺序和 Linear 的实际形状验证所有捕获位置。"""

    expected_types = (nn.Linear, nn.BatchNorm1d, nn.LeakyReLU, nn.Dropout,
                      nn.Linear, nn.BatchNorm1d, nn.LeakyReLU, nn.Dropout, nn.Linear)
    for network in (model.encoder, model.decoder):
        require(tuple(type(module) for module in network) == expected_types, "模型模块顺序发生改变。")
        for module in network:
            if isinstance(module, nn.LeakyReLU):
                require(module.negative_slope == 0.2 and not module.inplace, "LeakyReLU 参数不匹配。")
            if isinstance(module, nn.Dropout):
                require(module.p == 0.3 and not module.inplace, "Dropout 参数不匹配。")
            if isinstance(module, nn.BatchNorm1d):
                require(module.track_running_stats and module.affine, "BatchNorm 必须保存运行统计量和仿射参数。")
    for spec in LAYERS:
        linear = model.get_submodule(spec.linear_module)
        module = model.get_submodule(spec.capture_module)
        require(isinstance(linear, nn.Linear) and (linear.in_features, linear.out_features) ==
                (spec.in_features, spec.out_features), f"{spec.name} 的线性层形状不匹配。")
        require(type(module).__name__ == spec.module_type, f"{spec.name} 的捕获模块类型不匹配。")


def load_checkpoint(model_number: int, architecture: dict) -> dict:
    """仅加载参数与基本元数据；严格验证 epoch、种子及全部 state_dict 键。"""

    path = Path(f"step1/core_models/model_{model_number:02d}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(checkpoint["model_number"] == model_number, f"{path} 模型编号不匹配。")
    require(checkpoint["model_seed"] == MASTER_SEED + model_number, f"{path} 模型种子不匹配。")
    require(checkpoint["final_epoch"] == FINAL_EPOCH, f"{path} 不是第 100 epoch。")
    require(checkpoint["architecture_config"] == architecture, f"{path} 与结构配置不一致。")
    # meta 模型只分配形状信息，用于检查 20 个 checkpoint，不分配 1 GB 参数。
    with torch.device("meta"):
        template = Autoencoder()
    validate_layer_definitions(template)
    expected = template.state_dict()
    actual = checkpoint["model_state_dict"]
    require(set(actual) == set(expected), f"{path} 参数键缺失或多出。")
    for key, value in expected.items():
        require(actual[key].shape == value.shape and actual[key].dtype == value.dtype,
                f"{path}: {key} 形状或精度错误。")
    return checkpoint


def build_model(checkpoint: dict, device: torch.device) -> nn.Module:
    """加载所有参数和 BN 缓冲区后切换 eval；不执行训练或重新初始化已加载参数。"""

    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    validate_layer_definitions(model)
    return model


def iter_layer_outputs(model: nn.Module, inputs: torch.Tensor):
    """依次前向传播，只返回明确选中的模块；保留 batch=1 的二维形状。"""

    require(not model.training, "提取输出时必须使用 eval 模式。")
    values = inputs
    for network_name in ("encoder", "decoder"):
        for index, module in enumerate(getattr(model, network_name)):
            values = module(values)
            spec = CAPTURE_LAYERS.get(f"{network_name}.{index}")
            if spec is not None:
                require(tuple(values.shape) == (len(inputs), spec.out_features)
                        and values.dtype == torch.float32, f"{spec.name} 输出维度或精度错误。")
                yield spec, values


def feature_columns(spec: LayerSpec) -> list[str]:
    """列编号从 0 开始；最终输出列与 gene_order.csv 的行一一对应。"""

    prefix = "gene" if spec.name == FINAL_LAYER else "unit"
    return [f"{prefix}_{index:05d}" for index in range(spec.out_features)]


def write_definitions(inputs: Inputs, root: Path) -> None:
    """记录六个线性层和五个审计层的编号、保存范围及输出基因顺序。"""

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in LAYERS:
        row = asdict(spec)
        network, index = spec.capture_module.split(".")
        row["global_module_index_1based"] = int(index) + 1 + (9 if network == "decoder" else 0)
        row["capture_point"] = "linear_output_before_bn_activation" if spec.role == "primary" else "module_output"
        row["raw_dtype"] = "float32" if spec.save_raw else "not_saved"
        row["median_scope"] = "full"
        rows.append(row)
    pd.DataFrame(rows).to_csv(metadata / "layer_definitions.csv", index=False)
    gene_map = inputs.gene_order.copy()
    gene_map.insert(0, "output_column", feature_columns(next(s for s in LAYERS if s.name == FINAL_LAYER)))
    gene_map.to_csv(metadata / "output_gene_map.csv", index=False)


def ensure_output_available(root: Path, model_number: int, overwrite: bool) -> None:
    """在更新元数据之前检查输出占用，防止已完成结果被配上新配置。"""

    if overwrite:
        return
    tag = f"model_{model_number:02d}"
    for directory in (root / "raw_outputs" / tag, root / "tissue_medians" / tag):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} 已有内容；请调整模型范围，或显式使用 --overwrite 重算此模型。")
    for path in (root / "quality_control" / f"{tag}.json", root / "metadata" / f"{tag}.json"):
        if path.exists():
            raise FileExistsError(f"{path} 已存在；请确认模型范围，或显式使用 --overwrite 重算此模型。")


def extract_model(model: nn.Module, inputs: Inputs, model_number: int, root: Path,
                  batch_size: int, overwrite: bool, logger: logging.Logger,
                  provenance: dict | None = None) -> dict:
    """按组织推理、按完整位置回写；最终输出仅在单个组织的内存缓冲区存在。

    eval 模式的 BN 使用固定运行统计量，Dropout 关闭，因此组织分批不会改变
    样本的数学映射。浮点计算在不同设备/批量下仍可能有微小舍入差异。
    正式 NPY 第 r 行始终对应 sample_map.csv 第 r 行，不按组织重新排列。
    """

    require(batch_size > 0, "batch_size 必须为正整数。")
    tag = f"model_{model_number:02d}"
    raw_dir, median_dir = root / "raw_outputs" / tag, root / "tissue_medians" / tag
    report_path = root / "quality_control" / f"{tag}.json"
    ensure_output_available(root, model_number, overwrite)
    for directory in (raw_dir, median_dir):
        directory.mkdir(parents=True, exist_ok=True)
    require(np.array_equal(inputs.sample_map["row_index"], np.arange(len(inputs.sample_map))), "输出行号不连续。")
    require(inputs.sample_map["sample_id"].is_unique and inputs.sample_map["full_data_position"].is_unique,
            "输出样本 ID 或完整数据位置重复。")
    sample_map = inputs.sample_map.assign(model_number=model_number, model_seed=MASTER_SEED + model_number)
    tissues = sorted(sample_map["tissue_name"].unique())
    groups = [np.flatnonzero(sample_map["tissue_name"].to_numpy() == name) for name in tissues]
    counts = pd.DataFrame({"tissue_name": tissues, "n_samples": [len(rows) for rows in groups]})
    device = next(model.parameters()).device
    report = {
        "status": "running", "model_number": model_number, "model_seed": MASTER_SEED + model_number,
        "final_epoch": FINAL_EPOCH, "n_samples": len(sample_map), "n_tissues": len(tissues),
        "batch_size": batch_size, "device": str(device), "inference_dtype": "float32",
        "median_scope": "full" if len(sample_map) == N_SAMPLES else "smoke_test_subset",
        "model_mode": "eval", "grad_enabled": False, "final_output_raw_saved": False,
        "inference_order": "tissue_groups", "saved_row_order": "sample_map.row_index",
        "median_method": "numpy.median over every sample in each tissue; no batch aggregation",
    }
    write_json(report_path, report)
    arrays = {}
    medians = {spec.name: np.empty((len(tissues), spec.out_features), dtype=np.float32) for spec in LAYERS}
    extrema = {spec.name: [float("inf"), float("-inf")] for spec in LAYERS}
    visited = np.zeros(len(sample_map), dtype=np.int64)
    try:
        # 先将状态改为 running，再更新模型配置、映射或矩阵。中途停止时，
        # 即使仍有旧文件存在，也不能被下游当作一套已完成的新结果读取。
        if provenance is not None:
            write_json(root / "metadata" / f"{tag}.json", provenance)
        sample_map.to_csv(raw_dir / "sample_map.csv", index=False)
        counts.to_csv(median_dir / "tissue_sample_counts.csv", index=False)
        for spec in LAYERS:
            if spec.save_raw:
                arrays[spec.name] = np.lib.format.open_memmap(
                    raw_dir / f"{spec.name}.npy", mode="w+", dtype=np.float32,
                    shape=(len(sample_map), spec.out_features))
        with torch.no_grad():
            for tissue_index, rows in enumerate(groups):
                # 单个组织的完整输出约数十 MB；不能逐 batch 取中位数再汇总。
                final_values = np.empty((len(rows), N_GENES), dtype=np.float32)
                for start in range(0, len(rows), batch_size):
                    batch_rows = rows[start:start + batch_size]
                    positions = sample_map.iloc[batch_rows]["full_data_position"].to_numpy(dtype=np.int64)
                    values = np.array(inputs.matrix[positions], dtype=np.float32, copy=True)
                    require(np.isfinite(values).all(), "输入 batch 含 NaN 或无穷值。")
                    tensor = torch.from_numpy(values).to(device)
                    for spec, output in iter_layer_outputs(model, tensor):
                        cpu_output = output.detach().cpu().numpy()
                        require(np.isfinite(cpu_output).all(), f"{tag} {spec.name} 含非有限值。")
                        extrema[spec.name][0] = min(extrema[spec.name][0], float(cpu_output.min()))
                        extrema[spec.name][1] = max(extrema[spec.name][1], float(cpu_output.max()))
                        if spec.save_raw:
                            arrays[spec.name][batch_rows] = cpu_output
                        else:
                            final_values[start:start + len(batch_rows)] = cpu_output
                    visited[batch_rows] += 1
                for spec in LAYERS:
                    group_values = arrays[spec.name][rows] if spec.save_raw else final_values
                    # 这里持有完整组织；偶数样本的中位数取中间两个数的均值。
                    medians[spec.name][tissue_index] = np.median(group_values, axis=0)
                del group_values, final_values
                logger.info("%s tissue=%02d/%02d %s n=%d", tag, tissue_index + 1, len(tissues), tissues[tissue_index], len(rows))
        require(np.all(visited == 1), "部分样本未写入或被重复写入。")
        layer_checks = []
        for spec in LAYERS:
            if spec.save_raw:
                arrays[spec.name].flush()
                # 用落盘的逐样本输出重新计算全部组织中位数，检查回写行号。
                recomputed = np.stack([np.median(arrays[spec.name][rows], axis=0) for rows in groups])
                require(np.array_equal(recomputed, medians[spec.name]), f"{spec.name} 中位数与 NPY 不一致。")
            table = pd.DataFrame(medians[spec.name], columns=feature_columns(spec))
            table.insert(0, "n_samples", counts["n_samples"].to_numpy())
            table.insert(0, "tissue_name", tissues)
            path = median_dir / f"{spec.name}.csv"
            # 9 位有效数字足以使 float32 精确往返；统计列需显式排除 tissue_name/n_samples。
            table.to_csv(path, index=False, float_format="%.9g")
            restored = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
            require(restored["tissue_name"].tolist() == tissues
                    and restored["n_samples"].tolist() == counts["n_samples"].tolist(), "中位数 CSV 组织映射不一致。")
            require(list(restored.columns[2:]) == feature_columns(spec)
                    and np.array_equal(restored.iloc[:, 2:].to_numpy(dtype=np.float32), medians[spec.name]),
                    f"{spec.name} CSV 保存后数值不一致。")
            layer_checks.append({"layer": spec.name, "n_features": spec.out_features,
                                 "raw_saved": spec.save_raw, "min": extrema[spec.name][0],
                                 "max": extrema[spec.name][1], "all_finite": True,
                                 "median_csv_roundtrip_verified": True})
        require(not (raw_dir / f"{FINAL_LAYER}.npy").exists(), "最终输出不应存在逐样本 NPY。")
        restored_map = pd.read_csv(raw_dir / "sample_map.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(restored_map, sample_map, check_dtype=False)
        report.update(status="complete", sample_coverage_verified=True, layers=layer_checks)
        write_json(report_path, report)
        return report
    except Exception:
        report["status"] = "failed"
        write_json(report_path, report)
        raise
    finally:
        # Windows 下显式释放映射，保证中断重跑及临时检查目录清理可执行。
        for array in arrays.values():
            array.flush()
            array._mmap.close()


def smoke_test(inputs: Inputs, model_number: int, device: torch.device, logger: logging.Logger) -> dict:
    """真实参数的九样本集成检查；用标准 forward 的 hooks 独立核对捕获逻辑。

    选取两个组织的 5/4 个样本，batch_size=4，覆盖奇数/偶数中位数、末批
    单样本、组织交错及非连续完整位置。临时层输出只写入检查目录并自动清理。
    """

    checkpoint = load_checkpoint(model_number, inputs.architecture)
    model = build_model(checkpoint, device)
    names = inputs.sample_map["tissue_name"].value_counts().index[:2]
    selected = np.concatenate([
        np.flatnonzero(inputs.sample_map["tissue_name"].to_numpy() == names[0])[:5],
        np.flatnonzero(inputs.sample_map["tissue_name"].to_numpy() == names[1])[:4],
    ])
    selected.sort()
    sample_map = inputs.sample_map.iloc[selected].copy().reset_index(drop=True)
    sample_map["row_index"] = np.arange(len(sample_map))
    subset = replace(inputs, sample_map=sample_map)
    # 显式写出模块对应关系，不通过 iter_layer_outputs 生成参考值。
    reference_modules = {
        "encoder_linear_60498_to_2048": model.encoder[0],
        "encoder_linear_2048_to_512": model.encoder[4],
        "encoder_linear_512_to_64": model.encoder[8],
        "decoder_linear_64_to_512": model.decoder[0],
        "decoder_linear_512_to_2048": model.decoder[4],
        "decoder_linear_2048_to_60498": model.decoder[8],
        "audit_encoder_leakyrelu_2048": model.encoder[2],
        "audit_encoder_batchnorm_512": model.encoder[5],
        "audit_encoder_leakyrelu_512": model.encoder[6],
        "audit_decoder_leakyrelu_512": model.decoder[2],
        "audit_decoder_batchnorm_2048": model.decoder[5],
    }
    reference = {}
    handles = []
    bn_before = {name: value.clone() for name, value in model.named_buffers()}
    for name, module in reference_modules.items():
        def capture(_module, _arguments, output, key=name):
            reference[key] = output.detach().cpu().numpy().copy()
        handles.append(module.register_forward_hook(capture))
    try:
        positions = sample_map["full_data_position"].to_numpy(dtype=np.int64)
        tensor = torch.from_numpy(np.array(inputs.matrix[positions], dtype=np.float32, copy=True)).to(device)
        with torch.no_grad():
            latent, reconstruction = model(tensor)
            # 第一层直接用线性公式验证，明确不包含 BatchNorm 或激活变换。
            linear_formula = torch.nn.functional.linear(tensor, model.encoder[0].weight, model.encoder[0].bias)
        np.testing.assert_array_equal(reference["encoder_linear_60498_to_2048"], linear_formula.cpu().numpy())
        np.testing.assert_array_equal(reference[LATENT_LAYER], latent.cpu().numpy())
        np.testing.assert_array_equal(reference[FINAL_LAYER], reconstruction.cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    qc_dir = STEP2_ROOT / "quality_control"
    qc_dir.mkdir(parents=True, exist_ok=True)
    maximum_error = 0.0
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=qc_dir) as temporary:
        root = Path(temporary)
        extract_model(model, subset, model_number, root, 4, False, logger)
        tag = f"model_{model_number:02d}"
        for spec in LAYERS:
            if spec.save_raw:
                saved = np.load(root / "raw_outputs" / tag / f"{spec.name}.npy", allow_pickle=False)
                require(saved.dtype == np.float32, "逐样本输出必须是 float32。")
                np.testing.assert_allclose(saved, reference[spec.name], rtol=1e-4, atol=1e-4)
                maximum_error = max(maximum_error, float(np.max(np.abs(saved - reference[spec.name]))))
            table = pd.read_csv(root / "tissue_medians" / tag / f"{spec.name}.csv", float_precision="round_trip")
            expected = np.stack([
                np.median(reference[spec.name][sample_map["tissue_name"].to_numpy() == tissue], axis=0)
                for tissue in table["tissue_name"]
            ])
            np.testing.assert_allclose(table.iloc[:, 2:].to_numpy(), expected, rtol=1e-4, atol=1e-4)
        saved_files = {path.stem for path in (root / "raw_outputs" / tag).glob("*.npy")}
        require(saved_files == {spec.name for spec in LAYERS if spec.save_raw}, "逐样本输出范围不正确。")
        # 重复执行不能在没有显式选项的情况下覆盖已存在的结果。
        try:
            extract_model(model, subset, model_number, root, 4, False, logger)
        except FileExistsError:
            pass
        else:
            raise AssertionError("已有输出未被保护。")
    for name, value in model.named_buffers():
        require(torch.equal(value, bn_before[name]), f"推理改变了 BN 缓冲区：{name}")
    report = {"status": "passed", "model_number": model_number, "n_samples": 9, "n_tissues": 2,
              "reference": "standard forward hooks and explicit first Linear formula",
              "checked_layers": len(LAYERS), "checked_raw_arrays": 10,
              "odd_even_medians_checked": True, "single_sample_batch_checked": True,
              "bn_buffers_unchanged": True, "existing_output_protection_checked": True,
              "raw_max_abs_error": maximum_error, "rtol": 1e-4, "atol": 1e-4,
              "device": str(device), "temporary_arrays_retained": False}
    write_json(qc_dir / "smoke_test.json", report)
    return report


def main() -> None:
    """命令行统一入口；check、smoke-test 和 extract 共用相同核心函数。"""

    parser = argparse.ArgumentParser(description="Step2: layer outputs and tissue medians")
    parser.add_argument("stage", choices=("check", "smoke-test", "extract"))
    parser.add_argument("--model-start", type=int, default=1)
    parser.add_argument("--model-end", type=int, default=N_MODELS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="重算所选模型并覆盖其已知输出文件")
    args = parser.parse_args()
    require(1 <= args.model_start <= args.model_end <= N_MODELS, "模型范围必须处于 1 到 20。")
    require(args.batch_size > 0, "batch_size 必须为正整数。")
    os.chdir(TRIAL_ROOT)
    log_dir = STEP2_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log_dir / f"{args.stage}.log", encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("step2")
    # 显式关闭 TF32，不启用 autocast；保存精度和矩阵计算精度均为 float32。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    try:
        inputs = load_inputs()
        if args.stage == "extract":
            for number in range(args.model_start, args.model_end + 1):
                ensure_output_available(STEP2_ROOT, number, args.overwrite)
        write_definitions(inputs, STEP2_ROOT)
        if args.stage == "smoke-test":
            logger.info("Smoke test: %s", smoke_test(inputs, args.model_start, device, logger))
            return
        checked = []
        for number in range(args.model_start, args.model_end + 1):
            checkpoint = load_checkpoint(number, inputs.architecture)
            checked.append({"model_number": number, "model_seed": checkpoint["model_seed"], "final_epoch": checkpoint["final_epoch"]})
            del checkpoint
        write_json(STEP2_ROOT / "quality_control/input_check.json", {
            "status": "passed", "n_samples": N_SAMPLES, "n_genes": N_GENES, "n_tissues": N_TISSUES,
            "sample_order_checked": True, "split_checked": True, "models": checked,
            "checkpoint_check": "metadata, keys, shapes and dtypes; no forward pass in check stage"})
        logger.info("Input and checkpoint checks passed for %d models.", len(checked))
        if args.stage == "check":
            return
        # 来源指纹仅在正式提取时计算，checkpoint 按模型分别记录。
        source_paths = [MATRIX_PATH, *sorted(STEP1_META.glob("*.csv")), *sorted(STEP1_META.glob("*.json")),
                        Path("step1/code/step1_core_experiment.py")]
        source_hashes = {path.as_posix(): sha256_file(path) for path in source_paths}
        for number in range(args.model_start, args.model_end + 1):
            checkpoint = load_checkpoint(number, inputs.architecture)
            checkpoint_path = Path(f"step1/core_models/model_{number:02d}.pt")
            model = build_model(checkpoint, device)
            provenance = {
                "model_number": number, "model_seed": MASTER_SEED + number, "final_epoch": FINAL_EPOCH,
                "checkpoint": checkpoint_path.as_posix(), "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_sha256": source_hashes, "script_sha256": sha256_file(Path("step2/code/step2_extract_layer_outputs.py")),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "device": str(device), "tf32": False, "autocast": False,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "architecture": inputs.architecture, "batch_size": args.batch_size,
                "layer_definitions": "step2/metadata/layer_definitions.csv"}
            extract_model(model, inputs, number, STEP2_ROOT, args.batch_size, args.overwrite, logger, provenance)
            del model, checkpoint
            logger.info("Model %02d completed.", number)
    except Exception:
        logger.exception("Step2 stage '%s' failed.", args.stage)
        raise


if __name__ == "__main__":
    main()

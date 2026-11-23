"""
正式复现实验 Step 1：数据冻结、表达预处理、样本划分和浅层自编码器训练。

本文件是本次复现重新编写的核心 Python 实现。它不读取原研究已经保存的
模型、激活输出、Jacobian、相关系数或 LMM 结果；唯一使用的输入是 trial
目录中的原始数据、样本注释和基因参考文件。

使用方式（命令均从 trial 目录执行，路径均为相对路径）：

    python step1/code/step1_core_experiment.py prepare
    python step1/code/step1_core_experiment.py train

也可以一次完成两阶段：

    python step1/code/step1_core_experiment.py all

prepare 阶段会：
1. 冻结表达矩阵的样本 ID、基因 ID 和基因名称顺序；
2. 以表达矩阵样本 ID 为主键，从 GTEX_phenotype 生成并覆盖
   input_data/tsne_label.csv 和 input_data/tissue_appear_times.csv；
3. 按原研究的数学逻辑把存储的 log2(TPM+0.001) 转换为 log2(TPM+1)；
4. 保留 float16 中间结果，并把样本×基因矩阵保存为 NPY 二进制文件；
5. 使用主种子 629 做固定的样本级 80/20 划分，保存三套样本顺序。

train 阶段会：
1. 读取 prepare 阶段保存的 NPY 矩阵；
2. 使用同一训练/测试划分顺序训练 20 个独立模型；
3. 使用 629+model_number 作为模型派生种子（model_number 为 1 到 20）；
4. 保存每个 epoch 的训练/测试 MSE、最终第 100 epoch 的模型 state_dict，
   以及模型配置和实验元数据。

重要边界：
- 本文件不提取层输出，不计算组织中位数；这些属于下一步。
- 同名基因按平均值合并属于 GSEA 输入阶段，本步保持 60,498 维输入。
- 本步不重新进行已有的 log offset 比较；该比较文件保留在 step1 中，
  但不被本文件调用。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 固定的复现实验配置
# ---------------------------------------------------------------------------
MASTER_SEED = 629
N_MODELS = 20
TEST_FRACTION = 0.20
N_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
STORED_OFFSET = 0.001
FINAL_OFFSET = 1.0
FLOAT16_INTERMEDIATE = True


@dataclass(frozen=True)
class Step1Paths:
    """集中保存 Step 1 所有相对路径，避免在代码中散落路径字符串。"""

    trial_root: Path = Path(".")
    expression_file: Path = Path("dataset/gtex_RSEM_gene_tpm")
    phenotype_file: Path = Path("dataset/GTEX_phenotype")
    gene_name_file: Path = Path("input_data/gene_name_60498.csv")
    tsne_label_file: Path = Path("input_data/tsne_label.csv")
    tissue_count_file: Path = Path("input_data/tissue_appear_times.csv")
    step1_root: Path = Path("step1")
    metadata_dir: Path = Path("step1/core_metadata")
    preprocessed_dir: Path = Path("step1/core_preprocessed")
    results_dir: Path = Path("step1/core_results")
    losses_dir: Path = Path("step1/core_results/losses")
    models_dir: Path = Path("step1/core_models")
    logs_dir: Path = Path("step1/core_logs")

    def absolute(self, relative_path: Path) -> Path:
        """把相对路径解析到 trial 根目录；配置本身仍只保存相对路径。"""

        if relative_path.is_absolute():
            raise ValueError(f"Step 1 只接受相对路径：{relative_path}")
        return self.trial_root / relative_path

    def make_output_directories(self) -> None:
        """创建正式核心实验所需的输出目录，不触碰旧尺度比较文件。"""

        for relative_path in (
            self.metadata_dir,
            self.preprocessed_dir,
            self.results_dir,
            self.losses_dir,
            self.models_dir,
            self.logs_dir,
        ):
            self.absolute(relative_path).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitIndices:
    """保存完整样本顺序中的训练集和测试集位置。"""

    train_indices: np.ndarray
    test_indices: np.ndarray


class Step1Error(RuntimeError):
    """用于把输入或复现实验一致性问题明确报告给调用者。"""


def configure_logging(log_path: Path) -> logging.Logger:
    """同时写入日志文件和终端，方便追踪 20 个模型的长时间运行。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step1_core_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def set_global_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机源。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 固定 cuDNN 的算法选择，避免 benchmark 根据运行时状态切换卷积实现。
    # 当前网络以 Linear 和 BatchNorm1d 为主；这里不宣称跨硬件位级一致。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_expression_header(expression_path: Path) -> List[str]:
    """只读取表达矩阵表头，冻结样本 ID 的原始顺序。"""

    with expression_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().rstrip("\r\n")
    fields = header_line.split("\t")
    if len(fields) < 2 or fields[0].strip().lower() != "sample":
        raise Step1Error("表达矩阵第一列表头必须是 sample，且至少包含一个样本列。")
    sample_ids = fields[1:]
    if any(not sample_id for sample_id in sample_ids):
        raise Step1Error("表达矩阵表头存在空样本 ID。")
    if len(sample_ids) != len(set(sample_ids)):
        raise Step1Error("表达矩阵表头存在重复样本 ID。")
    return sample_ids


def standardize_tissue_name(raw_name: str) -> str:
    """把 GTEx 原始组织名称转换为原项目使用的紧凑标签形式。

    原项目的标签示例为 Muscle-Skeletal、WholeBlood 和
    Brain-Anteriorcingulatecortex(BA24)。这里仅删除组织名称中的空格，
    并把组织层级分隔符 `` - `` 变为 ``-``；括号和大小写保持不变。
    """

    cleaned = str(raw_name).strip().replace(" - ", "-")
    return cleaned.replace(" ", "")


def load_and_validate_metadata(
    paths: Step1Paths, sample_ids: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """读取 phenotype 和基因名称，并按表达矩阵顺序进行严格校验。"""

    phenotype_path = paths.absolute(paths.phenotype_file)
    phenotype = pd.read_csv(phenotype_path, sep="\t", dtype=str)
    required_columns = {
        "Sample",
        "body_site_detail (SMTSD)",
        "_gender",
        "_patient",
        "_cohort",
    }
    missing = required_columns.difference(phenotype.columns)
    if missing:
        raise Step1Error(f"phenotype 缺少必要列：{sorted(missing)}")
    if phenotype["Sample"].duplicated().any():
        raise Step1Error("phenotype 中存在重复 Sample ID。")

    indexed = phenotype.set_index("Sample", drop=False).reindex(sample_ids)
    if indexed["Sample"].isna().any():
        missing_samples = indexed.loc[indexed["Sample"].isna()].index.tolist()[:5]
        raise Step1Error(f"有表达矩阵样本没有 phenotype 标签，例如：{missing_samples}")
    if indexed["body_site_detail (SMTSD)"].isna().any():
        raise Step1Error("表达矩阵样本存在空组织标签。")

    indexed = indexed.copy()
    indexed["tissue_name"] = indexed["body_site_detail (SMTSD)"].map(
        standardize_tissue_name
    )
    if indexed["tissue_name"].nunique() != 54:
        raise Step1Error(
            "按原项目名称规则生成的组织数量不是 54："
            f"{indexed['tissue_name'].nunique()}"
        )

    gene_names = pd.read_csv(
        paths.absolute(paths.gene_name_file),
        header=None,
        dtype=str,
        keep_default_na=False,
    ).iloc[:, 0]
    return indexed, phenotype, gene_names


def rebuild_tissue_label_files(
    paths: Step1Paths, sample_manifest: pd.DataFrame
) -> None:
    """按表达矩阵样本顺序覆盖 tsne_label.csv 和 tissue_appear_times.csv。

    tsne_label.csv 保持原项目的两列格式：第一列是 0 到 n-1 的样本位置，
    第二列列名为 0、内容为标准化组织名。tissue_appear_times.csv 保持无表头
    的两列格式，并按样本数从高到低排列。
    """

    labels = pd.Series(
        sample_manifest["tissue_name"].to_numpy(),
        name=0,
        index=np.arange(len(sample_manifest), dtype=np.int64),
    )
    labels.to_csv(
        paths.absolute(paths.tsne_label_file),
        index=True,
        index_label="",
        header=True,
    )

    counts = sample_manifest["tissue_name"].value_counts()
    count_rows = pd.DataFrame({0: counts.index, 1: counts.to_numpy()})
    count_rows.to_csv(
        paths.absolute(paths.tissue_count_file),
        index=False,
        header=False,
    )


def load_expression_matrix(
    paths: Step1Paths, expected_sample_ids: Sequence[str], expected_gene_count: int
) -> Tuple[pd.DataFrame, List[str]]:
    """一次性读取完整表达矩阵，并检查样本/基因顺序。"""

    expression_path = paths.absolute(paths.expression_file)
    # 第一列 sample 实际保存 ENSG 基因 ID，必须保留为字符串。
    # pandas 会把单个 dtype 应用于索引列，所以按列指定类型，
    # 仅将 7,862 个样本表达列读为 float32，不改变任何行列顺序。
    column_dtypes = {sample_id: np.float32 for sample_id in expected_sample_ids}
    column_dtypes["sample"] = str
    expression = pd.read_csv(
        expression_path,
        sep="\t",
        index_col=0,
        dtype=column_dtypes,
    )
    if expression.shape[1] != len(expected_sample_ids):
        raise Step1Error(
            f"表达矩阵样本数 {expression.shape[1]} 与表头 {len(expected_sample_ids)} 不一致。"
        )
    if expression.shape[0] != expected_gene_count:
        raise Step1Error(
            f"表达矩阵基因数 {expression.shape[0]} 与基因名称文件 {expected_gene_count} 不一致。"
        )
    actual_sample_ids = expression.columns.astype(str).tolist()
    if actual_sample_ids != list(expected_sample_ids):
        raise Step1Error("表达矩阵读取后的样本顺序与已冻结的表头顺序不一致。")
    if expression.index.astype(str).duplicated().any():
        raise Step1Error("表达矩阵存在重复基因 ID。")
    values = expression.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise Step1Error("表达矩阵含 NaN 或无穷值，未自动删除或替换。")
    return expression, expression.index.astype(str).tolist()


def preprocess_expression(
    expression: pd.DataFrame, output_path: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """按原研究逻辑把 log2(TPM+0.001) 转换为 log2(TPM+1)。

    为控制内存，转换在 genes×samples 的 float32 数组上原位完成，之后转置
    为 samples×genes 并保存为 float16 NPY。训练时再把每个 batch 转回 float32，
    与原代码中 ``torch.tensor(data, dtype=torch.float32)`` 的用法一致。
    """

    raw_values = expression.to_numpy(copy=False)
    if raw_values.dtype != np.float32:
        raw_values = raw_values.astype(np.float32, copy=False)
    if not raw_values.flags.writeable:
        # 部分 pandas/NumPy 版本会返回只读视图；仅在这种情况下复制，
        # 正常的 CSV 读取路径仍使用原位变换以控制峰值内存。
        raw_values = raw_values.copy()
    input_min = float(np.min(raw_values))
    input_max = float(np.max(raw_values))

    # 输入文件是 log2(TPM+0.001)。先逆变换，再应用正式训练尺度。
    np.exp2(raw_values, out=raw_values)
    raw_values -= np.float32(STORED_OFFSET)
    # 原始文件由有限小数位保存；零 TPM 在逆变换后可能出现极小负数。
    # 该截断只消除浮点舍入误差，并与已完成的尺度比较脚本保持一致。
    np.maximum(raw_values, 0.0, out=raw_values)
    raw_values += np.float32(FINAL_OFFSET)
    np.log2(raw_values, out=raw_values)

    if not np.isfinite(raw_values).all():
        raise Step1Error("log2(TPM+1) 变换后产生 NaN 或无穷值。")
    transformed = raw_values.T.astype(np.float16, copy=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, transformed, allow_pickle=False)
    logger.info("Saved preprocessed matrix: %s; shape=%s; dtype=%s", output_path, transformed.shape, transformed.dtype)
    return {
        "stored_input_scale": "log2(TPM + 0.001)",
        "inverse_transform": "TPM_approx = maximum(exp2(stored_value) - 0.001, 0)",
        "nonnegative_clip_after_inverse": True,
        "final_transform": "log2(TPM_approx + 1)",
        "input_min_before_transform": input_min,
        "input_max_before_transform": input_max,
        "output_min_after_float16": float(np.min(transformed)),
        "output_max_after_float16": float(np.max(transformed)),
        "output_dtype": str(transformed.dtype),
        "output_shape": [int(x) for x in transformed.shape],
    }


def make_sample_split(n_samples: int, seed: int) -> SplitIndices:
    """复现 sklearn train_test_split 的 80/20 样本级随机划分语义。

    不依赖 scikit-learn：先使用 RandomState(seed).permutation，再取前 n_test
    个位置为测试集，剩余位置为训练集。这与原研究的随机划分目的相同，且
    当前运行环境不需要额外安装 scikit-learn。
    """

    if not 0 < TEST_FRACTION < 1:
        raise Step1Error("TEST_FRACTION 必须在 0 和 1 之间。")
    n_test = int(np.ceil(n_samples * TEST_FRACTION))
    permutation = np.random.RandomState(seed).permutation(n_samples)
    return SplitIndices(
        train_indices=permutation[n_test:],
        test_indices=permutation[:n_test],
    )


def save_csv_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写入字典行，并保证空表也有稳定的列结构。"""

    rows = list(rows)
    if not rows:
        raise Step1Error(f"不能写入空表：{path}")
    pd.DataFrame(rows).to_csv(path, index=False)


def save_preparation_metadata(
    paths: Step1Paths,
    sample_manifest: pd.DataFrame,
    gene_ids: Sequence[str],
    gene_names: pd.Series,
    split: SplitIndices,
    preprocess_metadata: Dict[str, Any],
    sample_ids: Sequence[str],
    logger: logging.Logger,
) -> None:
    """保存数据冻结、划分、基因顺序和预处理参数。"""

    paths.make_output_directories()
    metadata_dir = paths.absolute(paths.metadata_dir)

    manifest = sample_manifest.copy()
    manifest.insert(0, "expression_column_position", np.arange(len(manifest)))
    manifest.to_csv(metadata_dir / "sample_manifest.csv", index=False)

    full_rows = [
        {"order": int(i), "sample_id": sample_id}
        for i, sample_id in enumerate(sample_ids)
    ]
    train_rows = [
        {"order_in_split": int(i), "full_data_position": int(index), "sample_id": sample_ids[index]}
        for i, index in enumerate(split.train_indices)
    ]
    test_rows = [
        {"order_in_split": int(i), "full_data_position": int(index), "sample_id": sample_ids[index]}
        for i, index in enumerate(split.test_indices)
    ]
    save_csv_rows(metadata_dir / "sample_order_full.csv", full_rows)
    save_csv_rows(metadata_dir / "sample_order_train.csv", train_rows)
    save_csv_rows(metadata_dir / "sample_order_test.csv", test_rows)

    gene_rows = [
        {
            "expression_row_position": int(i),
            "gene_id": str(gene_id),
            "gene_name": str(gene_names.iloc[i]),
        }
        for i, gene_id in enumerate(gene_ids)
    ]
    save_csv_rows(metadata_dir / "gene_order.csv", gene_rows)

    split_metadata = {
        "master_seed": MASTER_SEED,
        "split_seed": MASTER_SEED,
        "model_seed_rule": "model_seed = master_seed + model_number",
        "model_seeds": {str(i): MASTER_SEED + i for i in range(1, N_MODELS + 1)},
        "test_fraction": TEST_FRACTION,
        "n_samples": len(sample_ids),
        "n_train": len(split.train_indices),
        "n_test": len(split.test_indices),
        "train_indices_are_in_saved_order": True,
        "test_indices_are_in_saved_order": True,
        "full_data_order": "original expression matrix sample order",
        "train_sample_ids": [sample_ids[int(i)] for i in split.train_indices],
        "test_sample_ids": [sample_ids[int(i)] for i in split.test_indices],
    }
    (metadata_dir / "sample_split.json").write_text(
        json.dumps(split_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    architecture = {
        "input_dim": 60498,
        "encoder_dims": [60498, 2048, 512, 64],
        "decoder_dims": [64, 512, 2048, 60498],
        "encoder_layers": [
            "Linear(60498, 2048)",
            "BatchNorm1d(2048)",
            "LeakyReLU(negative_slope=0.2)",
            "Dropout(p=0.3)",
            "Linear(2048, 512)",
            "BatchNorm1d(512)",
            "LeakyReLU(negative_slope=0.2)",
            "Dropout(p=0.3)",
            "Linear(512, 64)",
        ],
        "decoder_layers": [
            "Linear(64, 512)",
            "BatchNorm1d(512)",
            "LeakyReLU(negative_slope=0.2)",
            "Dropout(p=0.3)",
            "Linear(512, 2048)",
            "BatchNorm1d(2048)",
            "LeakyReLU(negative_slope=0.2)",
            "Dropout(p=0.3)",
            "Linear(2048, 60498)",
        ],
        "activation": "LeakyReLU",
        "leaky_relu_negative_slope": 0.2,
        "dropout_rate": 0.3,
        "batch_norm": True,
        "batch_norm_position": "after_linear_before_activation",
        "initialization": "kaiming_uniform_(a=0.2, nonlinearity='leaky_relu')",
        "loss": "MSELoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": N_EPOCHS,
        "final_model_epoch": N_EPOCHS,
    }
    (metadata_dir / "model_architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_metadata = {
        "master_seed": MASTER_SEED,
        "n_models": N_MODELS,
        "stored_input_scale": "log2(TPM + 0.001)",
        "final_training_scale": "log2(TPM + 1)",
        "float16_intermediate_conversion": FLOAT16_INTERMEDIATE,
        "duplicate_gene_name_rule": "mean aggregation deferred to GSEA input stage",
        "randomness": {
            "split_seed": MASTER_SEED,
            "model_seed_rule": "model_seed = master_seed + model_number",
            "python_numpy_torch_seeded_per_model": True,
            "data_loader_generator_seeded_per_model": True,
            "data_loader_num_workers": 0,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "tissue_label_source": "dataset/GTEX_phenotype matched by expression sample ID",
        "regenerated_label_files": [
            "input_data/tsne_label.csv",
            "input_data/tissue_appear_times.csv",
        ],
        "preprocessed_matrix_relative_path": str(paths.preprocessed_dir / "log2_tpm_plus_1.npy"),
        "preprocess": preprocess_metadata,
    }
    (metadata_dir / "experiment_config.json").write_text(
        json.dumps({"architecture": architecture, "run": run_metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved sample, gene, split, architecture and run metadata.")


def prepare_step1(paths: Step1Paths, logger: logging.Logger) -> None:
    """执行 Step1 的数据冻结、标签重建、预处理和样本划分。"""

    paths.make_output_directories()
    expression_path = paths.absolute(paths.expression_file)
    if not expression_path.is_file():
        raise FileNotFoundError(expression_path)

    sample_ids = read_expression_header(expression_path)
    logger.info("Detected %d expression samples.", len(sample_ids))
    sample_manifest, _, gene_names = load_and_validate_metadata(paths, sample_ids)
    if len(gene_names) != 60498:
        raise Step1Error(f"gene_name_60498.csv 应包含 60498 行，实际为 {len(gene_names)}。")

    rebuild_tissue_label_files(paths, sample_manifest)
    logger.info("Regenerated tissue labels for %d tissue types.", sample_manifest['tissue_name'].nunique())

    expression, gene_ids = load_expression_matrix(paths, sample_ids, len(gene_names))
    split = make_sample_split(len(sample_ids), MASTER_SEED)
    preprocessed_path = paths.absolute(paths.preprocessed_dir) / "log2_tpm_plus_1.npy"
    preprocess_metadata = preprocess_expression(expression, preprocessed_path, logger)
    save_preparation_metadata(
        paths,
        sample_manifest,
        gene_ids,
        gene_names,
        split,
        preprocess_metadata,
        sample_ids,
        logger,
    )
    logger.info(
        "Preparation complete: n_train=%d, n_test=%d, full_order=original expression order.",
        len(split.train_indices),
        len(split.test_indices),
    )


class ExpressionDataset(Dataset):
    """从 float16 NPY 矩阵按样本位置提供 float32 batch 输入。"""

    def __init__(self, matrix: np.ndarray, indices: Sequence[int]) -> None:
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> torch.Tensor:
        row = np.asarray(self.matrix[int(self.indices[item])], dtype=np.float32)
        return torch.from_numpy(row)


class Autoencoder(nn.Module):
    """原研究浅层对称自编码器，初始化显式匹配 LeakyReLU(0.2)。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60498, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 60498),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """对每个 Linear 使用与负斜率 0.2 匹配的 Kaiming uniform 初始化。"""

        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


def load_saved_split(paths: Step1Paths) -> SplitIndices:
    """从 prepare 阶段保存的 CSV 恢复固定划分。"""

    metadata_dir = paths.absolute(paths.metadata_dir)
    train = pd.read_csv(metadata_dir / "sample_order_train.csv")
    test = pd.read_csv(metadata_dir / "sample_order_test.csv")
    return SplitIndices(
        train_indices=train["full_data_position"].to_numpy(dtype=np.int64),
        test_indices=test["full_data_position"].to_numpy(dtype=np.int64),
    )


def train_one_model(
    paths: Step1Paths,
    split: SplitIndices,
    model_number: int,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """训练并保存一个模型，最终参数固定为第 100 个 epoch 的参数。"""

    model_seed = MASTER_SEED + model_number
    set_global_seed(model_seed)
    matrix_path = paths.absolute(paths.preprocessed_dir) / "log2_tpm_plus_1.npy"
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"未找到预处理矩阵 {matrix_path}，请先运行 prepare。"
        )
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape[1] != 60498 or matrix.dtype != np.float16:
        raise Step1Error(
            f"预处理矩阵应为 (n_samples, 60498) float16，实际为 {matrix.shape} {matrix.dtype}。"
        )

    train_dataset = ExpressionDataset(matrix, split.train_indices)
    test_dataset = ExpressionDataset(matrix, split.test_indices)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(model_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    loss_rows: List[Dict[str, Any]] = []
    loss_path = paths.absolute(paths.losses_dir) / f"model_{model_number:02d}_loss.csv"

    logger.info("Model %02d/%02d started with seed=%d on %s.", model_number, N_MODELS, model_seed, device)
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_total = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device, non_blocking=False)
            _, reconstruction = model(inputs)
            loss = criterion(reconstruction, inputs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu()) * inputs.shape[0]

        model.eval()
        test_total = 0.0
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device, non_blocking=False)
                _, reconstruction = model(inputs)
                test_total += float(criterion(reconstruction, inputs).cpu()) * inputs.shape[0]

        train_loss = train_total / len(train_loader.dataset)
        test_loss = test_total / len(test_loader.dataset)
        loss_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "model_number": model_number,
                "model_seed": model_seed,
            }
        )
        # 每个 epoch 都更新一次 CSV。先写临时文件再替换，避免中断时留下半行，
        # 也让长时间训练过程中的损失曲线可以随时检查。
        temporary_loss_path = loss_path.with_suffix(".tmp")
        pd.DataFrame(loss_rows).to_csv(temporary_loss_path, index=False)
        temporary_loss_path.replace(loss_path)
        logger.info(
            "model=%02d epoch=%03d/%03d train_loss=%.8f test_loss=%.8f",
            model_number,
            epoch,
            N_EPOCHS,
            train_loss,
            test_loss,
        )

    checkpoint_path = paths.absolute(paths.models_dir) / f"model_{model_number:02d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_number": model_number,
            "model_seed": model_seed,
            "final_epoch": N_EPOCHS,
            "architecture_config": json.loads(
                (paths.absolute(paths.metadata_dir) / "model_architecture.json").read_text(encoding="utf-8")
            ),
        },
        checkpoint_path,
    )
    logger.info("Model %02d saved: %s", model_number, checkpoint_path)


def train_step1(
    paths: Step1Paths,
    model_start: int = 1,
    model_end: int = N_MODELS,
    device_name: str = "auto",
) -> None:
    """按固定划分顺序训练指定范围的独立模型。"""

    if not 1 <= model_start <= model_end <= N_MODELS:
        raise ValueError(f"模型范围必须在 1 到 {N_MODELS} 内且 start<=end。")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    paths.make_output_directories()
    logger = configure_logging(paths.absolute(paths.logs_dir) / "train_step1.log")
    split = load_saved_split(paths)
    for model_number in range(model_start, model_end + 1):
        train_one_model(paths, split, model_number, device, logger)
    logger.info("Training complete for models %d through %d.", model_start, model_end)


def parse_args() -> argparse.Namespace:
    """解析命令行阶段和参数。"""

    parser = argparse.ArgumentParser(description="Run the formal reproduction Step 1.")
    parser.add_argument("stage", choices=["prepare", "train", "all"])
    parser.add_argument("--model-start", type=int, default=1)
    parser.add_argument("--model-end", type=int, default=N_MODELS)
    parser.add_argument("--device", default="auto", help="auto, cpu 或 cuda")
    return parser.parse_args()


def main() -> None:
    """命令行入口；默认工作目录是 trial，所有输入输出配置均为相对路径。"""

    args = parse_args()
    paths = Step1Paths(Path("."))
    paths.make_output_directories()
    logger = configure_logging(paths.absolute(paths.logs_dir) / "prepare_step1.log")
    try:
        if args.stage in {"prepare", "all"}:
            prepare_step1(paths, logger)
        if args.stage in {"train", "all"}:
            train_step1(paths, args.model_start, args.model_end, args.device)
    except Exception:
        # 同时把完整 traceback 写入当前阶段日志，便于追溯子进程失败原因。
        # 继续抛出异常，确保调用方收到非零退出码，不能把失败当作完成。
        logger.exception("Step1 stage '%s' failed.", args.stage)
        raise


if __name__ == "__main__":
    main()

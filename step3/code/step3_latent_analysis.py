"""Step3：64 维潜在空间可视化与组织间神经元差异分析。

从 trial 目录调用：
    python step3/code/step3_latent_analysis.py check
    python step3/code/step3_latent_analysis.py self-test
    python step3/code/step3_latent_analysis.py all

也可分别调用 kw、tsne、plots 阶段。所有模型各自处理，全部 7862 个样本
参加分析。每个模型仅运行一次 t-SNE，固定主种子为 222。
每个模型内对 64 次 KW 检验做 BH 校正；效应量严格采用 H/(n-1)，候选
条件为 q<0.05 且 epsilon2>0.14，仅用于探索性 Jacobian/GSEA。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")  # 子进程只保存文件，不打开阻塞窗口。
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


TRIAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("step3")
LAYER_NAME = "encoder_linear_512_to_64"
MAIN_SEED = 222
N_SAMPLES, N_UNITS, N_TISSUES = 7862, 64, 54
UNIT_COLUMNS = [f"unit_{i:05d}" for i in range(N_UNITS)]

# sklearn 1.9.0 已将迭代次数参数命名为 max_iter；不能传入 n_iter。
# max_iter 是上限；无进展或梯度阈值可能提前终止，实际 n_iter_ 另行记录。
TSNE_PARAMETERS = {
    "n_components": 2, "perplexity": 30.0, "max_iter": 1000,
    "init": "pca", "learning_rate": "auto", "metric": "euclidean",
    "early_exaggeration": 12.0, "method": "barnes_hut", "angle": 0.5,
    "n_iter_without_progress": 300, "min_grad_norm": 1e-7, "n_jobs": 1,
}
KW_PARAMETERS = {
    "test": "Kruskal-Wallis", "alternative": "omnibus distribution difference",
    "p_value_method": "chi_square_approximation", "tie_correction": True,
    "bh_scope": "64 neurons within each model", "bh_method": "bh",
    "effect_size_formula": "epsilon2 = H / (n - 1)",
    "q_threshold_strict": 0.05, "epsilon2_threshold_strict": 0.14,
    "candidate_use": "exploratory Jacobian/GSEA only",
}
LOGGER = logging.getLogger("step3")


def require(condition: bool, message: str) -> None:
    """输入/结果错误明确失败，不自动删样本、替换数值或更改参数。"""
    if not condition:
        raise ValueError(message)


def write_json(path: Path, content: dict) -> None:
    """原子更新运行状态，避免中断时留下无法解析的完成记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """文件指纹用于确认续跑时输入与输出均未改变。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}


@dataclass
class ModelData:
    """矩阵行序与 sample_map 一致；组织中位数按组织名称对齐。"""
    number: int
    matrix: np.ndarray
    sample_map: pd.DataFrame
    medians: pd.DataFrame
    source_hashes: dict
    checkpoint_sha256: str

    @property
    def tag(self) -> str:
        return f"model_{self.number:02d}"

    @property
    def tissues(self) -> list[str]:
        return sorted(self.sample_map["tissue_name"].unique())


def load_model_data(number: int) -> ModelData:
    """仅读取 Step2 输出和 Step1 样本映射；检查完整样本、64 维和完成状态。"""
    tag = f"model_{number:02d}"
    paths = {
        "matrix": Path(f"step2/raw_outputs/{tag}/{LAYER_NAME}.npy"),
        "sample_map": Path(f"step2/raw_outputs/{tag}/sample_map.csv"),
        "medians": Path(f"step2/tissue_medians/{tag}/{LAYER_NAME}.csv"),
        "completion": Path(f"step2/quality_control/{tag}.json"),
        "model_metadata": Path(f"step2/metadata/{tag}.json"),
        "full_order": Path("step1/core_metadata/sample_order_full.csv"),
        "sample_manifest": Path("step1/core_metadata/sample_manifest.csv"),
        "train_order": Path("step1/core_metadata/sample_order_train.csv"),
        "test_order": Path("step1/core_metadata/sample_order_test.csv"),
    }
    qc = json.loads(paths["completion"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["model_metadata"].read_text(encoding="utf-8"))
    require(qc["status"] == "complete" and qc["sample_coverage_verified"], f"{tag}: Step2 未完成。")
    for record in (qc, metadata):
        require(record["model_number"] == number and record["model_seed"] == 629 + number
                and record["final_epoch"] == 100, f"{tag}: 模型编号、种子或 epoch 错误。")
    matrix = np.load(paths["matrix"], allow_pickle=False)
    mapping = pd.read_csv(paths["sample_map"], keep_default_na=False)
    medians = pd.read_csv(paths["medians"], keep_default_na=False, float_precision="round_trip")
    full = pd.read_csv(paths["full_order"], keep_default_na=False)
    manifest = pd.read_csv(paths["sample_manifest"], keep_default_na=False).set_index("Sample")
    require(matrix.shape == (N_SAMPLES, N_UNITS) and matrix.dtype == np.float32,
            f"{tag}: 瓶颈矩阵必须为 (7862, 64) float32。")
    require(np.isfinite(matrix).all(), f"{tag}: 激活值含 NaN/Inf。")
    require(len(mapping) == N_SAMPLES and mapping["sample_id"].is_unique, f"{tag}: 样本数或 ID 错误。")
    require(mapping["sample_id"].tolist() == full["sample_id"].tolist(), f"{tag}: 完整样本顺序不一致。")
    require(np.array_equal(mapping["row_index"], np.arange(N_SAMPLES))
            and np.array_equal(mapping["full_data_position"], np.arange(N_SAMPLES)), f"{tag}: 样本行号不连续。")
    require(mapping["model_number"].eq(number).all() and mapping["model_seed"].eq(629 + number).all(),
            f"{tag}: sample_map 模型标记错误。")
    require(mapping["tissue_name"].tolist() == manifest.loc[mapping["sample_id"], "tissue_name"].tolist(),
            f"{tag}: 组织标签与样本 ID 不匹配。")
    require(mapping["split"].isin(["train", "test"]).all(), f"{tag}: split 含未知值。")
    for split in ("train", "test"):
        expected = pd.read_csv(paths[f"{split}_order"], keep_default_na=False)
        actual = mapping.loc[mapping["split"].eq(split)].sort_values("order_in_split")
        require(actual["sample_id"].tolist() == expected["sample_id"].tolist(), f"{tag}: {split} 映射错误。")
    tissues = sorted(mapping["tissue_name"].unique())
    require(len(tissues) == N_TISSUES and "" not in tissues, f"{tag}: 组织数量错误或标签为空。")
    require(medians["tissue_name"].is_unique and sorted(medians["tissue_name"]) == tissues
            and list(medians.columns[2:]) == UNIT_COLUMNS, f"{tag}: 中位数表结构错误。")
    medians = medians.set_index("tissue_name").loc[tissues].reset_index()
    for index, tissue in enumerate(tissues):
        group = matrix[mapping["tissue_name"].eq(tissue).to_numpy()]
        require(int(medians.iloc[index]["n_samples"]) == len(group), f"{tag}: 组织样本数不匹配。")
        require(np.array_equal(np.median(group, axis=0), medians.iloc[index, 2:].to_numpy(dtype=np.float32)),
                f"{tag}: 组织中位数与逐样本值不一致。")
    hashes = {path.as_posix(): file_hash(path) for path in paths.values()}
    return ModelData(number, matrix, mapping, medians, hashes, metadata["checkpoint_sha256"])


def configuration(data: ModelData, stage: str, parameters: dict) -> dict:
    """每个运行记录固定输入、模型来源、代码指纹和软件版本。"""
    return {"stage": stage, "model_number": data.number, "model_seed": 629 + data.number,
            "final_epoch": 100, "layer": LAYER_NAME,
            "sample_scope": "full" if len(data.matrix) == N_SAMPLES else "validation_subset",
            "n_samples": len(data.matrix),
            "sample_id_order_sha256": hashlib.sha256("\n".join(data.sample_map["sample_id"]).encode("utf-8")).hexdigest(),
            "source_sha256": data.source_hashes, "checkpoint_sha256": data.checkpoint_sha256,
            "script_sha256": file_hash(Path(__file__)), "versions": versions(),
            "parameters": parameters}


def configurations_match(saved: dict | None, current: dict) -> bool:
    """只兼容已审计的单种子精简；历史记录保留生成时的真实代码指纹。

    除列明的代码指纹外，参数、版本、输入指纹及样本顺序必须完全相同。
    后续任何未登记代码改动仍会失败；不会把旧结果改标为新代码生成。
    Step4通过本函数比较代码依赖，数值计算及输入检查保持不变。
    """
    if saved == current:
        return True
    if not isinstance(saved, dict):
        return False
    path = TRIAL_ROOT / "step3/metadata/single_seed_revision/compatibility.json"
    if not path.is_file():
        return False
    revision = json.loads(path.read_text(encoding="utf-8"))
    if revision.get("status") != "verified":
        return False
    adjusted = dict(saved)

    def approved(file: str, previous: str, present: str) -> bool:
        entry = revision["code_changes"].get(file, {})
        archive = TRIAL_ROOT / entry.get("archive", "missing")
        return (previous == entry.get("before_sha256") and present == entry.get("after_sha256")
                and archive.is_file() and file_hash(archive) == previous
                and file_hash(TRIAL_ROOT / file) == present)

    if saved.get("stage") in ("kw", "tsne"):
        if saved["stage"] == "tsne" and saved.get("parameters", {}).get("random_state") != MAIN_SEED:
            return False
        if approved("step3/code/step3_latent_analysis.py", saved.get("script_sha256"), current.get("script_sha256")):
            adjusted["script_sha256"] = current["script_sha256"]
    elif "code_sha256" in saved and "code_sha256" in current:
        codes = dict(saved["code_sha256"])
        for file, present in current["code_sha256"].items():
            if codes.get(file) != present and approved(file, codes.get(file), present):
                codes[file] = present
        adjusted["code_sha256"] = codes
    return adjusted == current


def begin_stage(report_path: Path, config: dict, outputs: list[Path], overwrite: bool) -> bool:
    """默认跳过已验证且配置相同的完成项，支持按模型及种子续跑。"""
    if report_path.exists() and not overwrite:
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete" and configurations_match(saved.get("configuration"), config):
            matches = all(path.is_file() and saved.get("output_sha256", {}).get(path.as_posix()) == file_hash(path)
                          for path in outputs)
            if matches:
                LOGGER.info("Skip completed: %s", report_path)
                return False
        raise FileExistsError(f"{report_path} 未完成、配置不符或文件已改变；请使用 --overwrite 重算该范围。")
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("输出已存在但没有匹配的完成记录；请使用 --overwrite 重算该范围。")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, {"status": "running", "configuration": config})
    return True


def finish_stage(report_path: Path, config: dict, outputs: list[Path], details: dict) -> None:
    """只有全部输出写入并计算指纹后，才能标记 complete。"""
    write_json(report_path, {"status": "complete", "configuration": config,
                           "output_sha256": {path.as_posix(): file_hash(path) for path in outputs}, **details})


def candidate_mask(q_values: np.ndarray, epsilon2: np.ndarray) -> np.ndarray:
    """严格不等号：q 恰好为 0.05 或 epsilon2 恰好为 0.14 均不入选。"""
    return (q_values < 0.05) & (epsilon2 > 0.14)


def calculate_kw(matrix: np.ndarray, labels: np.ndarray, model_number: int) -> pd.DataFrame:
    """每个模型包含完整 64 次检验，BH 不跨模型，也不对组织两两比较。

    scipy.stats.kruskal 含并列秩校正。若某个单元对全部样本为常数，KW 的
    通常公式分母为零；这里显式记 H=0、p=1，并将该单元保留在64次BH中。
    极小 p 值可能下溢为 0，保存库返回值，不把 0 改成任意常数。
    """
    require(matrix.ndim == 2 and matrix.shape[1] == N_UNITS and len(matrix) == len(labels), "KW 输入维度错误。")
    require(np.isfinite(matrix).all(), "KW 输入含非有限值。")
    tissues = sorted(set(labels))
    require(len(tissues) > 1 and len(matrix) > len(tissues), "KW 样本或分组不足。")
    positions = [np.flatnonzero(labels == tissue) for tissue in tissues]
    rows = []
    for unit in range(N_UNITS):
        values = matrix[:, unit]
        if np.ptp(values) == 0:
            h, p_value, status = 0.0, 1.0, "constant_all_samples"
        else:
            result = stats.kruskal(*(values[index] for index in positions), nan_policy="raise")
            h, p_value, status = float(result.statistic), float(result.pvalue), "ok"
        require(np.isfinite(h) and np.isfinite(p_value) and h >= 0 and 0 <= p_value <= 1, "KW 返回无效结果。")
        rows.append({"model_number": model_number, "model_seed": 629 + model_number,
                     "neuron_index": unit, "unit_id": UNIT_COLUMNS[unit],
                     "n_samples": len(values), "n_tissues": len(tissues), "df": len(tissues) - 1,
                     "H": h, "p_value": p_value, "epsilon2": h / (len(values) - 1), "test_status": status})
    table = pd.DataFrame(rows)
    # 固定在这个模型的64个p值上一次性校正，恒定单元也计入检验族。
    table["q_value_bh"] = stats.false_discovery_control(table["p_value"].to_numpy(), method="bh")
    table["p_lt_0_05"] = table["p_value"] < 0.05
    table["q_lt_0_05"] = table["q_value_bh"] < 0.05
    table["epsilon2_gt_0_14"] = table["epsilon2"] > 0.14
    table["candidate"] = candidate_mask(table["q_value_bh"].to_numpy(), table["epsilon2"].to_numpy())
    table["candidate_use"] = "exploratory_jacobian_gsea_only"
    return table


def tissue_ranks(data: ModelData) -> pd.DataFrame:
    """使用 Step2 中位数排序；并列值按组织名稳定排序，不解释为显著性。"""
    parts = []
    for unit in UNIT_COLUMNS:
        frame = data.medians[["tissue_name", "n_samples", unit]].rename(columns={unit: "median_activation"})
        frame = frame.sort_values(["median_activation", "tissue_name"], ascending=[False, True]).reset_index(drop=True)
        frame["rank_desc"] = np.arange(1, len(frame) + 1)
        frame["unit_id"] = unit
        frame["model_number"] = data.number
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def run_kw(data: ModelData, overwrite: bool, root: Path = ROOT) -> None:
    folder = root / "statistics" / data.tag
    outputs = [folder / "kw_results.csv", folder / "candidates.csv", folder / "tissue_median_ranks.csv"]
    report = root / "quality_control" / f"{data.tag}_kw.json"
    config = configuration(data, "kw", KW_PARAMETERS)
    if not begin_stage(report, config, outputs, overwrite):
        return
    table = calculate_kw(data.matrix, data.sample_map["tissue_name"].to_numpy(), data.number)
    table.to_csv(outputs[0], index=False, float_format="%.17g")
    table.loc[table["candidate"]].to_csv(outputs[1], index=False, float_format="%.17g")
    tissue_ranks(data).to_csv(outputs[2], index=False, float_format="%.9g")
    finish_stage(report, config, outputs, {
        "n_tests": 64, "n_candidates": int(table["candidate"].sum()),
        "n_zero_pvalues": int(table["p_value"].eq(0).sum()),
        "smallest_tissue_n": int(data.medians["n_samples"].min()),
        "interpretation": "exploratory omnibus tests; repeated donors and small groups remain limitations"})
    LOGGER.info("%s KW: %d/64 candidates", data.tag, table["candidate"].sum())


def fit_tsne(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """固定线程数与全部算法参数；PCA 仅用于二维初始化，不预降维输入。"""
    require(sklearn.__version__ == "1.9.0", "本步骤要求 scikit-learn==1.9.0。")
    start = time.perf_counter()
    estimator = TSNE(**TSNE_PARAMETERS, random_state=seed, verbose=1)
    # 控制 OpenMP/BLAS，避免笔记本电脑线程过度竞争；n_jobs 仅控制部分路径。
    with threadpool_limits(limits=1):
        coordinates = estimator.fit_transform(matrix)
    require(coordinates.shape == (len(matrix), 2) and np.isfinite(coordinates).all(), "t-SNE 坐标无效。")
    require(np.isfinite(estimator.kl_divergence_), "t-SNE KL divergence 非有限值。")
    return coordinates, {"seed": seed, "learning_rate_actual": float(estimator.learning_rate_),
                         "sklearn_n_iter_zero_based": int(estimator.n_iter_),
                         "kl_divergence": float(estimator.kl_divergence_),
                         "elapsed_seconds": time.perf_counter() - start,
                         "thread_limit": 1, "input_transformation": "none"}


def run_tsne(data: ModelData, overwrite: bool, root: Path = ROOT) -> None:
    """仅调用主种子222；已完成且来源相符时只校验文件，不重复计算。"""
    path = root / "coordinates" / data.tag / f"tsne_seed_{MAIN_SEED}.csv"
    report = root / "quality_control" / f"{data.tag}_tsne_seed_{MAIN_SEED}.json"
    config = configuration(data, "tsne", {**TSNE_PARAMETERS, "random_state": MAIN_SEED, "thread_limit": 1})
    if not begin_stage(report, config, [path], overwrite):
        return
    LOGGER.info("%s t-SNE seed=%d (main)", data.tag, MAIN_SEED)
    coordinates, diagnostics = fit_tsne(data.matrix, MAIN_SEED)
    table = data.sample_map.copy()
    table["tsne_seed"], table["run_role"] = MAIN_SEED, "main"
    table["tsne_1"], table["tsne_2"] = coordinates[:, 0], coordinates[:, 1]
    table.to_csv(path, index=False, float_format="%.9g")
    restored = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
    require(restored["sample_id"].tolist() == data.sample_map["sample_id"].tolist(), "坐标样本映射改变。")
    require(np.array_equal(restored[["tsne_1", "tsne_2"]].to_numpy(dtype=coordinates.dtype), coordinates),
            "坐标 CSV 往返数值不一致。")
    finish_stage(report, config, [path], diagnostics)


def palette_for(tissues: list[str]) -> dict[str, str]:
    """54 个组织使用固定的离散颜色；排序保证各模型和图页保持一致。"""
    # 组合三套类别色板，再交错深浅色，提高相邻图例项的区分度。
    colors = np.concatenate([plt.get_cmap(name)(np.arange(20)) for name in ("tab20", "tab20b", "tab20c")])
    return {name: to_hex(colors[index]) for index, name in enumerate(tissues)}


def draw_scatter(ax, coordinates: np.ndarray, labels: np.ndarray, palette: dict, title: str) -> None:
    """组织标签只决定颜色；坐标已经由不含标签的64维矩阵计算完成。"""
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=[palette[name] for name in labels],
               s=5, alpha=0.8, linewidths=0, rasterized=True)
    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def legend_handles(data: ModelData, palette: dict) -> list[Line2D]:
    counts = data.sample_map["tissue_name"].value_counts()
    return [Line2D([], [], marker="o", linestyle="", color=palette[tissue], markersize=4,
                   label=f"{tissue} (n={counts[tissue]})") for tissue in data.tissues]


def save_tsne_figures(data: ModelData, coordinates: np.ndarray, paths: list[Path]) -> None:
    """保存主种子222的PNG/PDF；主图布局和组织配色保持不变。"""
    palette = palette_for(data.tissues)
    labels = data.sample_map["tissue_name"].to_numpy()
    handles = legend_handles(data, palette)
    sample_scope = "Full sample set" if len(labels) == N_SAMPLES else "Validation subset"
    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_axes((0.065, 0.12, 0.58, 0.79))
    draw_scatter(ax, coordinates, labels, palette, f"{data.tag} | 64-dimensional latent space | seed 222")
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.67, 0.52),
               frameon=False, fontsize=7.7, labelspacing=0.45)
    fig.text(0.065, 0.035, f"{sample_scope}: n={len(labels)} | scikit-learn 1.9.0 | perplexity=30 | max_iter=1000", fontsize=9)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)


def save_boxplots(data: ModelData, statistics: pd.DataFrame, path: Path, units=range(N_UNITS)) -> None:
    """一页一个神经元，展示全部54个组织；箱体为IQR，须线为1.5 IQR。

    页内按组织中位数降序排列，并列时按组织名排列；所有离群点保留。
    q和效应量是该神经元的总体检验结果，不画组织之间显著性星号。
    """
    palette = palette_for(data.tissues)
    labels = data.sample_map["tissue_name"].to_numpy()
    with PdfPages(path, metadata={"Title": f"{data.tag} latent neuron tissue distributions"}) as pdf:
        for unit in units:
            unit_name = UNIT_COLUMNS[unit]
            ordered = data.medians.sort_values([unit_name, "tissue_name"], ascending=[False, True])
            names = ordered["tissue_name"].tolist()
            groups = [data.matrix[labels == tissue, unit] for tissue in names]
            result = statistics.loc[statistics["neuron_index"].eq(unit)].iloc[0]
            fig, ax = plt.subplots(figsize=(12, 14))
            fig.subplots_adjust(left=0.40, right=0.95, bottom=0.08, top=0.91)
            boxes = ax.boxplot(groups, orientation="horizontal", patch_artist=True, whis=1.5,
                               showfliers=True, widths=0.55,
                               flierprops={"markersize": 1.7, "markeredgewidth": 0.2, "alpha": 0.5},
                               medianprops={"color": "black", "linewidth": 1.0})
            for patch, tissue in zip(boxes["boxes"], names):
                patch.set_facecolor(palette[tissue])
                patch.set_alpha(0.65)
            ax.set_yticks(np.arange(1, len(names) + 1), [f"{name} (n={len(group)})" for name, group in zip(names, groups)], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Latent neuron activation (linear output)", fontsize=11)
            ax.grid(axis="x", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"{data.tag} | {unit_name} | neuron index {unit} (zero-based)", y=0.968, fontsize=15)
            fig.text(0.04, 0.931, f"KW H={result['H']:.4g} | p={result['p_value']:.4g} | BH q={result['q_value_bh']:.4g} | "
                     f"epsilon2={result['epsilon2']:.4f} | candidate={bool(result['candidate'])}", fontsize=11)
            fig.text(0.04, 0.025, "epsilon2 = H / (n - 1). Candidates: q < 0.05 AND epsilon2 > 0.14. Exploratory use only.", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def completed_output(report_path: Path, config: dict, output: Path) -> None:
    """绘图只读取当前输入对应且完整的统计/坐标文件。"""
    require(report_path.is_file(), f"请先计算所需结果：{report_path}")
    record = json.loads(report_path.read_text(encoding="utf-8"))
    require(record.get("status") == "complete" and configurations_match(record.get("configuration"), config),
            f"{report_path} 未完成或配置与当前输入不一致。")
    require(output.is_file() and record["output_sha256"].get(output.as_posix()) == file_hash(output),
            f"{output} 不存在或内容已改变。")


def run_plots(data: ModelData, overwrite: bool, root: Path = ROOT) -> None:
    path = root / "coordinates" / data.tag / f"tsne_seed_{MAIN_SEED}.csv"
    config = configuration(data, "tsne", {**TSNE_PARAMETERS, "random_state": MAIN_SEED, "thread_limit": 1})
    completed_output(root / "quality_control" / f"{data.tag}_tsne_seed_{MAIN_SEED}.json", config, path)
    table = pd.read_csv(path, float_precision="round_trip")
    upstream_hashes = {path.as_posix(): file_hash(path)}
    require(table["sample_id"].tolist() == data.sample_map["sample_id"].tolist(), "绘图坐标样本顺序不匹配。")
    coordinates = table[["tsne_1", "tsne_2"]].to_numpy(dtype=np.float32)
    stats_path = root / "statistics" / data.tag / "kw_results.csv"
    completed_output(root / "quality_control" / f"{data.tag}_kw.json", configuration(data, "kw", KW_PARAMETERS), stats_path)
    upstream_hashes[stats_path.as_posix()] = file_hash(stats_path)
    folder = root / "figures" / data.tag
    outputs = [folder / "tsne_main_seed_222.png", folder / "tsne_main_seed_222.pdf"]
    if data.number == 1:
        outputs.append(folder / "bottleneck_64_neurons_boxplots.pdf")
    report = root / "quality_control" / f"{data.tag}_plots.json"
    config = configuration(data, "plots", {"main_seed": MAIN_SEED,
                                            "boxplot_model": 1, "boxplot_pages": 64 if data.number == 1 else 0,
                                            "upstream_sha256": upstream_hashes})
    if not begin_stage(report, config, outputs, overwrite):
        return
    save_tsne_figures(data, coordinates, outputs[:2])
    if data.number == 1:
        save_boxplots(data, pd.read_csv(stats_path), outputs[2])
    finish_stage(report, config, outputs, {"coordinates_averaged": False,
                                          "boxplot_pages": 64 if data.number == 1 else 0})
    LOGGER.info("%s plots completed; main seed=%d", data.tag, MAIN_SEED)


def self_test(data: ModelData) -> None:
    """验证秩统计、BH、阈值边界及真实输入的小样本绘图；不写正式结果目录。"""
    write_json(ROOT / "quality_control/self_test.json", {
        "status": "running", "versions": versions(), "model_number": data.number})
    rng = np.random.default_rng(222)
    toy = rng.normal(size=(9, 64))
    toy[:, 0] = [1, 1, 2, 2, 3, 4, 5, 5, 6]  # 含并列秩。
    toy[:, 1] = 1  # 恒定单元仍属于64次检验族。
    labels = np.repeat(["A", "B", "C"], 3)
    table = calculate_kw(toy, labels, 1)
    ranks = stats.rankdata(toy[:, 0])
    _, multiplicities = np.unique(toy[:, 0], return_counts=True)
    tie_factor = 1 - np.sum(multiplicities**3 - multiplicities) / (9**3 - 9)
    manual_h = (12 / (9 * 10) * sum(ranks[labels == g].sum()**2 / 3 for g in ("A", "B", "C")) - 30) / tie_factor
    np.testing.assert_allclose(table.iloc[0]["H"], manual_h, rtol=1e-12)
    np.testing.assert_allclose(table.iloc[0]["epsilon2"], manual_h / 8, rtol=1e-12)
    np.testing.assert_allclose(table.iloc[0]["p_value"], stats.chi2.sf(manual_h, 2), rtol=1e-12)
    require(table.iloc[1]["H"] == 0 and table.iloc[1]["p_value"] == 1 and not table.iloc[1]["candidate"], "恒定单元处理错误。")
    np.testing.assert_allclose(stats.false_discovery_control([0, .01, .03, .04, 1], method="bh"), [0, .025, .05, .05, 1])
    # 独立按 BH 定义验证整个64次检验族，包括并列p值和p=1。
    p = table["p_value"].to_numpy()
    order = np.argsort(p)
    expected = np.empty(64)
    expected[order] = np.minimum(1, np.minimum.accumulate((p[order] * 64 / np.arange(1, 65))[::-1])[::-1])
    np.testing.assert_allclose(table["q_value_bh"], expected, rtol=1e-12)
    np.testing.assert_array_equal(candidate_mask(np.array([.05, .049, .049]), np.array([.15, .14, .141])), [False, False, True])
    # 每组织选3个样本，所有54个标签都出现在图中；仅用于程序与版面检查。
    selected = np.sort(np.concatenate([np.flatnonzero(data.sample_map["tissue_name"].eq(t).to_numpy())[:3] for t in data.tissues]))
    subset_map = data.sample_map.iloc[selected].reset_index(drop=True)
    subset_map["row_index"] = np.arange(len(selected))
    subset_matrix = data.matrix[selected]
    subset_medians = pd.DataFrame([
        np.median(subset_matrix[subset_map["tissue_name"].eq(tissue).to_numpy()], axis=0)
        for tissue in data.tissues], columns=UNIT_COLUMNS)
    subset_medians.insert(0, "n_samples", 3)
    subset_medians.insert(0, "tissue_name", data.tissues)
    subset = replace(data, matrix=subset_matrix, sample_map=subset_map, medians=subset_medians)
    folder = ROOT / "quality_control" / "self_test"
    folder.mkdir(parents=True, exist_ok=True)
    # 调用正式函数，验证坐标/统计CSV、状态、图形及分页的完整读写链条。
    # 通过输出根目录隔离检查产物，配置明确标记 validation_subset。
    pipeline_root = folder / "pipeline"
    run_kw(subset, overwrite=True, root=pipeline_root)
    run_tsne(subset, overwrite=True, root=pipeline_root)
    run_plots(subset, overwrite=True, root=pipeline_root)
    stored_statistics = pd.read_csv(pipeline_root / "statistics" / data.tag / "kw_results.csv", float_precision="round_trip")
    expected_statistics = calculate_kw(subset.matrix, subset.sample_map["tissue_name"].to_numpy(), data.number)
    pd.testing.assert_frame_equal(stored_statistics, expected_statistics, check_exact=True)
    saved_candidates = pd.read_csv(pipeline_root / "statistics" / data.tag / "candidates.csv")
    require(saved_candidates["unit_id"].tolist() == stored_statistics.loc[stored_statistics["candidate"], "unit_id"].tolist(),
            "候选清单与双阈值标记不一致。")
    # 正式run_*函数的续跑路径也必须通过；这里不会再次执行降维。
    run_kw(subset, overwrite=False, root=pipeline_root)
    run_tsne(subset, overwrite=False, root=pipeline_root)
    run_plots(subset, overwrite=False, root=pipeline_root)
    # 保存范围和续跑保护：相同签名跳过；改变参数时必须拒绝静默复用。
    test_report, test_output = folder / "resume_test.json", folder / "resume_test.csv"
    require(begin_stage(test_report, {"test": 1}, [test_output], True), "检查初始化失败。")
    pd.DataFrame({"value": [1]}).to_csv(test_output, index=False)
    finish_stage(test_report, {"test": 1}, [test_output], {})
    require(not begin_stage(test_report, {"test": 1}, [test_output], False), "完成项未跳过。")
    try:
        begin_stage(test_report, {"test": 2}, [test_output], False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("参数变化未被阻止。")
    write_json(ROOT / "quality_control/self_test.json", {
        "status": "passed", "versions": versions(), "kw_ties_checked": True,
        "bh_64_checked": True, "constant_neuron_checked": True, "strict_thresholds_checked": True,
        "resume_checked": True, "tsne_samples": len(selected), "main_seed": MAIN_SEED,
        "boxplot_pages": 64 if data.number == 1 else 0,
        "csv_roundtrips_checked": True, "candidate_list_checked": True,
        "pipeline_output_directory": pipeline_root.as_posix(),
        "note": "Subset checks only; formal 20-model analysis has separate completion records."})


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 latent space and tissue analysis")
    parser.add_argument("stage", choices=("check", "self-test", "kw", "tsne", "plots", "all"))
    parser.add_argument("--model-start", type=int, default=1)
    parser.add_argument("--model-end", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", help="重新计算所选阶段和模型范围")
    args = parser.parse_args()
    require(1 <= args.model_start <= args.model_end <= 20, "模型范围必须为1到20。")
    require(sklearn.__version__ == "1.9.0", f"需要 scikit-learn==1.9.0，当前为 {sklearn.__version__}。")
    os.chdir(TRIAL_ROOT)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / f"{args.stage}.log", encoding="utf-8"), logging.StreamHandler()])
    try:
        summaries = []
        numbers = [args.model_start] if args.stage == "self-test" else range(args.model_start, args.model_end + 1)
        for number in numbers:
            data = load_model_data(number)
            summaries.append({"model_number": number, "n_samples": len(data.matrix),
                              "n_neurons": N_UNITS, "n_tissues": len(data.tissues),
                              "source_sha256": data.source_hashes})
            LOGGER.info("Input verified: %s (7862 x 64, 54 tissues)", data.tag)
            if args.stage == "self-test":
                self_test(data)
                return
            metadata_dir = ROOT / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"tissue_name": data.tissues, "color": list(palette_for(data.tissues).values())}).to_csv(
                metadata_dir / "tissue_colors.csv", index=False)
            if args.stage in ("kw", "all"):
                run_kw(data, args.overwrite)
            if args.stage in ("tsne", "all"):
                run_tsne(data, args.overwrite)
            if args.stage in ("plots", "all"):
                run_plots(data, args.overwrite)
        write_json(ROOT / "quality_control/input_check.json", {"status": "passed", "models": summaries})
        write_json(ROOT / "metadata/analysis_design.json", {
            "versions": versions(), "layer": LAYER_NAME, "sample_scope": "full",
            "model_numbers": list(range(1, 21)), "main_display_model": 1,
            "tsne": {**TSNE_PARAMETERS, "main_seed": MAIN_SEED, "runs_per_model": 1,
                     "user_iteration_parameter": {"n_iter": 1000}, "api_iteration_parameter": "max_iter", "thread_limit": 1},
            "kw": KW_PARAMETERS, "coordinate_averaging": False, "extra_clustering_metrics": False,
            "display": {"boxplot_model": 1, "boxplot_neurons": 64, "tissue_palette": "metadata/tissue_colors.csv"}})
        LOGGER.info("Step3 stage %s completed for requested model range.", args.stage)
    except Exception:
        # running 状态与非零退出码共同防止半成品被下游读取。
        LOGGER.exception("Step3 stage %s failed.", args.stage)
        raise


if __name__ == "__main__":
    main()

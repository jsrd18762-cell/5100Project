"""Step6：Jacobian 来源的瓶颈 NGCI 与组织间 Pearson 相关性。

从 trial 或其子目录调用：
    python step6/code/step6_ngci.py check
    python step6/code/step6_ngci.py self-test
    python step6/code/step6_ngci.py run

每个模型的每个神经元独立使用全部 53 个可匹配组织。输入为完整样本的
组织激活中位数及同组织 GSEA ES；NGCI 定义为带符号 Pearson r。
采用双侧检验，不做多重校正，不以 p 值筛选神经元，不平均不同模型。
本文件只写 step6；不会启动 R、模型训练或 GSEA 计算。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import warnings

import matplotlib
matplotlib.use("Agg")  # Notebook 子进程只保存图形，避免弹出阻塞窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn


TRIAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("step6")
CODE = ROOT / "code/step6_ngci.py"
LAYER = "encoder_linear_512_to_64"
N_MODELS, N_UNITS, N_TISSUES = 20, 64, 53
UNITS = [f"unit_{i:05d}" for i in range(N_UNITS)]
EXCLUDED_TISSUE = "Cells-Leukemiacellline(CML)"
LOGGER = logging.getLogger("step6")
DESIGN = {
    "score_source": "jacobian", "gsea_score": "ES", "ngci_definition": "signed Pearson r",
    "reference_id": "tissue_top100_mean_tpm_v1",
    "activation_layer": LAYER, "activation_summary": "tissue median over full samples",
    "analysis_unit": "one model x one neuron", "n_models": N_MODELS, "n_units": N_UNITS,
    "n_tissues": N_TISSUES, "tissue_weighting": "equal", "sample_scope": "full",
    "correlation_method": "Pearson", "implementation": "scipy.stats.pearsonr",
    "alternative": "two-sided", "null_hypothesis": "rho = 0", "alternative_hypothesis": "rho != 0",
    "degrees_of_freedom": 51, "p_value_threshold_strict": 0.05,
    "p_roundoff_policy": "clip only boundary roundoff within 1e-14; also retain raw SciPy p",
    "multiple_testing_correction": "none", "p_value_used_for_selection": False,
    "gsea_significance_used_for_selection": False, "pairwise_deletion": False,
    "excluded_tissue": EXCLUDED_TISSUE, "exclusion_reason": "no same-tissue reference set",
    "constant_vector_policy": "retain neuron; r and p are NA with an explicit status",
    "quartile_method": "linear", "model_aggregation": "within-model description only",
    "display_model": "model_01", "display_unit": "unit_00000",
    "display_highlights": "fixed Step5 selection; all 53 tissues enter the test",
    "random_sampling": "none; deterministic analysis",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, content: dict) -> None:
    """状态原子更新；JSON 禁止 NaN，统计缺失值只在 CSV 中以 NA 表示。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__}


def pair_model(scores: pd.DataFrame, medians: pd.DataFrame, tissues: pd.DataFrame) -> pd.DataFrame:
    """按组织名称及 unit_id 配对，不能依靠 CSV 当前行序横向拼接。"""
    require(list(medians.columns) == ["tissue_name", "n_samples", *UNITS], "中位数表的列或神经元顺序错误。")
    require(len(medians) == len(tissues) and medians.tissue_name.is_unique
            and set(medians.tissue_name) == set(tissues.tissue_name), "中位数表存在重复、缺失或未知组织。")
    aligned = medians.set_index("tissue_name").loc[tissues.tissue_name]
    require(np.array_equal(aligned.n_samples.to_numpy(), tissues.n_samples.to_numpy()), "组织样本数不匹配。")
    require(np.isfinite(aligned[UNITS].to_numpy(dtype=np.float64)).all(), "组织激活中位数含 NaN/Inf。")
    require(not scores.duplicated(["source_tissue_id", "unit_id"]).any(), "同组织 ES 存在重复记录。")
    long = medians.melt(id_vars=["tissue_name", "n_samples"], var_name="unit_id", value_name="activation_median")
    result = scores.merge(long, left_on=["source_tissue", "unit_id"], right_on=["tissue_name", "unit_id"],
                          how="left", validate="one_to_one", indicator=True)
    require(result._merge.eq("both").all(), "ES 与组织激活中位数未能完整配对。")
    return result.drop(columns=["tissue_name", "_merge"])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """检查直接输入及小型来源记录，不重新加载大型表达矩阵或 Jacobian。

    来源链为 finalize -> fgsea 完成记录 -> prepared 记录 -> Step4 模型记录
    -> Step2 中位数指纹。只核对本步依赖的组织和文件，不重跑上游算法。
    """
    sources = {}

    def fingerprint(path: Path, expected: str | None = None) -> str:
        key = path.as_posix()
        if key not in sources:
            sources[key] = sha(path)
        if expected is not None:
            require(sources[key] == expected, f"输入指纹不一致：{path}")
        return sources[key]

    def completed(path: Path) -> dict:
        fingerprint(path)
        record = read_json(path)
        require(record.get("status") == "complete", f"上游阶段尚未完成：{path}")
        return record

    def recorded_csv(path: Path, record: dict) -> pd.DataFrame:
        fingerprint(path, record["output_sha256"][path.as_posix()])
        return pd.read_csv(path, float_precision="round_trip")

    final_path = Path("step5/quality_control/finalize.json")
    reference_path = Path("step5/quality_control/reference.json")
    final, reference = completed(final_path), completed(reference_path)
    require(reference.get("configuration", {}).get("design", {}).get("reference_selection", {}).get("id")
            == DESIGN["reference_id"], "Step6需要平均TPM Top100参考库；请先完成Step5 prepare、R GSEA及finalize。")
    require(final.get("configuration", {}).get("design", {}).get("reference_selection", {}).get("id")
            == DESIGN["reference_id"], "Step5统计结果仍属于旧参考库，需要重新计算。")
    fingerprint(reference_path, final["configuration"]["reference_record_sha256"])
    tissues = recorded_csv(Path("step5/reference/tissue_map.csv"), reference)
    refs = recorded_csv(Path("step5/reference/reference_sets.csv"), reference)
    selection_path = Path("step5/reference/reference_selection.json")
    fingerprint(selection_path, reference["output_sha256"][selection_path.as_posix()])
    require(read_json(selection_path)["specification"]["id"] == DESIGN["reference_id"]
            and refs.n_matched.eq(np.minimum(refs.n_candidates_matched, 100)).all(), "Top100参考成员数量不一致。")
    display = recorded_csv(Path("step5/reference/display_selection.csv"), reference)
    matched = recorded_csv(Path("step5/statistics/matched_gsea.csv"), final)
    require(len(tissues) == 54 and tissues.tissue_id.is_unique and tissues.tissue_name.is_unique
            and tissues.n_samples.gt(0).all() and tissues.n_samples.sum() == 7862, "完整组织映射错误。")
    require(len(refs) == N_TISSUES and refs.reference_tissue_id.is_unique
            and refs.reference_tissue.is_unique, "必须有 53 个唯一同组织参考集合。")
    names = tissues.set_index("tissue_id").tissue_name
    require(refs.reference_tissue_id.map(names).eq(refs.reference_tissue).all(), "来源和参考组织名称不一致。")
    excluded = tissues.loc[~tissues.tissue_id.isin(refs.reference_tissue_id)]
    require(excluded.tissue_name.tolist() == [EXCLUDED_TISSUE], "无参考组织不符合既定输入。")

    # 保留所有 20 x 64 x 54 条上游记录做覆盖检查，再按参考可用性取出 53 组织。
    keys = ["model_number", "neuron_index", "source_tissue_id"]
    expected = pd.MultiIndex.from_product([range(1, 21), range(64), tissues.tissue_id], names=keys)
    actual = pd.MultiIndex.from_frame(matched[keys])
    require(actual.is_unique and len(actual) == len(expected) and len(expected.difference(actual)) == 0,
            "同组织 GSEA 表未完整覆盖 20 x 64 x 54 个唯一条目。")
    require(matched.unit_id.eq(matched.neuron_index.map(lambda u: f"unit_{u:05d}")).all(), "神经元编号映射错误。")
    require(matched.source_tissue_id.map(names).eq(matched.source_tissue).all(), "GSEA 来源组织 ID/名称错误。")
    available = matched.source_tissue_id.isin(refs.reference_tissue_id)
    require(matched.reference_available.eq(available).all(), "参考可用性标记与组织参考库不一致。")
    selected = matched.loc[available].copy()
    require(selected.source_tissue_id.eq(selected.reference_tissue_id).all()
            and selected.source_tissue.eq(selected.reference_tissue).all(), "输入不是同组织 GSEA ES。")
    require(np.isfinite(selected.ES.to_numpy(dtype=np.float64)).all()
            and selected.ES.abs().le(1 + 1e-12).all(), "ES 含缺失、无穷或超出合法范围的值。")

    columns = ["model_number", "neuron_index", "unit_id", "source_tissue_id", "source_tissue",
               "reference_tissue_id", "reference_tissue", "ES"]
    frames = []
    for number in range(1, N_MODELS + 1):
        tag = f"model_{number:02d}"
        model_path = Path(f"step4/quality_control/{tag}/model.json")
        model = completed(model_path)
        cfg = model["configuration"]
        require(cfg["model_number"] == number and cfg["model_seed"] == 629 + number
                and cfg["final_epoch"] == 100 and cfg["layer"] == LAYER and cfg["sample_scope"] == "full"
                and cfg["n_samples"] == 7862 and cfg["candidate_neuron_indices"] == list(range(N_UNITS)),
                f"{tag}: 模型、层、样本范围或候选编号不匹配。")
        median_path = Path(f"step2/tissue_medians/{tag}/{LAYER}.csv")
        qc_path = Path(f"step2/quality_control/{tag}.json")
        fingerprint(median_path, cfg["source_sha256"][median_path.as_posix()])
        fingerprint(qc_path, cfg["source_sha256"][qc_path.as_posix()])
        qc = completed(qc_path)
        require(qc["median_scope"] == "full" and qc["sample_coverage_verified"], f"{tag}: 中位数范围未验证。")

        for tissue_id in refs.reference_tissue_id:
            r_path = Path(f"step5/quality_control/fgsea/{tag}/{tissue_id}.json")
            prepared_path = Path(f"step5/quality_control/prepared/{tag}/{tissue_id}.json")
            fingerprint(r_path, final["configuration"]["r_records_sha256"][r_path.as_posix()])
            r_record = completed(r_path)
            fingerprint(prepared_path, r_record["configuration"]["prepared_record_sha256"])
            prepared = completed(prepared_path)
            pcfg = prepared["configuration"]
            require(pcfg["model_number"] == number and pcfg["source_tissue_id"] == tissue_id
                    and pcfg["n_candidates"] == N_UNITS, f"{tag}/{tissue_id}: GSEA 来源配置错误。")
            fingerprint(model_path, prepared["source_sha256"][model_path.as_posix()])
            fingerprint(reference_path, r_record["configuration"]["reference_record_sha256"])

        scores = selected.loc[selected.model_number.eq(number), columns]
        medians = pd.read_csv(median_path, float_precision="round_trip")
        frames.append(pair_model(scores, medians, tissues))
        LOGGER.info("Input checked: %s, 64 neurons x 53 tissues", tag)

    pairs = pd.concat(frames, ignore_index=True).sort_values(keys).reset_index(drop=True)
    pairs.insert(1, "model_id", pairs.model_number.map(lambda n: f"model_{n:02d}"))
    pairs["score_source"], pairs["gsea_score"] = "jacobian", "ES"
    pairs["activation_layer"], pairs["sample_scope"] = LAYER, "full"
    pairs["display_group"] = "other"
    require(len(display) == 6 and display.model_number.eq(1).all() and display.neuron_index.eq(0).all()
            and display.unit_id.eq("unit_00000").all() and display.source_tissue_id.is_unique
            and display.groupby("group").size().to_dict() == {"high": 3, "low": 3}, "固定六组织展示表错误。")
    for row in display.itertuples(index=False):
        mask = pairs.model_number.eq(1) & pairs.neuron_index.eq(0) & pairs.source_tissue_id.eq(row.source_tissue_id)
        require(mask.sum() == 1 and pairs.loc[mask, "source_tissue"].iloc[0] == row.source_tissue
                and np.isclose(pairs.loc[mask, "activation_median"].iloc[0], row.median_activation, rtol=0, atol=1e-12),
                "固定展示组织的名称或激活值与输入不一致。")
        pairs.loc[mask, "display_group"] = row.group
    for group in ("high", "low"):
        require(sorted(display.loc[display.group.eq(group), "rank_in_group"]) == [1, 2, 3], "展示组织排名错误。")

    diagnostics = {
        "n_upstream_rows": len(matched), "n_tissue_pairs": len(pairs), "n_neurons": N_MODELS * N_UNITS,
        "n_tissues_per_neuron": N_TISSUES, "excluded_tissues": excluded.to_dict("records"),
        "n_excluded_no_reference_rows": int((~available).sum()),
        "n_retained_rows_with_missing_gsea_p": int(selected.pval.isna().sum()),
        "n_retained_rows_with_missing_NES": int(selected.NES.isna().sum()),
        "n_nonfinite_ES": 0, "display_selection": display.to_dict("records"),
        "provenance_check_scope": "direct inputs and upstream completion-record chain; large arrays not reread",
    }
    return pairs, display, sources, diagnostics


def pearson_result(x: np.ndarray, y: np.ndarray) -> dict:
    """常规双侧 Pearson；n 是组织数，不是组织内样本数。

    SciPy 默认使用零相关假设下 r 的精确 Beta 分布计算双侧 p，等价于
    t = r * sqrt((n-2)/(1-r^2)) 的双侧 t 检验。这里不使用置换或抽样。
    常量向量的相关性未定义，应保留 NA，不能将它填为 0 或解释为不显著。
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.shape == y.shape == (N_TISSUES,), "每次 Pearson 检验必须有完整 53 对组织值。")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "相关性输入有 NaN/Inf；不执行自动删行。")
    constant = []
    if np.ptp(x) == 0:
        constant.append("activation")
    if np.ptp(y) == 0:
        constant.append("ES")
    if constant:
        return {"ngci": np.nan, "pearson_r": np.nan, "pearson_p": np.nan, "pearson_p_raw": np.nan,
                "p_value_roundoff_clipped": False, "p_lt_0_05": pd.NA,
                "status": "undefined_constant_" + "_and_".join(constant)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", stats.NearConstantInputWarning)
        result = stats.pearsonr(x, y, alternative="two-sided")
    r, raw_p = float(result.statistic), float(result.pvalue)
    require(np.isfinite(r) and np.isfinite(raw_p) and -1 <= r <= 1 and -1e-14 <= raw_p <= 1 + 1e-14,
            "Pearson 返回无效结果。")
    # SciPy 的 Beta 尾概率在 r 接近 0 时可能比 1 大几个浮点舍入单位。
    # 仅将该边界误差限制到概率区间，同时保留未经处理的值供核对。
    p = float(np.clip(raw_p, 0, 1))
    near_constant = any(issubclass(w.category, stats.NearConstantInputWarning) for w in caught)
    return {"ngci": r, "pearson_r": r, "pearson_p": p, "pearson_p_raw": raw_p,
            "p_value_roundoff_clipped": p != raw_p, "p_lt_0_05": p < 0.05,
            "status": "ok_near_constant_warning" if near_constant else "ok"}


def calculate(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每模型保存全部神经元；模型汇总仅是该模型内 r 分布的描述统计。"""
    rows = []
    for (number, unit), group in pairs.groupby(["model_number", "neuron_index"], sort=True):
        require(group.source_tissue_id.nunique() == N_TISSUES, "同一神经元的组织重复或缺失。")
        result = pearson_result(group.activation_median.to_numpy(), group.ES.to_numpy())
        rows.append({"model_number": int(number), "model_id": f"model_{number:02d}",
                     "neuron_index": int(unit), "unit_id": f"unit_{unit:05d}",
                     "score_source": "jacobian", "gsea_score": "ES", "n_tissues": N_TISSUES,
                     "degrees_of_freedom": N_TISSUES - 2, **result,
                     "alternative": "two-sided", "multiple_testing_correction": "none",
                     "p_value_used_for_selection": False})
    neurons = pd.DataFrame(rows)
    neurons["p_lt_0_05"] = neurons.p_lt_0_05.astype("boolean")
    summaries = []
    for number, group in neurons.groupby("model_number", sort=True):
        valid = group.ngci.dropna()
        q1, median, q3 = valid.quantile([0.25, 0.5, 0.75], interpolation="linear")
        significant = int(group.p_lt_0_05.sum())
        summaries.append({"model_number": int(number), "model_id": f"model_{number:02d}",
                          "score_source": "jacobian", "gsea_score": "ES", "n_neurons": len(group),
                          "n_valid": len(valid), "n_undefined": len(group) - len(valid),
                          "ngci_median": median, "ngci_q1": q1, "ngci_q3": q3, "ngci_iqr": q3 - q1,
                          "ngci_min": valid.min(), "ngci_max": valid.max(),
                          "n_p_lt_0_05": significant,
                          "fraction_p_lt_0_05_among_valid": significant / len(valid) if len(valid) else np.nan})
    return neurons, pd.DataFrame(summaries)


def plot_scatter(pairs: pd.DataFrame, display: pd.DataFrame, result: pd.Series, path: Path) -> None:
    """固定主图：53 个组织等大小；高低六组织由 Step5 清单着色并编号。

    右侧列出完整组织名称，避免长名称遮挡数据。所有 53 点共同计算 r/p，
    右侧清单只是图例。没有按结果选择神经元，也没有添加额外回归检验。
    """
    group = pairs.loc[pairs.model_number.eq(1) & pairs.neuron_index.eq(0)]
    require(len(group) == N_TISSUES, "主图必须包含 53 个组织。")
    fig, (ax, legend_ax) = plt.subplots(1, 2, figsize=(11.8, 6.2),
                                      gridspec_kw={"width_ratios": [1.55, 1]}, layout="constrained")
    colors = {"other": "#8b949c", "high": "#bf4d36", "low": "#007d88"}
    markers = {"other": "o", "high": "^", "low": "s"}
    for kind in ("other", "low", "high"):
        points = group.loc[group.display_group.eq(kind)]
        ax.scatter(points.activation_median, points.ES, s=48, c=colors[kind], marker=markers[kind],
                   edgecolors="white", linewidths=0.5, alpha=0.9, zorder=3)
    ax.set(xlabel="Tissue median activation", ylabel="Same-tissue GSEA ES (Jacobian)", ylim=(-1.08, 1.08))
    ax.margins(x=0.12)
    ax.grid(alpha=0.16, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Bottleneck NGCI | model_01 / unit_00000", fontsize=15)
    legend_ax.axis("off")
    p_label = f"{result.pearson_p:.6g}" if np.isfinite(result.pearson_p) else "NA"
    r_label = f"{result.pearson_r:.6f}" if np.isfinite(result.pearson_r) else "NA"
    legend_ax.text(0.02, 0.97, f"Pearson r = {r_label}\np = {p_label}\nn = 53 tissues",
                   va="top", fontsize=12, linespacing=1.6, transform=legend_ax.transAxes)
    for kind, top in (("high", 0.65), ("low", 0.34)):
        legend_ax.text(0.02, top, f"{kind.capitalize()} activation", color=colors[kind],
                       fontsize=11, fontweight="bold", transform=legend_ax.transAxes)
        for index, row in enumerate(display.loc[display.group.eq(kind)].sort_values("rank_in_group").itertuples(index=False)):
            label = f"{kind[0].upper()}{row.rank_in_group}"
            point = group.loc[group.source_tissue_id.eq(row.source_tissue_id)].iloc[0]
            offsets = {"high": [(7, 9), (7, -15), (7, -15)], "low": [(7, 9), (7, 9), (7, -15)]}
            offset = offsets[kind][row.rank_in_group - 1]
            ax.annotate(label, (point.activation_median, point.ES), xytext=offset, textcoords="offset points",
                        fontsize=9, color=colors[kind], fontweight="bold")
            name = row.source_tissue.replace("-", "-\n", 1) if len(row.source_tissue) > 28 else row.source_tissue
            legend_ax.text(0.02, top - 0.055 - index * 0.078, f"{label}  {name}", fontsize=9,
                           va="top", linespacing=1.15, transform=legend_ax.transAxes)
    legend_ax.text(0.02, 0.015, "Gray: remaining 47 tissues", color=colors["other"], fontsize=9,
                   transform=legend_ax.transAxes)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def self_test() -> None:
    """用已知相关性的 53 对合成值独立核对公式、方向、缺失处理和键连接。

    合成数据仅驻留内存，不写正式统计目录，也不借用实际结果当作期望值。
    """
    x = np.linspace(-1, 1, N_TISSUES)
    x /= np.linalg.norm(x)
    z = x**2 - np.mean(x**2)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)
    y = 0.54 * x + np.sqrt(1 - 0.54**2) * z
    result = pearson_result(x, y)
    expected_p = 2 * stats.t.sf(abs(0.54) * np.sqrt(51 / (1 - 0.54**2)), df=51)
    np.testing.assert_allclose(result["pearson_r"], 0.54, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["pearson_p"], expected_p, rtol=1e-12)
    np.testing.assert_allclose(pearson_result(x, -y)["pearson_r"], -0.54, atol=1e-14)
    np.testing.assert_allclose(pearson_result(x, -y)["pearson_p"], expected_p, rtol=1e-12)
    np.testing.assert_allclose(pearson_result(y, x)["pearson_p"], expected_p, rtol=1e-12)
    require(result["p_lt_0_05"] and not pearson_result(x, z)["p_lt_0_05"], "显著性标记错误。")
    require(pd.isna(pearson_result(x, np.ones(53))["p_lt_0_05"]), "未定义的 p 不能标为不显著。")

    def must_fail(function, *arguments) -> None:
        try:
            function(*arguments)
        except ValueError:
            return
        raise AssertionError("无效输入没有被拒绝。")

    must_fail(pearson_result, x[:-1], y[:-1])
    must_fail(pearson_result, x, np.where(np.arange(53) == 0, np.nan, y))
    tissues = pd.DataFrame({"tissue_id": [f"tissue_{i:02d}" for i in range(1, 55)],
                            "tissue_name": [f"T{i}" for i in range(54)], "n_samples": np.arange(1, 55)})
    medians = pd.concat([tissues[["tissue_name", "n_samples"]],
                         pd.DataFrame({unit: np.arange(54) + u for u, unit in enumerate(UNITS)})], axis=1)
    scores = pd.DataFrame({"source_tissue_id": tissues.tissue_id[:53], "source_tissue": tissues.tissue_name[:53],
                           "model_number": 1, "neuron_index": 0, "unit_id": "unit_00000", "ES": y})
    paired = pair_model(scores.iloc[::-1], medians.iloc[::-1], tissues)
    np.testing.assert_array_equal(paired.sort_values("source_tissue_id").activation_median, np.arange(53))
    must_fail(pair_model, pd.concat([scores, scores.iloc[:1]]), medians, tissues)
    must_fail(pair_model, scores, medians.iloc[:-1], tissues)
    a, summary_a = calculate(paired)
    altered = paired.copy()
    altered["n_samples"] = 100000
    altered["display_group"] = "high"
    altered["pval"], altered["NES"], altered["q_matched_global"] = np.nan, np.nan, np.nan
    b, summary_b = calculate(altered)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    require(not any(c.startswith("q_") for c in a.columns), "本步不应生成 q 值。")
    write_json(ROOT / "quality_control/self_test.json", {
        "status": "passed", "script_sha256": sha(CODE), "versions": versions(),
        "known_r": 0.54, "n_tissues": 53, "df": 51, "expected_two_sided_p": float(expected_p),
        "checks": ["known signed correlation", "independent two-sided t formula", "negative correlation",
                   "x/y exchange", "p threshold", "undefined constant vector", "reject 52 pairs",
                   "reject nonfinite values", "join after row reordering", "reject duplicate/missing tissues",
                   "no sample-count weighting", "display groups do not select observations",
                   "missing GSEA p/NES/q do not select observations", "no Pearson q column"],
    })


def run(overwrite: bool) -> None:
    """读取小型输入后一次完成全模型分析，已完成且指纹一致时直接跳过。"""
    pairs, display, sources, diagnostics = load_inputs()
    config = {"design": DESIGN, "source_sha256": sources, "script_sha256": sha(CODE), "versions": versions()}
    report = ROOT / "quality_control/run.json"
    base = ROOT / "figures/model_01/unit_00000_ngci_scatter"
    outputs = [ROOT / f"statistics/{name}.csv" for name in
               ("ngci_tissue_pairs", "ngci_per_neuron", "ngci_model_summary")]
    outputs += [base.with_suffix(".png"), base.with_suffix(".pdf")]
    if report.exists() and not overwrite:
        saved = read_json(report)
        require(saved.get("status") == "complete" and saved.get("configuration") == config,
                "Step6 未完成或输入/配置已改变；检查原因后使用 --overwrite。")
        require(set(saved.get("output_sha256", {})) == {p.as_posix() for p in outputs}
                and all(p.is_file() and sha(p) == saved["output_sha256"][p.as_posix()] for p in outputs),
                "Step6 输出缺失或已改变；检查原因后使用 --overwrite。")
        LOGGER.info("Skip completed Step6: configuration and all output fingerprints match.")
        return
    require(overwrite or not any(p.exists() for p in outputs), "存在未完成输出；检查原因后使用 --overwrite。")
    started = datetime.now(timezone.utc).isoformat()
    write_json(report, {"status": "running", "started_utc": started, "configuration": config})
    try:
        neurons, summary = calculate(pairs)
        require(len(pairs) == 67840 and len(neurons) == 1280 and len(summary) == 20
                and summary.n_neurons.eq(64).all(), "全模型分析输出数量不符合设计。")
        for path, frame in zip(outputs[:3], (pairs, neurons, summary)):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, float_format="%.17g", na_rep="NA")
        selected = neurons.loc[neurons.model_number.eq(1) & neurons.neuron_index.eq(0)].iloc[0]
        plot_scatter(pairs, display, selected, base)
        write_json(report, {"status": "complete", "started_utc": started,
                            "completed_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
                            "input_diagnostics": diagnostics, "n_neurons": len(neurons),
                            "n_valid": int(neurons.ngci.notna().sum()),
                            "n_p_roundoff_clipped": int(neurons.p_value_roundoff_clipped.sum()),
                            "n_near_constant_warnings": int(neurons.status.eq("ok_near_constant_warning").sum()),
                            "output_sha256": {p.as_posix(): sha(p) for p in outputs}})
        LOGGER.info("Complete: %d tissue pairs, %d neurons, %d model summaries", len(pairs), len(neurons), len(summary))
        LOGGER.info("model_01 / unit_00000: r=%.9g, p=%.9g, n=53", selected.pearson_r, selected.pearson_p)
    except Exception as error:
        write_json(report, {"status": "failed", "started_utc": started, "configuration": config, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6 Jacobian ES and tissue activation NGCI")
    parser.add_argument("stage", choices=("check", "self-test", "run"))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rebuild Step6 analysis outputs")
    args = parser.parse_args()
    os.chdir(TRIAL_ROOT)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / f"logs/{args.stage}.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    require(sklearn.__version__ == "1.9.0", "需要 scikit-learn==1.9.0；本步相关性由 SciPy 计算。")
    require(not args.overwrite or args.stage == "run", "--overwrite 仅用于 run。")
    if args.stage == "run":
        run(args.overwrite)
        return
    report = ROOT / "quality_control" / ("input_check.json" if args.stage == "check" else "self_test.json")
    write_json(report, {"status": "running"})
    try:
        if args.stage == "self-test":
            self_test()
        else:
            _, _, sources, diagnostics = load_inputs()
            write_json(report, {"status": "passed", "design": DESIGN, "versions": versions(),
                                "script_sha256": sha(CODE), "source_sha256": sources, **diagnostics})
        LOGGER.info("%s passed", args.stage)
    except Exception as error:
        write_json(report, {"status": "failed", "error": str(error), "script_sha256": sha(CODE)})
        raise


if __name__ == "__main__":
    main()

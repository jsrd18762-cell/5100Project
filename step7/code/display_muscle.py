"""固定展示 model_11 / unit_00014，读取 Step3–7 已完成的结果。

从 trial 或其子目录运行：python -B step7/code/display_muscle.py
本脚本只更新 step7/display_muscle；不训练模型、不重新求导或运行 GSEA。
对象由用户在六组织两路线比较后指定；仍核验 Step3 候选资格和骨骼肌最高激活。
高、低各三组织按该对象完整样本的组织激活中位数确定，不按 ES 选择组织。
"""
from pathlib import Path
import hashlib
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUT = Path("step7/display_muscle")
LAYER = "encoder_linear_512_to_64"
MUSCLE = "Muscle-Skeletal"
# 模型号从 1 开始，神经元索引从 0 开始；固定对象不随其他模型的激活值变化。
DISPLAY_MODEL = 11
DISPLAY_NEURON = 14
DISPLAY_UNIT = f"unit_{DISPLAY_NEURON:05d}"
COLORS = {"high": "#bf4d36", "low": "#007d88", "other": "#969da3"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trial_root():
    """只依赖目录结构定位工作目录，代码中不保存本机绝对路径。"""
    for folder in (Path.cwd(), *Path.cwd().parents):
        if (folder / "step7/code/display_muscle.py").is_file():
            os.chdir(folder)
            return
    raise RuntimeError("请从 trial 或其子目录启动。")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Audit:
    """核对消费的文件与计算时的指纹，另存展示来源，不改写计算记录。"""
    def __init__(self):
        self.sources = {}

    def track(self, path, expected=None):
        name = Path(path).as_posix()
        value = digest(name)
        require(expected is None or value == expected, f"来源已改变：{name}")
        self.sources[name] = value
        return name

    def record(self, path):
        record = read_json(self.track(path))
        require(record.get("status") == "complete", f"计算未完成：{path}")
        return record

    def checked(self, path, hashes):
        name = Path(path).as_posix()
        require(name in hashes, f"计算记录未包含输入：{name}")
        return self.track(name, hashes[name])


def select_object(ranks, kw):
    """核验固定对象，并保留全部 1,280 个神经元的候选资格审计。

    用户查看六组织两路线比较后指定 model_11 / unit_00014。
    此处不重新优化 ES、NGCI 或 Spearman；缺失或不符合资格时停止，
    不自动替换为激活值更高的对象。组织选择仍只使用激活中位数。
    """
    keys = ["model_number", "unit_id"]
    require(not kw.duplicated(keys).any(), "神经元统计主键重复。")
    require(not ranks.duplicated(keys + ["tissue_name"]).any(), "组织中位数主键重复。")
    require(np.isfinite(ranks.median_activation).all(), "组织中位数含非有限值。")
    maximum = ranks.groupby(keys).median_activation.max().rename("maximum_median")
    muscle = ranks.loc[ranks.tissue_name.eq(MUSCLE)].merge(maximum, on=keys, validate="one_to_one")
    require(len(muscle) == len(kw), "部分神经元缺少骨骼肌组织。")
    muscle = muscle.merge(kw, on=keys, validate="one_to_one", suffixes=("_muscle", ""))
    eligible = muscle.candidate.eq(True) & muscle.median_activation.eq(muscle.maximum_median)
    muscle["eligible"] = eligible
    muscle = muscle.sort_values(["median_activation", "model_number", "neuron_index"],
                                ascending=[False, True, True], kind="stable")
    muscle["selected_for_display"] = (muscle.model_number.eq(DISPLAY_MODEL)
                                      & muscle.neuron_index.eq(DISPLAY_NEURON)
                                      & muscle.unit_id.eq(DISPLAY_UNIT))
    selected = muscle.loc[muscle.selected_for_display]
    require(len(selected) == 1, f"固定展示对象 model_{DISPLAY_MODEL:02d} / {DISPLAY_UNIT} 缺失或不唯一。")
    require(bool(selected.iloc[0].eligible), "固定对象必须是 Step3 候选且骨骼肌激活中位数最高；不自动更换对象。")
    return selected.iloc[0], muscle


def high_low(medians):
    """在全部54组织内独立选高/低各三；并列按组织ID。不依据参考库筛选。"""
    parts = []
    for group, ascending in (("high", False), ("low", True)):
        part = medians.sort_values(["median_activation", "source_tissue_id"], ascending=[ascending, True]).head(3).copy()
        part["group"] = group
        part["rank_in_group"] = np.arange(1, 4)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    require(result.source_tissue_id.nunique() == 6, "高低六组织重叠，无法按当前规则展示。")
    return result


def focus(table, model, neuron):
    return table.loc[table.model_number.eq(model) & table.neuron_index.eq(neuron)].copy()


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=200)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def scatter(pairs, result, selection, route, base):
    """两条路线使用相同坐标方向、等大小的53组织点及同一六组织图例。"""
    fig, (ax, legend) = plt.subplots(1, 2, figsize=(11.8, 6.2), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    for group, marker in (("other", "o"), ("low", "s"), ("high", "^")):
        rows = pairs.loc[pairs.display_group.eq(group)]
        ax.scatter(rows.activation_median, rows.ES, s=46, color=COLORS[group], marker=marker,
                   edgecolors="white", linewidths=.5, zorder=3)
    ax.set(xlabel="Tissue median activation", ylabel=f"Same-tissue GSEA ES ({route})", ylim=(-1.08, 1.08))
    ax.margins(x=.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    fig.suptitle(f"Bottleneck NGCI: {route}", fontsize=14)
    legend.axis("off")
    legend.text(0, .97, f"Pearson r = {result.pearson_r:.6f}\np = {result.pearson_p:.6g}\nn = {len(pairs)} tissues",
                va="top", fontsize=12, linespacing=1.6)
    for group, top in (("high", .65), ("low", .32)):
        legend.text(0, top, f"{group.capitalize()} activation", color=COLORS[group], fontsize=11)
        for i, row in enumerate(selection.loc[selection.group.eq(group)].itertuples(index=False)):
            label = f"{group[0].upper()}{row.rank_in_group}"
            point = pairs.loc[pairs.source_tissue_id.eq(row.source_tissue_id)]
            if len(point):
                point = point.iloc[0]
                ax.annotate(label, (point.activation_median, point.ES), xytext=(7, 8 if i != 1 else -14),
                            textcoords="offset points", fontsize=9, color=COLORS[group])
            name = textwrap.fill(row.source_tissue, width=37, break_long_words=False, break_on_hyphens=True)
            legend.text(0, top - .06 - i * .08, f"{label}  {name}", fontsize=9, va="top")
    save_figure(fig, base)


def boxplot(values, samples, medians, kw, base):
    """使用逐样本激活值绘制54组织分布；零p/q按浮点下溢显示。"""
    ordered = medians.sort_values(["median_activation", "source_tissue_id"], ascending=[False, True])
    groups = [values[samples.tissue_name.eq(name)] for name in ordered.source_tissue]
    require(all(len(a) == n for a, n in zip(groups, ordered.n_samples)), "箱线图组织样本数不符。")
    require(np.allclose([np.median(a) for a in groups], ordered.median_activation, rtol=2e-7, atol=2e-7),
            "逐样本输出与组织中位数不符。")
    fig, ax = plt.subplots(figsize=(12, 14))
    fig.subplots_adjust(left=.40, right=.96, bottom=.06, top=.92)
    boxes = ax.boxplot(groups, orientation="horizontal", patch_artist=True, widths=.55,
                      flierprops={"markersize": 1.6, "alpha": .4}, medianprops={"color": "black"})
    for patch, name in zip(boxes["boxes"], ordered.source_tissue):
        patch.set_facecolor(COLORS["high"] if name == MUSCLE else "#afc7d5")
    ax.set_yticks(np.arange(1, len(groups) + 1),
                  [f"{name} (n={n})" for name, n in zip(ordered.source_tissue, ordered.n_samples)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Bottleneck activation (linear output)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    def probability(value):
        return "below floating-point range" if value == 0 else f"{value:.4g}"
    fig.suptitle("Tissue activation: skeletal-muscle-highest neuron", fontsize=15, y=.974)
    fig.text(.04, .936, f"KW H = {kw.H:.4f} | p: {probability(kw.p_value)} | BH q: {probability(kw.q_value_bh)}\n"
             f"epsilon2 = {kw.epsilon2:.6f}; H / (n - 1); 54 tissues", fontsize=10)
    save_figure(fig, base)


def require_current_weight_results():
    """传播层次变更后必须先重算Step7；不能只刷新图形沿用旧评分。"""
    from step7_weight_analysis import DESIGN
    design = read_json("step7/reference/analysis_design.json")
    final = read_json("step7/quality_control/finalize.json")
    require(design == DESIGN and final.get("status") == "complete" and
            final.get("configuration", {}).get("design") == DESIGN,
            "Step7尚未完成当前decoder[1:]分析：先重跑prepare、compare、R GSEA及finalize，再生成展示。")
    for name, expected in final["configuration"]["code_sha256"].items():
        require(digest(name) == expected, f"Step7计算代码与结果不匹配，请先重建结果：{name}")


def load_current_display():
    """Notebook只展示当前来源和输出指纹均通过核验的文件。"""
    trial_root()
    require_current_weight_results()
    record = read_json(OUT / "quality_control/python.json")
    require(record.get("status") == "complete", "请先运行 display_muscle.py。")
    require(record.get("model_number") == DISPLAY_MODEL and record.get("neuron_index") == DISPLAY_NEURON
            and record.get("unit_id") == DISPLAY_UNIT and record.get("selection_mode") == "fixed_user_choice",
            f"展示记录不是固定对象 model_{DISPLAY_MODEL:02d} / {DISPLAY_UNIT}，请重新运行 display_muscle.py。")
    for mapping in (record["source_sha256"], record["output_sha256"]):
        for name, expected in mapping.items():
            require(digest(name) == expected, f"展示输入或输出已改变，请重新运行展示脚本：{name}")
    return record


def show_enrichment(record, comparison=True):
    """R图形必须同时对应本次Python展示记录，防止显示旧对象曲线。"""
    from IPython.display import Image, display
    path = OUT / "quality_control/enrichment.json"
    if not path.is_file():
        print('请在 R 中运行 source("step7/code/03_plot_muscle_enrichment.R", encoding="UTF-8")。')
        return
    report = read_json(path)
    require(report.get("status") == "complete", "R绘图尚未完成。")
    for mapping in (report["source_sha256"], report["output_sha256"]):
        for name, expected in mapping.items():
            require(digest(name) == expected, "R图形已过期，请重新运行03_plot_muscle_enrichment.R。")
    suffix = "enrichment_comparison" if comparison else "jacobian_enrichment"
    display(Image(filename=f'{record["figure_prefix"]}_{suffix}.png', width=1100))


def main():
    trial_root()
    require_current_weight_results()
    audit = Audit()
    audit.track("step7/code/display_muscle.py")
    audit.track("step7/code/step7_weight_analysis.py")
    ranks, statistics = [], []
    for model in range(1, 21):
        tag = f"model_{model:02d}"
        record = audit.record(f"step3/quality_control/{tag}_kw.json")
        prefix = f"step3/statistics/{tag}"
        rank = read_csv(audit.checked(f"{prefix}/tissue_median_ranks.csv", record["output_sha256"]))
        kw = read_csv(audit.checked(f"{prefix}/kw_results.csv", record["output_sha256"]))
        require(len(kw) == 64 and kw.model_number.eq(model).all() and kw.neuron_index.nunique() == 64,
                f"{tag} KW神经元范围不完整。")
        require((kw.candidate == (kw.q_value_bh.lt(.05) & kw.epsilon2.gt(.14))).all(), "候选规则与Step3不符。")
        require(rank.model_number.eq(model).all() and len(rank) == 64 * 54 and
                rank.groupby("unit_id").tissue_name.nunique().eq(54).all(), "组织排序不完整。")
        # 用Step2组织中位数交叉核对，并以其保存精度排序，避免Step3 CSV显示舍入引入并列。
        med = read_csv(audit.checked(f"step2/tissue_medians/{tag}/{LAYER}.csv", record["configuration"]["source_sha256"]))
        exact = med.melt(id_vars=["tissue_name", "n_samples"], var_name="unit_id", value_name="exact_median")
        rank = rank.merge(exact, on=["tissue_name", "n_samples", "unit_id"], validate="one_to_one")
        require(len(rank) == 64 * 54 and np.allclose(rank.median_activation, rank.exact_median, rtol=2e-7, atol=2e-7),
                "Step2与Step3组织中位数不一致。")
        rank["median_activation"] = rank.pop("exact_median")
        ranks.append(rank)
        statistics.append(kw)
    ranks, statistics = pd.concat(ranks, ignore_index=True), pd.concat(statistics, ignore_index=True)
    winner, candidates = select_object(ranks, statistics)
    model, neuron, unit = int(winner.model_number), int(winner.neuron_index), winner.unit_id
    tag = f"model_{model:02d}"
    print(f"Fixed display {tag} / {unit}; {int(candidates.eligible.sum())} eligible neurons audited; muscle median={winner.median_activation:.9g}", flush=True)

    # 已完成分析的主记录只读；不改写其展示常量或指纹。
    finals = {5: audit.record("step5/quality_control/finalize.json"),
              6: audit.record("step6/quality_control/run.json"),
              7: audit.record("step7/quality_control/finalize.json")}
    audit.checked("step5/quality_control/finalize.json", finals[6]["configuration"]["source_sha256"])
    audit.checked("step6/quality_control/run.json", finals[7]["source_sha256"])
    refs = {step: audit.record(f"step{step}/quality_control/reference.json") for step in (5, 7)}
    for step in (5, 7):
        audit.checked(f"step{step}/quality_control/reference.json", finals[step]["source_sha256"])
        for name in ("gene_map.csv", "pathways.json", "tissue_map.csv", "analysis_design.json"):
            audit.checked(f"step{step}/reference/{name}", refs[step]["output_sha256"])
    for name in ("gene_map.csv", "pathways.json", "tissue_map.csv"):
        require(digest(f"step5/reference/{name}") == digest(f"step7/reference/{name}"), "两路线基因/组织/参考集合不同。")
    design = read_json("step7/reference/analysis_design.json")
    require(design["mode"] == "eval" and design["entry_formula"] == "10000 * decoder.0.weight.T" and
            design["propagation_entry"] == "decoder[1:]" and
            design["bn_statistics"] == "trained running statistics; decoder.1 and decoder.5 each called once" and
            design["reference_selection"]["id"] == "tissue_top100_mean_tpm_v1", "权重路线不是当前批准配置。")
    tissue_map = read_csv("step5/reference/tissue_map.csv")
    medians = ranks.loc[ranks.model_number.eq(model) & ranks.unit_id.eq(unit)].merge(tissue_map,
        on=["tissue_name", "n_samples"], validate="one_to_one").rename(columns={"tissue_id": "source_tissue_id", "tissue_name": "source_tissue"})
    require(len(medians) == 54, "展示组织映射不完整。")
    medians["neuron_index"] = neuron
    selection = high_low(medians)
    pathway = read_json("step5/reference/pathways.json")
    selection["reference_available"] = selection.source_tissue_id.isin(pathway)
    # 基因符号按字面读取，不能将名为NA的字符串解释成缺失值。
    genes = pd.read_csv("step5/reference/gene_map.csv", keep_default_na=False)
    require(len(genes) == 58581 and genes.gene_name.nunique() == len(genes) and
            np.array_equal(genes.merged_column, np.arange(len(genes))), "基因列映射不正确。")
    frames, summaries, gsea = {}, [], []
    for step, route, gsea_step in ((6, "Jacobian", 5), (7, "Weight", 7)):
        pairs = focus(read_csv(audit.checked(f"step{step}/statistics/ngci_tissue_pairs.csv", finals[step]["output_sha256"])), model, neuron)
        if step == 7:
            pairs = pairs.rename(columns={"activation_tissue_id": "source_tissue_id", "activation_tissue": "source_tissue"})
        require(len(pairs) == 53 and pairs.source_tissue_id.nunique() == 53 and set(pairs.source_tissue_id) == set(pathway), "NGCI组织应为53个参考组织。")
        expected = pairs.source_tissue_id.map(medians.set_index("source_tissue_id").median_activation)
        require(np.allclose(pairs.activation_median, expected, rtol=2e-7, atol=2e-7), "NGCI激活值对应对象错误。")
        pairs["display_group"] = pairs.source_tissue_id.map(selection.set_index("source_tissue_id").group).fillna("other")
        result = focus(read_csv(audit.checked(f"step{step}/statistics/ngci_per_neuron.csv", finals[step]["output_sha256"])), model, neuron)
        require(len(result) == 1, "NGCI神经元主键不唯一。")
        r, p = pearsonr(pairs.activation_median, pairs.ES)
        require(np.isclose(r, result.iloc[0].pearson_r, atol=1e-12, rtol=1e-12) and
                np.isclose(p, result.iloc[0].pearson_p, rtol=1e-9, atol=1e-300), "保存的NGCI与53个配对不一致。")
        result["route"] = route
        frames[route] = pairs
        summaries.append(result)
        full = focus(read_csv(audit.checked(f"step{gsea_step}/statistics/matched_gsea.csv", finals[gsea_step]["output_sha256"])), model, neuron)
        if step == 7:
            full = full.rename(columns={"activation_tissue_id": "source_tissue_id", "activation_tissue": "source_tissue"})
        columns = ["source_tissue_id", "ES", "NES", "pval", "q_matched_global", "size"]
        joined = selection.merge(full[columns], on="source_tissue_id", how="left", validate="one_to_one")
        require(np.isfinite(joined.loc[joined.reference_available, "ES"]).all(), "展示组织的ES缺失。")
        joined["route"] = route
        gsea.append(joined)
    agreement = focus(read_csv(audit.checked("step7/statistics/rank_agreement_all.csv", finals[7]["output_sha256"])), model, neuron)
    require(len(agreement) == 54 and agreement.source_tissue_id.nunique() == 54, "基因排序比较应有54组织。")

    # 输出只在展示目录。完成标记最后写入；中途失败不能被Notebook视为完成。
    for sub in ("statistics", "rank_inputs", "figures", "quality_control"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    write_json(OUT / "quality_control/python.json", {"status": "running"})
    output_paths = []
    def csv(table, name):
        path = OUT / "statistics" / f"{name}.csv"
        table.to_csv(path, index=False, na_rep="NA")
        output_paths.append(path)
    csv(candidates, "selection_candidates")
    csv(selection, "display_selection")
    csv(medians, "tissue_medians")
    csv(focus(statistics, model, neuron), "kw_result")
    csv(pd.concat(summaries, ignore_index=True), "ngci_results")
    csv(pd.concat(gsea, ignore_index=True), "display_gsea")
    csv(agreement, "rank_agreement")
    for route, pairs in frames.items():
        csv(pairs, f"{route.lower()}_ngci_tissue_pairs")
    prefix = OUT / "figures" / f"{tag}_{unit}"
    selected_kw = focus(statistics, model, neuron).iloc[0]
    kw_record = audit.record(f"step3/quality_control/{tag}_kw.json")
    samples = read_csv(audit.checked(f"step2/raw_outputs/{tag}/sample_map.csv", kw_record["configuration"]["source_sha256"]))
    matrix_path = audit.checked(f"step2/raw_outputs/{tag}/{LAYER}.npy", kw_record["configuration"]["source_sha256"])
    matrix = np.load(matrix_path, mmap_mode="r")
    require(matrix.shape == (len(samples), 64) and samples.sample_id.is_unique and
            np.array_equal(samples.row_index, np.arange(len(samples))), "逐样本激活顺序错误。")
    boxplot(matrix[:, neuron], samples, medians, selected_kw, Path(f"{prefix}_activation_boxplot"))
    for route, result in zip(("Jacobian", "Weight"), summaries):
        scatter(frames[route], result.iloc[0], selection, route, Path(f"{prefix}_{route.lower()}_ngci"))
    for suffix in ("activation_boxplot", "jacobian_ngci", "weight_ngci"):
        output_paths.extend(Path(f"{prefix}_{suffix}.{ext}") for ext in ("png", "pdf"))

    # 按jobs的matrix_row读取选中神经元，不能把神经元编号或第一行当矩阵行号。
    score_rows, top_lists = [], []
    for route, tissue in [("Weight", None)] + [("Jacobian", x) for x in selection.source_tissue_id]:
        step = 7 if route == "Weight" else 5
        tail = tag if step == 7 else f"{tag}/{tissue}"
        prepared_path = f"step{step}/quality_control/prepared/{tail}.json"
        prepared = audit.record(prepared_path)
        fgsea_path = f"step{step}/quality_control/fgsea/{tail}.json"
        fgsea = audit.record(fgsea_path)
        audit.checked(fgsea_path, finals[step]["source_sha256"])
        audit.checked(prepared_path, fgsea["source_sha256"])
        audit.checked(f"step{step}/quality_control/reference.json", prepared["source_sha256"])
        folder = f"step{step}/rank_inputs/{tail}"
        jobs = read_csv(audit.checked(f"{folder}/jobs.csv", prepared["output_sha256"]))
        score_path = audit.checked(f"{folder}/scores.f64.bin", prepared["output_sha256"])
        job = focus(jobs, model, neuron)
        require(len(job) == 1 and job.iloc[0].unit_id == unit, "评分矩阵缺少唯一的展示神经元。")
        row = int(job.iloc[0].matrix_row)
        require(0 <= row < len(jobs) and Path(score_path).stat().st_size == len(jobs) * len(genes) * 8, "评分矩阵形状或行号错误。")
        scores = np.memmap(score_path, dtype="<f8", mode="r", shape=(len(jobs), len(genes)))[row].copy()
        require(np.isfinite(scores).all(), "基因评分含非有限值。")
        name = f"{route.lower()}_{tissue or 'all_tissues'}.f64.bin"
        target = OUT / "rank_inputs" / name
        scores.astype("<f8").tofile(target)
        output_paths.append(target)
        score_rows.append({"route": route, "source_tissue_id": tissue or "all_tissues", "path": target.as_posix(),
                           "model_number": model, "neuron_index": neuron, "unit_id": unit, "matrix_row": row, "n_genes": len(genes)})
        for direction, order in (("positive", np.argsort(-scores, kind="stable")), ("negative", np.argsort(scores, kind="stable"))):
            keep = order[scores[order] > 0] if direction == "positive" else order[scores[order] < 0]
            keep = keep[:200]
            top_lists.append(pd.DataFrame({"model_number": model, "unit_id": unit, "route": route,
                "source_tissue_id": tissue or "all_tissues", "direction": direction, "rank": np.arange(1, len(keep) + 1),
                "gene_name": genes.gene_name.iloc[keep].to_numpy(), "score": scores[keep]}))
    csv(pd.DataFrame(score_rows), "score_inputs")
    csv(pd.concat(top_lists, ignore_index=True), "display_top200")
    record = {"status": "complete", "model_number": model, "neuron_index": neuron, "unit_id": unit,
              "figure_prefix": prefix.as_posix(), "n_eligible": int(candidates.eligible.sum()),
              "selection_mode": "fixed_user_choice",
              "selection_rule": f"Fixed model_{DISPLAY_MODEL:02d} / {DISPLAY_UNIT}; validate Step3 candidate AND Muscle-Skeletal median equals within-neuron maximum",
              "selection_basis": "User-specified after exploratory comparison of the two routes in six high/low activation tissues",
              "selection_scope": "Fixed object; eligibility audited across 20 models x 64 neurons; full samples; 54 tissues",
              "tissue_selection_rule": "Highest three and lowest three activation medians among all 54 tissues; ties by tissue ID; no ES-based tissue selection or replacement",
              "interpretation": "Illustrative case chosen after method-agreement review; does not establish model-wide superiority; post-selection p-values are descriptive.",
              "ngci_scope": "53 reference tissues, equal weights, saved two-sided Pearson statistics",
              "source_sha256": audit.sources,
              "output_sha256": {p.as_posix(): digest(p) for p in output_paths}}
    write_json(OUT / "quality_control/python.json", record)
    print(f"Display complete: {OUT}. Run step7/code/03_plot_muscle_enrichment.R manually for enrichment curves.")


if __name__ == "__main__":
    main()

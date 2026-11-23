"""Step5：组织参考库、完整基因评分、GSEA 结果整理与展示。

从 trial 或其子目录调用本脚本。阶段：check、reference、prepare、finalize、self-test。
reference 只构建组织内平均 TPM Top100；prepare 同时准备参考库和完整评分矩阵。
prepare 为独立 R 分析准备小端 float64 矩阵；不会启动 R 或安装软件。
finalize 仅在全部20模型的 R 结果可核验时，执行同组织主要检验的统一BH。
不同模型的神经元不对齐、不平均。所有路径由工作目录推导，无机器专属路径。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats

sys.dont_write_bytecode = True
TRIAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("step5")
CODE = ROOT / "code/step5_gsea.py"
ATLAS = Path("gene_atlas/GTEx_Tissues_V8_2023.txt")
RAW_EXPRESSION = Path("dataset/gtex_RSEM_gene_tpm")
PHENOTYPE = Path("dataset/GTEX_phenotype")
SAMPLE_MANIFEST = Path("step1/core_metadata/sample_manifest.csv")
FULL_ORDER = Path("step1/core_metadata/sample_order_full.csv")
GENE_ORDER = Path("step1/core_metadata/gene_order.csv")
TRAINING_CONFIG = Path("step1/core_metadata/experiment_config.json")
N_MODELS, N_TISSUES, N_GENES, N_UNITS = 20, 54, 60498, 64
LOGGER = logging.getLogger("step5")

# 绘图对象提前固定，不依据NES或显著性择优。修改这里后需明确重建prepare。
DISPLAY_MODEL = 1
DISPLAY_UNITS = (0,)
PARAMETERS = {"minSize": 15, "maxSize": 500, "eps": 0, "scoreType": "std",
              "gseaParam": 1, "sampleSize": 101, "nPermSimple": 1000}
REFERENCE_SELECTION = {
    "id": "tissue_top100_mean_tpm_v1", "target_size": 100,
    "candidate_pool": "exact-name-matched union of same-tissue atlas strata",
    "sample_scope": "all expression samples of the same tissue; equal sample weights",
    "input_scale": "log2(TPM + 0.001)",
    "inverse_transform": "maximum(exp2(stored_value) - 0.001, 0)",
    "expression_statistic": "arithmetic mean on linear TPM, not mean log expression",
    "duplicate_gene_names": "arithmetic mean across features on linear TPM, then across samples",
    "precision": "float64 from raw stored values; no float16 training-matrix roundtrip",
    "selection_order": "descending mean TPM; ties by UTF-8 gene-name bytes",
    "shortfall": "keep all matched candidates when fewer than 100; no outside-pool substitution",
    "model_dependence": "none; one shared reference for both routes and all models",
}
DESIGN = {
    "method": "fgseaMultilevel", "parameters": PARAMETERS, "R_required": "4.6.1",
    "fgsea_version_policy": "freeze installed compatible version before first analysis; reject any later mismatch",
    "gene_reference": "53 tissue Top100 sets selected by mean TPM; retain 511 source sets and candidate unions",
    "reference_selection": REFERENCE_SELECTION,
    "gene_matching": "exact case-sensitive names; no alias update",
    "gene_score": "arithmetic mean across gene features sharing an exact name, using Step4 signed mean",
    "ranking_universe": "all 58581 unique names; not restricted to reference members",
    "rank_ties": "descending score; stable existing UTF-8 byte order for equal scores; no jitter",
    "primary_family": "all same-tissue tests across all 20 models, all candidate neurons, all 53 matched tissues",
    "primary_significance": "q_matched_global < 0.05; no additional NES cutoff",
    "missing_pvalues": "keep NA; include planned estimable-reference tests in BH family size",
    "no_reference": "CML same-tissue result remains NA and is outside the primary testing family",
    "cross_tissue_results": "exploratory; retain fgsea within-ranking BH, not a global primary q-value",
    "stability": "tissue detection frequency and per-model significant fractions, without neuron correspondence",
    "model_direction": "latent signs can change between models; positive/negative counts are descriptive",
    "seed_formula": "629 + 5000000 + (model-1)*54*64 + (tissue_number-1)*64 + neuron_index",
    "display_model": DISPLAY_MODEL, "display_neurons": list(DISPLAY_UNITS),
    "display_tissues": "top and bottom 3 by tissue median activation; ties by tissue name",
    "top200": "only display neurons and selected source tissues; signed lists after name merging; no ORA",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_record(path: Path, *, sources: bool = False) -> dict:
    """只认完整记录和实际文件指纹；sources=True 还检查本次分析所依赖的文件。"""
    record = read_json(path)
    require(record.get("status") == "complete", f"{path} 尚未完成。")
    require(bool(record.get("output_sha256")), f"{path} 没有输出指纹。")
    for file, expected in record.get("output_sha256", {}).items():
        require(Path(file).is_file() and sha(Path(file)) == expected, f"{file} 缺失或指纹不匹配。")
    if sources:
        for file, expected in record.get("source_sha256", {}).items():
            require(Path(file).is_file() and sha(Path(file)) == expected, f"来源已变化：{file}")
    return record


def begin(path: Path, config: dict, outputs: list[Path], overwrite: bool) -> bool:
    """完成项可续跑；部分输出或配置变化必须明确使用overwrite。"""
    if not overwrite and path.exists():
        record = verify_record(path)
        require(record.get("configuration") == config, f"{path} 配置已变化，请明确重算。")
        require(set(record["output_sha256"]) == {p.as_posix() for p in outputs}, "完成记录的输出列表不匹配。")
        LOGGER.info("Skip completed: %s", path)
        return False
    require(overwrite or not any(p.exists() for p in outputs), "存在未完成输出，请明确使用 --overwrite。")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"status": "running", "configuration": config})
    return True


def finish(path: Path, config: dict, outputs: list[Path], **details) -> None:
    write_json(path, {"status": "complete", "configuration": config,
                     "output_sha256": {p.as_posix(): sha(p) for p in outputs}, **details})


@dataclass
class ModelInputs:
    number: int
    genes: pd.DataFrame
    neurons: pd.DataFrame
    tissues: pd.DataFrame
    model_record: dict
    sources: dict

    @property
    def tag(self) -> str:
        return f"model_{self.number:02d}"


def load_model(number: int) -> ModelInputs:
    """仅读取Step4映射与状态，不加载模型权重或其他五种梯度统计矩阵。"""
    tag = f"model_{number:02d}"
    model_path = Path(f"step4/quality_control/{tag}/model.json")
    model = read_json(model_path)
    cfg = model["configuration"]
    require(model["status"] == "complete" and cfg["model_number"] == number
            and cfg["model_seed"] == 629 + number and cfg["final_epoch"] == 100
            and cfg["sample_scope"] == "full" and cfg["n_samples"] == 7862,
            f"{tag}: Step4模型记录不匹配。")
    mappings_path = Path(f"step4/quality_control/{tag}/mappings.json")
    maps = verify_record(mappings_path)
    require(maps["configuration"] == cfg, "Step4模型和映射的配置不一致。")
    frames = [pd.read_csv(f"step4/mappings/{tag}/{name}_map.csv", keep_default_na=False)
              for name in ("gene", "neuron", "tissue")]
    genes, neurons, tissues = frames
    require(len(genes) == N_GENES and genes.gene_id.is_unique
            and np.array_equal(genes.matrix_column, np.arange(N_GENES)), "基因映射顺序错误。")
    require(genes.gene_name.str.strip().ne("").all(), "不允许空基因名。")
    require(np.array_equal(neurons.matrix_row, np.arange(len(neurons)))
            and neurons.neuron_index.tolist() == cfg["candidate_neuron_indices"], "候选行映射错误。")
    require(neurons.neuron_index.between(0, 63).all() and neurons.neuron_index.is_unique, "候选编号错误。")
    require(len(tissues) == 54 and tissues.tissue_id.is_unique and tissues.tissue_name.is_unique
            and tissues.n_samples.sum() == 7862, "组织映射错误。")
    sources = {model_path.as_posix(): sha(model_path), mappings_path.as_posix(): sha(mappings_path),
               **maps["output_sha256"]}
    return ModelInputs(number, genes, neurons, tissues, model, sources)


def checked_mean(model: ModelInputs, tissue) -> tuple[np.ndarray, dict]:
    """重新核对每份平均梯度的指纹，避免只凭文件存在继续分析。"""
    path = Path(f"step4/summaries/{model.tag}/{tissue.tissue_id}/mean.npy")
    report = Path(f"step4/quality_control/{model.tag}/{tissue.tissue_id}.json")
    record = read_json(report)
    require(record["status"] == "complete" and record["n_samples"] == int(tissue.n_samples)
            and record["configuration"] == {**model.model_record["configuration"],
                "tissue_id": tissue.tissue_id, "tissue_name": tissue.tissue_name,
                "n_tissue_samples": int(tissue.n_samples)}, "组织梯度配置错误。")
    digest = sha(path)
    require(digest == record["output_sha256"][path.as_posix()], f"{path} 指纹不一致。")
    matrix = np.load(path, allow_pickle=False)
    require(matrix.shape == (len(model.neurons), N_GENES) and matrix.dtype == np.float32
            and np.isfinite(matrix).all(), "平均梯度矩阵形状或数值错误。")
    return matrix, {path.as_posix(): digest, report.as_posix(): sha(report)}


def make_gene_map(genes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UTF-8字节序作为固定基因顺序；它同时决定完全并列评分的次序。"""
    names = sorted(set(genes.gene_name), key=lambda x: x.encode("utf-8"))
    indices = {name: i for i, name in enumerate(names)}
    mapping = genes.copy()
    mapping["merged_column"] = mapping.gene_name.map(indices)
    counts = mapping.groupby("merged_column").size().reindex(range(len(names))).to_numpy()
    merged = pd.DataFrame({"merged_column": np.arange(len(names)), "gene_name": names, "n_features": counts})
    return merged, mapping


def merge_scores(matrix: np.ndarray, feature_map: pd.DataFrame, counts: np.ndarray) -> np.ndarray:
    """按同名特征作float64算术平均；不平均SD/SEM/中位数等其他统计量。"""
    code = feature_map.merged_column.to_numpy(dtype=np.int64)
    order = np.argsort(code, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    result = np.add.reduceat(matrix[:, order].astype(np.float64), starts, axis=1) / counts
    require(result.shape == (len(matrix), len(counts)) and np.isfinite(result).all(), "合并评分无效。")
    return result


def build_reference(genes: pd.DataFrame, tissues: pd.DataFrame) -> dict:
    """先建立53个组织候选并集；这里尚未按表达量筛选，不可直接供GSEA使用。"""
    names = set(genes.gene_name)
    lookup = {re.sub(r"\s+", "", t.tissue_name).casefold(): t for t in tissues.itertuples(index=False)}
    sources, memberships, unions, original_names = [], [], {}, {}
    all_atlas_genes = set()
    for line_number, line in enumerate(ATLAS.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        match = re.fullmatch(r"(.*?) (Male|Female) (\d{2}-\d{2}) Up", parts[0])
        require(match is not None and len(parts) >= 3, f"参考库第{line_number}行格式不匹配。")
        original, sex, age = match.groups()
        normalized = re.sub(r"\s+", "", original).casefold()
        rename = normalized == "cells-culturedfibroblasts"
        if rename:
            normalized = "cells-transformedfibroblasts"
        require(normalized in lookup, f"参考组织无显式对应：{original}")
        tissue = lookup[normalized]
        members = sorted({g.strip() for g in parts[2:] if g.strip()}, key=lambda x: x.encode("utf-8"))
        require(all(";" not in gene for gene in members), "基因名不能含leading-edge分隔符。")
        source_id = f"source_{line_number:04d}"
        sources.append({"source_id": source_id, "source_name": parts[0], "reference_tissue_id": tissue.tissue_id,
                        "reference_tissue": tissue.tissue_name, "atlas_tissue": original, "sex": sex, "age": age,
                        "direction": "Up", "n_genes": len(members), "genes": members,
                        "name_rule": "GTEx tissue rename confirmed by user" if rename else "spaces and case normalization"})
        for gene in members:
            memberships.append({"source_id": source_id, "reference_tissue_id": tissue.tissue_id,
                                "gene_name": gene, "matched": gene in names})
        unions.setdefault(tissue.tissue_id, set()).update(members)
        original_names[tissue.tissue_id] = original
        all_atlas_genes.update(members)
    paths, sets, union_members = {}, [], []
    for tissue in tissues.itertuples(index=False):
        if tissue.tissue_id not in unions:
            continue
        union = sorted(unions[tissue.tissue_id], key=lambda x: x.encode("utf-8"))
        matched = [gene for gene in union if gene in names]
        require(15 <= len(matched) <= 500, f"{tissue.tissue_name} 集合大小不在15–500。")
        paths[tissue.tissue_id] = matched
        sets.append({"reference_tissue_id": tissue.tissue_id, "reference_tissue": tissue.tissue_name,
                     "atlas_tissue": original_names[tissue.tissue_id], "n_union": len(union),
                     "n_matched": len(matched), "n_samples": int(tissue.n_samples),
                     "n_sources": sum(s["reference_tissue_id"] == tissue.tissue_id for s in sources)})
        union_members.extend({"reference_tissue_id": tissue.tissue_id, "gene_name": gene,
                              "matched": gene in names} for gene in union)
    require(len(sources) == 511 and len(sets) == 53, "参考集合数量与固定输入不一致。")
    require(len({s["source_name"] for s in sources}) == 511, "来源集合名称重复。")
    missing = tissues.loc[~tissues.tissue_id.isin(paths), "tissue_name"].tolist()
    require(missing == ["Cells-Leukemiacellline(CML)"], "缺失参考组织与预定情况不一致。")
    unmatched = sorted(all_atlas_genes - names, key=lambda x: x.encode("utf-8"))
    require(len(unmatched) == 190, "未匹配基因数量发生变化，请核对输入。")
    return {"source_sets": sources, "provenance": pd.DataFrame(memberships),
            "reference_sets": pd.DataFrame(sets), "union_members": pd.DataFrame(union_members),
            "pathways": paths, "unmatched": pd.DataFrame({"gene_name": unmatched}), "missing_tissues": missing}


def align_reference_samples(sample_ids: list[str], manifest: pd.DataFrame,
                            phenotype: pd.DataFrame, tissues: pd.DataFrame) -> np.ndarray:
    """用样本ID连接正式标签；即使注释表行序改变也不按行号猜测组织。"""
    require(len(set(sample_ids)) == len(sample_ids) and manifest.Sample.is_unique
            and phenotype.Sample.is_unique, "表达或注释中的样本ID重复。")
    require(set(manifest.Sample) == set(sample_ids) and set(sample_ids) <= set(phenotype.Sample),
            "表达样本与完整样本清单或phenotype不一致。")
    aligned = manifest.set_index("Sample").loc[sample_ids]
    official = phenotype.set_index("Sample").loc[sample_ids, "body_site_detail (SMTSD)"]
    # 与Step1一致：只删除空格，不更新组织别名。
    normalized = official.str.strip().str.replace(" - ", "-", regex=False).str.replace(" ", "", regex=False)
    require(aligned.tissue_name.equals(normalized.rename("tissue_name")), "组织标签与正式phenotype不一致。")
    require(np.array_equal(aligned.expression_column_position, np.arange(len(sample_ids))),
            "冻结的表达列位置与原始样本顺序不一致。")
    codes = pd.Index(tissues.tissue_name).get_indexer(aligned.tissue_name)
    require((codes >= 0).all() and np.array_equal(np.bincount(codes, minlength=len(tissues)), tissues.n_samples),
            "组织样本数与Step4完整样本映射不一致。")
    return codes


def scan_candidate_mean_tpm(path: Path, genes: pd.DataFrame, codes: np.ndarray,
                            tissue_counts: np.ndarray, candidates: list[str]) -> tuple[np.ndarray, dict]:
    """逐行读原始表达文件，仅解析候选基因的数值，避免另存大型TPM矩阵。

    输入是log2(TPM+0.001)，必须逐数值逆变换后才求均值。组织均值和
    同名特征均值在线性尺度上可交换，以下先汇总每个特征，再除以同名特征数。
    浮点累加均为float64；不使用log2(TPM+1)的float16训练缓存。
    """
    lookup = {name: i for i, name in enumerate(candidates)}
    totals = np.zeros((len(candidates), len(tissue_counts)), dtype=np.float64)
    feature_counts = np.zeros(len(candidates), dtype=np.int64)
    digest = hashlib.sha256()
    before = path.stat()
    n_rows, n_clipped = 0, 0
    with path.open("rb") as stream:
        header = stream.readline()
        digest.update(header)
        require(len(header.rstrip(b"\r\n").split(b"\t")) == len(codes) + 1, "表达表列数不匹配。")
        for position, line in enumerate(stream):
            digest.update(line)
            require(position < len(genes), "表达表含多余基因行。")
            fields = line.rstrip(b"\r\n").split(b"\t", 1)
            require(len(fields) == 2 and fields[0].decode("utf-8") == genes.gene_id.iloc[position],
                    f"表达基因ID与冻结映射不一致：第{position + 1}行。")
            require(fields[1].count(b"\t") + 1 == len(codes), f"表达数据列数错误：第{position + 1}行。")
            name = genes.gene_name.iloc[position]
            if name in lookup:
                stored = np.fromstring(fields[1].decode("ascii"), dtype=np.float64, sep="\t")
                require(len(stored) == len(codes) and np.isfinite(stored).all(), "候选基因表达含无效值。")
                tpm = np.exp2(stored) - 0.001
                require(np.isfinite(tpm).all(), "TPM逆变换溢出。")
                n_clipped += int((tpm < 0).sum())
                np.maximum(tpm, 0, out=tpm)
                index = lookup[name]
                totals[index] += np.bincount(codes, weights=tpm, minlength=len(tissue_counts)) / tissue_counts
                feature_counts[index] += 1
            n_rows += 1
            if n_rows % 8192 == 0:
                LOGGER.info("Top100 expression scan: %d / %d gene features", n_rows, len(genes))
    after = path.stat()
    require(n_rows == len(genes) and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "表达文件不完整或在读取期间发生改变。")
    expected_counts = genes.gene_name.value_counts().reindex(candidates).to_numpy()
    require((feature_counts > 0).all() and np.array_equal(feature_counts, expected_counts), "候选同名特征计数错误。")
    means = totals / feature_counts[:, None]
    require(np.isfinite(means).all() and (means >= 0).all(), "平均TPM无效。")
    return means, {"expression_sha256": digest.hexdigest(), "n_expression_features": n_rows,
                   "n_expression_samples": len(codes), "n_unique_candidates": len(candidates),
                   "n_candidate_values_clipped_to_zero": n_clipped}


def rank_top100(members: list[str], mean_tpm: pd.Series) -> pd.DataFrame:
    """排名只依赖预定候选的平均TPM；并列时采用UTF-8字节序，缺额不补齐。"""
    require(len(members) == len(set(members)) and mean_tpm.index.is_unique, "候选或表达名称重复。")
    names = sorted(members, key=lambda name: name.encode("utf-8"))
    values = mean_tpm.loc[names].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all() and (values >= 0).all(), "候选平均TPM含缺失或无效值。")
    order = np.argsort(-values, kind="stable")
    ranks = np.arange(1, len(names) + 1)
    return pd.DataFrame({"gene_name": np.asarray(names)[order], "mean_tpm": values[order],
                         "expression_rank": ranks, "selected": ranks <= 100})


def select_reference_top100(model: ModelInputs, reference: dict) -> dict:
    """在候选并集内选Top100，保留全部候选的表达排名与未匹配名称。"""
    run = read_json(TRAINING_CONFIG)["run"]
    require(run["stored_input_scale"] == REFERENCE_SELECTION["input_scale"]
            and run["final_training_scale"] == "log2(TPM + 1)", "Step1尺度记录与TPM恢复规则不一致。")
    inputs = [PHENOTYPE, SAMPLE_MANIFEST, FULL_ORDER, GENE_ORDER, TRAINING_CONFIG]
    sources = {path.as_posix(): sha(path) for path in inputs}
    frozen_genes = pd.read_csv(GENE_ORDER, keep_default_na=False)
    pd.testing.assert_frame_equal(frozen_genes, model.genes[["expression_row_position", "gene_id", "gene_name"]])
    with RAW_EXPRESSION.open("r", encoding="utf-8") as stream:
        sample_ids = stream.readline().rstrip("\r\n").split("\t")[1:]
    full = pd.read_csv(FULL_ORDER, keep_default_na=False)
    require(full.sample_id.tolist() == sample_ids and np.array_equal(full.order, np.arange(len(sample_ids))),
            "表达样本顺序与冻结的完整样本顺序不一致。")
    manifest = pd.read_csv(SAMPLE_MANIFEST, keep_default_na=False)
    phenotype = pd.read_csv(PHENOTYPE, sep="\t", keep_default_na=False)
    codes = align_reference_samples(sample_ids, manifest, phenotype, model.tissues)
    candidates = sorted({g for members in reference["pathways"].values() for g in members}, key=lambda x: x.encode("utf-8"))
    means, audit = scan_candidate_mean_tpm(RAW_EXPRESSION, model.genes, codes, model.tissues.n_samples.to_numpy(), candidates)
    sources[RAW_EXPRESSION.as_posix()] = audit["expression_sha256"]
    require(all(sha(path) == sources[path.as_posix()] for path in inputs), "表达注释在参考库计算期间发生改变。")
    expression = pd.DataFrame(means, index=candidates, columns=model.tissues.tissue_id)
    feature_counts = model.genes.gene_name.value_counts()
    sets = reference["reference_sets"].copy()
    sets["n_candidates_matched"] = sets.n_matched
    tables, pathways = [], {}
    for row in sets.itertuples(index=False):
        ranked = rank_top100(reference["pathways"][row.reference_tissue_id], expression[row.reference_tissue_id])
        ranked.insert(0, "reference_tissue_id", row.reference_tissue_id)
        ranked.insert(1, "reference_tissue", row.reference_tissue)
        ranked["n_samples"], ranked["n_features"] = row.n_samples, ranked.gene_name.map(feature_counts)
        tables.append(ranked)
        # GMT/JSON成员顺序不参与GSEA；固定字节序，表达排名另存完整表。
        pathways[row.reference_tissue_id] = sorted(ranked.loc[ranked.selected, "gene_name"], key=lambda x: x.encode("utf-8"))
    sets["n_matched"] = sets.reference_tissue_id.map(lambda tid: len(pathways[tid]))
    sets["n_selected"] = sets.n_matched
    sets["shortfall"] = 100 - sets.n_selected
    sets["selection_status"] = np.where(sets.shortfall.eq(0), "complete_top100", "insufficient_matched_candidates")
    require(sets.n_selected.eq(np.minimum(sets.n_candidates_matched, 100)).all(), "Top100数量不符合候选范围。")
    reference["reference_sets"], reference["pathways"] = sets, pathways
    reference["candidate_expression"] = pd.concat(tables, ignore_index=True)
    reference["union_members"]["selected"] = [gene in pathways[tid] for tid, gene in
        reference["union_members"][["reference_tissue_id", "gene_name"]].itertuples(index=False, name=None)]
    reference["selection_metadata"] = {"specification": REFERENCE_SELECTION, "source_sha256": sources, **audit,
        "n_reference_sets": len(sets), "n_selected_members": int(sets.n_selected.sum()),
        "shortfalls": sets.loc[sets.shortfall.gt(0), ["reference_tissue_id", "reference_tissue", "n_selected", "shortfall"]].to_dict("records")}
    return reference


def display_selection(model: ModelInputs) -> tuple[pd.DataFrame, dict]:
    """按激活中位数预先固定六个组织；缺少参考集合时保留该面板，不换组织。"""
    path = Path(f"step3/statistics/{model.tag}/tissue_median_ranks.csv")
    qc_path = Path(f"step3/quality_control/{model.tag}_kw.json")
    qc = read_json(qc_path)
    require(qc["status"] == "complete" and qc["output_sha256"][path.as_posix()] == sha(path), "激活排序指纹不一致。")
    medians = pd.read_csv(path, float_precision="round_trip")
    selected = []
    for unit in DISPLAY_UNITS:
        require(unit in model.neurons.neuron_index.to_list(), "展示神经元必须是候选。")
        part = medians.loc[medians.unit_id.eq(f"unit_{unit:05d}")]
        require(len(part) == 54 and part.tissue_name.is_unique, "激活排序必须包含54组织。")
        for group, ascending in (("high", False), ("low", True)):
            ordered = part.sort_values(["median_activation", "tissue_name"], ascending=[ascending, True]).head(3)
            for rank, row in enumerate(ordered.itertuples(index=False), 1):
                t = model.tissues.set_index("tissue_name").loc[row.tissue_name]
                selected.append({"model_number": DISPLAY_MODEL, "neuron_index": unit, "unit_id": row.unit_id,
                                 "group": group, "rank_in_group": rank, "source_tissue_id": t.tissue_id,
                                 "source_tissue": row.tissue_name, "median_activation": row.median_activation})
    return pd.DataFrame(selected), {path.as_posix(): sha(path), qc_path.as_posix(): sha(qc_path)}


def seed_for(model: int, tissue_number: int, unit: int) -> int:
    """注入式任务编号保证全部69120任务种子不重复；不依赖执行顺序。"""
    return 629 + 5_000_000 + (model - 1) * 54 * 64 + (tissue_number - 1) * 64 + unit


def write_reference(root: Path, model: ModelInputs, reference: dict, overwrite: bool) -> None:
    folder = root / "reference"
    merged, feature_map = make_gene_map(model.genes)
    selected, display_sources = display_selection(model)
    outputs = [folder / name for name in ("source_sets.json", "source_members.csv", "reference_sets.csv",
                "union_members.csv", "pathways.json", "unmatched_genes.csv", "gene_map.csv", "feature_to_gene.csv",
                "tissue_map.csv", "display_selection.csv", "analysis_design.json", "top100_candidates.csv", "reference_selection.json")]
    require(reference["selection_metadata"]["specification"] == REFERENCE_SELECTION, "参考库尚未按平均TPM选择Top100。")
    expression_sources = reference["selection_metadata"]["source_sha256"]
    config = {"atlas_sha256": sha(ATLAS), "mapping_sha256": model.sources, "display_sources": display_sources,
              "expression_sources": expression_sources, "script_sha256": sha(CODE), "design": DESIGN}
    report = root / "quality_control/reference.json"
    if not begin(report, config, outputs, overwrite):
        return
    write_json(outputs[0], {"sets": reference["source_sets"]})
    reference["provenance"].to_csv(outputs[1], index=False)
    reference["reference_sets"].to_csv(outputs[2], index=False)
    reference["union_members"].to_csv(outputs[3], index=False)
    write_json(outputs[4], reference["pathways"])
    reference["unmatched"].to_csv(outputs[5], index=False)
    merged.to_csv(outputs[6], index=False)
    feature_map.to_csv(outputs[7], index=False)
    model.tissues.to_csv(outputs[8], index=False)
    selected.to_csv(outputs[9], index=False, float_format="%.17g")
    write_json(outputs[10], DESIGN)
    reference["candidate_expression"].to_csv(outputs[11], index=False, float_format="%.17g")
    write_json(outputs[12], reference["selection_metadata"])
    finish(report, config, outputs, source_sha256={ATLAS.as_posix(): sha(ATLAS), **model.sources,
           **display_sources, **expression_sources, CODE.as_posix(): sha(CODE)},
           n_reference_sets=53, n_source_sets=511, n_unique_genes=len(merged), n_unmatched_genes=190,
           reference_selection=REFERENCE_SELECTION, n_selected_members=reference["selection_metadata"]["n_selected_members"])


def prepare_tissue(root: Path, model: ModelInputs, tissue, merged: pd.DataFrame,
                   feature_map: pd.DataFrame, overwrite: bool) -> None:
    matrix, mean_sources = checked_mean(model, tissue)
    report = root / f"quality_control/prepared/{model.tag}/{tissue.tissue_id}.json"
    folder = root / f"rank_inputs/{model.tag}/{tissue.tissue_id}"
    outputs = [folder / "scores.f64.bin", folder / "jobs.csv", folder / "score_qc.csv"]
    reference_record = root / "quality_control/reference.json"
    source_hashes = {**model.sources, **mean_sources, reference_record.as_posix(): sha(reference_record), CODE.as_posix(): sha(CODE)}
    config = {"model_number": model.number, "source_tissue_id": tissue.tissue_id,
              "source_tissue": tissue.tissue_name, "n_candidates": len(model.neurons), "n_genes": len(merged),
              "binary_dtype": "little-endian float64", "binary_order": "row-major: candidate x gene",
              "source_sha256": source_hashes}
    if not begin(report, config, outputs, overwrite):
        return
    values = merge_scores(matrix, feature_map, merged.n_features.to_numpy())
    values.astype("<f8", copy=False).tofile(outputs[0])
    jobs = model.neurons[["matrix_row", "neuron_index", "unit_id"]].copy()
    jobs.insert(0, "model_number", model.number)
    jobs["source_tissue_id"], jobs["source_tissue"] = tissue.tissue_id, tissue.tissue_name
    jobs["seed"] = [seed_for(model.number, int(tissue.tissue_id[-2:]), int(u)) for u in jobs.neuron_index]
    jobs.to_csv(outputs[1], index=False)
    qc = pd.DataFrame({"unit_id": jobs.unit_id, "n_genes": len(merged),
                       "n_positive": (values > 0).sum(1), "n_negative": (values < 0).sum(1),
                       "n_zero": (values == 0).sum(1),
                       "n_tied_extra": [len(row) - len(np.unique(row)) for row in values],
                       "minimum": values.min(1), "maximum": values.max(1)})
    qc.to_csv(outputs[2], index=False, float_format="%.17g")
    require(outputs[0].stat().st_size == values.size * 8, "二进制矩阵文件长度错误。")
    finish(report, config, outputs, source_sha256=source_hashes)
    LOGGER.info("Prepared %s %s: %d x %d", model.tag, tissue.tissue_id, *values.shape)


def bh_planned(pvalues: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """BH族含全部已计划的同组织检验；缺失p保留NA，但不缩小检验族分母。

    对有效p排序后使用计划检验数m，而非仅有效p数。结构性缺失参考集合不
    属于检验族。此处理相当于校正计算中保守保留未估计检验，不把NA报告为1。
    """
    result = np.full(len(pvalues), np.nan)
    valid = eligible & np.isfinite(pvalues)
    require(np.all((pvalues[valid] >= 0) & (pvalues[valid] <= 1)), "p值不在[0,1]。")
    index = np.flatnonzero(valid)
    ordered = index[np.argsort(pvalues[index], kind="stable")]
    if len(ordered):
        raw = pvalues[ordered] * int(eligible.sum()) / np.arange(1, len(ordered) + 1)
        result[ordered] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return result


def stability_tables(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只按模型和组织汇总，不跨模型匹配unit_id，也不把20模型当独立供体。"""
    records = []
    for (model, tissue_id, tissue), rows in matched.groupby(["model_number", "source_tissue_id", "source_tissue"], sort=True):
        available = bool(rows.reference_available.all())
        valid = rows.q_matched_global.notna()
        n_sig = int(rows.significant.sum())
        complete = bool(valid.all()) if available else False
        detected = True if n_sig else (False if complete else None)
        records.append({"model_number": model, "source_tissue_id": tissue_id, "source_tissue": tissue,
                        "reference_available": available, "n_candidates": len(rows), "n_valid_p": int(valid.sum()),
                        "n_significant": n_sig if available else np.nan,
                        "n_positive_significant": int((rows.significant & (rows.NES > 0)).sum()) if available else np.nan,
                        "n_negative_significant": int((rows.significant & (rows.NES < 0)).sum()) if available else np.nan,
                        "significant_fraction_planned": n_sig / len(rows) if available else np.nan,
                        "pvalue_coverage": float(valid.mean()) if available else np.nan,
                        "has_significant": detected, "all_pvalues_available": complete})
    per_model = pd.DataFrame(records)
    across = []
    for (tid, tissue), rows in per_model.groupby(["source_tissue_id", "source_tissue"], sort=True):
        available = bool(rows.reference_available.all())
        assessable = rows.has_significant.notna()
        complete = bool(rows.all_pvalues_available.all())
        fractions = rows.significant_fraction_planned
        detected = int(rows.has_significant.eq(True).sum())
        across.append({"source_tissue_id": tid, "source_tissue": tissue, "reference_available": available,
                       "n_models": len(rows), "n_models_detected": detected if available else np.nan,
                       "n_models_detection_assessable": int(assessable.sum()) if available else np.nan,
                       "detection_frequency": detected / len(rows) if available and bool(assessable.all()) else np.nan,
                       "detection_frequency_lower_bound": detected / len(rows) if available else np.nan,
                       "median_significant_fraction": float(fractions.median()) if available and complete else np.nan,
                       "q25_significant_fraction": float(fractions.quantile(.25)) if available and complete else np.nan,
                       "q75_significant_fraction": float(fractions.quantile(.75)) if available and complete else np.nan,
                       "all_pvalues_available": complete,
                       "interpretation": "training-run sensitivity; no neuron alignment; no independent-cohort inference"})
    return per_model, pd.DataFrame(across)


def plot_heatmap(matched: pd.DataFrame, tissues: pd.DataFrame, path: Path, title_prefix: str = "") -> None:
    """全部64神经元同组织NES；缺失参考或数值为灰色，不填0。"""
    part = matched.loc[matched.model_number.eq(DISPLAY_MODEL)]
    values = part.pivot(index="source_tissue_id", columns="neuron_index", values="NES").reindex(
        index=tissues.tissue_id, columns=range(64)).to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    bound = float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(21, 15))
    fig.subplots_adjust(left=.29, right=.94, top=.93, bottom=.09)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-bound, vmax=bound, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(64), [str(i) for i in range(64)], fontsize=7)
    ax.set_yticks(np.arange(len(tissues)), tissues.tissue_name, fontsize=8)
    ax.set(xlabel="Bottleneck neuron index (zero-based)", ylabel="Source tissue = reference tissue",
           title=f"{title_prefix}model_01 | same-tissue NES | all 64 neurons")
    fig.colorbar(im, ax=ax, fraction=.022, pad=.02, label="NES (signed)")
    fig.text(.29, .025, "Grey: no matching reference or unavailable estimate. Primary significance is stored as q_matched_global.", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def read_r_results(root: Path, models: list[ModelInputs]) -> tuple[pd.DataFrame, dict]:
    """逐组织核验完整交叉结果，只将同组织行载入统一BH；不复制巨型长表。"""
    reference = pd.read_csv(root / "reference/reference_sets.csv")
    ref_names = reference.set_index("reference_tissue_id").reference_tissue.to_dict()
    expected_sizes = reference.set_index("reference_tissue_id").n_matched.to_dict()
    pieces, input_hashes = [], {}
    lock_path = root / "environment/r_environment_lock.json"
    lock_hash = sha(lock_path)
    for model in models:
        for tissue in model.tissues.itertuples(index=False):
            record_path = root / f"quality_control/fgsea/{model.tag}/{tissue.tissue_id}.json"
            record = verify_record(record_path, sources=True)
            require(record["environment_lock_sha256"] == lock_hash, "R结果使用了不同的版本锁。")
            require(record["model_number"] == model.number and record["source_tissue_id"] == tissue.tissue_id,
                    "R记录任务编号不匹配。")
            path = root / f"gsea_results/{model.tag}/{tissue.tissue_id}.csv.gz"
            require(path.as_posix() in record["output_sha256"], "R完成记录缺少结果表。")
            columns = ["model_number", "source_tissue_id", "source_tissue", "neuron_index", "unit_id", "seed",
                       "reference_tissue_id", "reference_tissue", "ES", "NES", "pval", "padj", "size", "log2err", "estimate_status"]
            table = pd.read_csv(path, usecols=columns, keep_default_na=True)
            expected = {(int(u), tid) for u in model.neurons.neuron_index for tid in ref_names}
            actual = list(zip(table.neuron_index, table.reference_tissue_id))
            require(len(actual) == len(expected) and set(actual) == expected, "GSEA结果缺行、重复或混入其他任务。")
            require(table.model_number.eq(model.number).all() and table.source_tissue_id.eq(tissue.tissue_id).all()
                    and table.source_tissue.eq(tissue.tissue_name).all(), "结果来源组织标注错误。")
            require(table.reference_tissue.eq(table.reference_tissue_id.map(ref_names)).all(), "参考组织标注错误。")
            require(table["size"].eq(table.reference_tissue_id.map(expected_sizes)).all(), "结果集合大小错误。")
            require(table.unit_id.eq(table.neuron_index.map(lambda u: f"unit_{u:05d}")).all(), "神经元编号标注错误。")
            require(table.seed.eq(table.neuron_index.map(lambda u: seed_for(model.number, int(tissue.tissue_id[-2:]), int(u)))).all(),
                    "GSEA种子不匹配。")
            for column in ("ES", "NES", "pval", "padj", "log2err"):
                require(not np.isinf(table[column].to_numpy(dtype=float)).any(), f"结果{column}含无穷大。")
            for column in ("pval", "padj"):
                valid = table[column].dropna()
                require(valid.between(0, 1).all(), f"{column}超出[0,1]。")
            require(table.ES.dropna().abs().le(1 + 1e-10).all(), "ES超出有效范围。")
            valid = table.pval.notna()
            require(table.loc[valid, ["ES", "NES"]].notna().all().all(), "有效p值缺少ES或NES。")
            require(table.estimate_status.isin(("ok", "pvalue_unavailable", "nes_unavailable", "es_unavailable", "all_zero_scores")).all(),
                    "未知的估计状态。")
            require(table.estimate_status.eq("ok").equals(valid), "估计状态与p值可用性不一致。")
            if tissue.tissue_id in ref_names:
                same = table.loc[table.reference_tissue_id.eq(tissue.tissue_id)].copy()
                same["reference_available"] = True
            else:
                same = table.drop_duplicates("neuron_index").copy()
                same["reference_tissue_id"], same["reference_tissue"] = "", ""
                same[["ES", "NES", "pval", "padj", "size", "log2err"]] = np.nan
                same["estimate_status"], same["reference_available"] = "no_matching_reference", False
            pieces.append(same)
            input_hashes[record_path.as_posix()] = sha(record_path)
        LOGGER.info("Read complete GSEA results: %s", model.tag)
    return pd.concat(pieces, ignore_index=True), input_hashes


def export_top200(root: Path, selection: pd.DataFrame) -> pd.DataFrame:
    """仅输出展示对象的正/负前200；数量不足时保留实际数量，不用0填充。"""
    genes = pd.read_csv(root / "reference/gene_map.csv", keep_default_na=False).gene_name.to_numpy()
    rows = []
    for task in selection.itertuples(index=False):
        folder = root / f"rank_inputs/model_{task.model_number:02d}/{task.source_tissue_id}"
        jobs = pd.read_csv(folder / "jobs.csv")
        position = int(jobs.loc[jobs.neuron_index.eq(task.neuron_index), "matrix_row"].iloc[0])
        values = np.memmap(folder / "scores.f64.bin", dtype="<f8", mode="r", shape=(len(jobs), len(genes)))
        scores = values[position].copy()
        values._mmap.close()
        for direction, indices in (("positive", np.flatnonzero(scores > 0)), ("negative", np.flatnonzero(scores < 0))):
            order = np.argsort(-scores[indices] if direction == "positive" else scores[indices], kind="stable")
            for rank, index in enumerate(indices[order[:200]], 1):
                rows.append({"model_number": task.model_number, "neuron_index": task.neuron_index,
                             "unit_id": task.unit_id, "source_tissue_id": task.source_tissue_id,
                             "source_tissue": task.source_tissue, "activation_group": task.group,
                             "direction": direction, "rank": rank, "gene_name": genes[index], "score": scores[index]})
    return pd.DataFrame(rows)


def finalize(root: Path, overwrite: bool) -> None:
    """主要BH必须等待20模型完成；不会对部分模型先计算一个临时全局q值。"""
    reference_record = verify_record(root / "quality_control/reference.json", sources=True)
    require(reference_record["configuration"]["design"] == DESIGN, "请先按当前Top100设计重建prepare和R结果。")
    models = [load_model(i) for i in range(1, 21)]
    table, inputs = read_r_results(root, models)
    table = table.sort_values(["model_number", "source_tissue_id", "neuron_index"]).reset_index(drop=True)
    table = table.rename(columns={"padj": "q_within_ranking"})
    eligible = table.reference_available.to_numpy(dtype=bool)
    table["q_matched_global"] = bh_planned(table.pval.to_numpy(), eligible)
    table["significant"] = table.q_matched_global.lt(.05)
    table["primary_family_size"] = int(eligible.sum())
    table["primary_use"] = "exploratory same-tissue enrichment"
    per_model, across = stability_tables(table)
    selection = pd.read_csv(root / "reference/display_selection.csv")
    chosen = table.merge(selection[["model_number", "neuron_index", "source_tissue_id", "group", "rank_in_group", "median_activation"]],
                         on=["model_number", "neuron_index", "source_tissue_id"], how="inner", validate="one_to_one")
    top = export_top200(root, selection)
    folder = root / "statistics"
    outputs = [folder / name for name in ("matched_gsea.csv", "stability_per_model.csv", "stability_across_models.csv",
                                          "display_gsea.csv", "display_top200.csv")]
    outputs += [root / f"figures/model_01/same_tissue_nes.{ext}" for ext in ("png", "pdf")]
    config = {"design": DESIGN, "script_sha256": sha(CODE), "r_records_sha256": inputs,
              "reference_record_sha256": sha(root / "quality_control/reference.json")}
    report = root / "quality_control/finalize.json"
    if not begin(report, config, outputs, overwrite):
        return
    for frame, path in zip((table, per_model, across, chosen, top), outputs):
        frame.to_csv(path, index=False, float_format="%.17g", na_rep="NA")
    plot_heatmap(table, models[0].tissues, outputs[5].with_suffix(""))
    n_missing = int((eligible & table.pval.isna().to_numpy()).sum())
    finish(report, config, outputs, n_primary_tests=int(eligible.sum()), n_missing_primary_pvalues=n_missing,
           analysis_status="complete" if n_missing == 0 else "complete_with_unavailable_estimates",
           n_cross_tissue_tests=sum(len(m.neurons) * 54 * 53 for m in models),
           n_same_tissue_significant=int(table.significant.sum()),
           source_sha256={**inputs, CODE.as_posix(): sha(CODE),
               (root / "quality_control/reference.json").as_posix(): sha(root / "quality_control/reference.json"),
               (root / "environment/r_environment_lock.json").as_posix(): sha(root / "environment/r_environment_lock.json")},
           note="No pooled cross-tissue significance claim; no neuron alignment between models.")


def test_reference_selection(folder: Path) -> None:
    """用小型已知TPM验证尺度、样本对齐、同名平均、排名和不足100的处理。"""
    folder.mkdir(parents=True, exist_ok=True)
    sample_ids = ["s1", "s2", "s3", "s4"]
    manifest = pd.DataFrame({"Sample": sample_ids, "expression_column_position": range(4),
                             "tissue_name": ["TissueB", "TissueA", "TissueB", "TissueA"]})
    phenotype = pd.DataFrame({"Sample": sample_ids, "body_site_detail (SMTSD)":
                              ["Tissue B", "Tissue A", "Tissue B", "Tissue A"]})
    tissues = pd.DataFrame({"tissue_name": ["TissueA", "TissueB"], "n_samples": [2, 2]})
    # 注释乱序后仍应按ID配对；缺失标签和重复ID必须拒绝。
    codes = align_reference_samples(sample_ids, manifest.iloc[::-1], phenotype.iloc[[2, 0, 3, 1]], tissues)
    np.testing.assert_array_equal(codes, [1, 0, 1, 0])
    for invalid in (phenotype.iloc[:-1], pd.concat([phenotype, phenotype.iloc[[0]]])):
        try:
            align_reference_samples(sample_ids, manifest, invalid, tissues)
        except ValueError:
            pass
        else:
            raise AssertionError("缺失或重复样本未被拒绝。")
    genes = pd.DataFrame({"gene_id": ["id1", "id2", "id3", "id4"], "gene_name": ["A", "A", "B", "C"]})
    tpm = np.array([[0, 1, 100, 3], [2, 3, 6, 5], [20, 2, 20, 4], [8, 9, 10, 11]], dtype=np.float64)
    stored = np.log2(tpm + .001)
    # A在组织B的平均TPM高于B，但平均log表达低于B；防止误用log均值。
    require(np.log2(tpm[:2, [0, 2]] + 1).mean() < np.log2(tpm[2, [0, 2]] + 1).mean(), "尺度反例无效。")
    with tempfile.TemporaryDirectory(prefix="top100_qa_", dir=folder) as temporary:
        path = Path(temporary) / "expression.tsv"
        pd.DataFrame(stored, index=genes.gene_id, columns=sample_ids).to_csv(path, sep="\t", float_format="%.17g")
        means, audit = scan_candidate_mean_tpm(path, genes, codes, np.array([2, 2]), ["A", "B"])
        np.testing.assert_allclose(means, [[3, 27], [3, 20]], rtol=1e-13, atol=1e-13)
        require(audit["expression_sha256"] == sha(path), "表达文件扫描指纹错误。")
        require(rank_top100(["B", "A"], pd.Series(means[:, 1], index=["A", "B"])).gene_name.tolist() == ["A", "B"],
                "TPM表达排名错误。")
    tied = [f"G{i:03d}" for i in range(101)]
    ranked = rank_top100(tied[::-1], pd.Series(1.0, index=tied))
    require(ranked.loc[ranked.selected, "gene_name"].tolist() == tied[:100], "Top100并列规则不稳定。")
    short = rank_top100(tied[:99], pd.Series(1.0, index=tied))
    require(len(short) == 99 and short.selected.all(), "不足100的候选被错误补齐。")


def self_test(root: Path) -> None:
    """只写quality_control：合并、BH、稳定性、输入格式和版面的独立检查。"""
    report = root / "quality_control/self_test.json"
    write_json(report, {"status": "running"})
    test_reference_selection(root / "quality_control/self_test")
    model = load_model(1)
    merged, features = make_gene_map(model.genes)
    reference = build_reference(merged, model.tissues)
    require(len(merged) == 58581, "合并基因数错误。")
    # 手工已知均值与含多个同名特征的结构；确保不会丢弃负梯度。
    feature = pd.DataFrame({"merged_column": [1, 0, 1, 0]})
    np.testing.assert_array_equal(merge_scores(np.array([[1, -2, 3, -4]], dtype=np.float32), feature, np.array([2, 2])), [[-3, 2]])
    p = np.array([.01, .04, .03, .002])
    np.testing.assert_allclose(bh_planned(p, np.ones(4, dtype=bool)), [.02, .04, .04, .008])
    np.testing.assert_allclose(bh_planned(p, np.ones(4, dtype=bool)), stats.false_discovery_control(p))
    missing = bh_planned(np.array([.01, np.nan, .03, np.nan]), np.array([True, True, True, False]))
    np.testing.assert_allclose(missing[[0, 2]], [.03, .045])
    require(np.isnan(missing[[1, 3]]).all(), "缺失p或缺失参考被错误填充。")
    seed_values = [seed_for(m, t, u) for m in range(1, 21) for t in range(1, 55) for u in range(64)]
    require(len(set(seed_values)) == 69120 and max(seed_values) < 2**31, "任务种子重复或超出R整数范围。")
    selected, _ = display_selection(model)
    require(len(selected) == 6 * len(DISPLAY_UNITS), "展示组织选择错误。")
    # 真实数据只处理一个组织两个神经元，用于检查跨语言二进制接口。
    matrix, _ = checked_mean(model, next(model.tissues.itertuples(index=False)))
    score = merge_scores(matrix[[0, -1]], features, merged.n_features.to_numpy())
    folder = root / "quality_control/self_test"
    folder.mkdir(parents=True, exist_ok=True)
    score.astype("<f8").tofile(folder / "real_scores.f64.bin")
    np.testing.assert_array_equal(np.fromfile(folder / "real_scores.f64.bin", dtype="<f8").reshape(score.shape), score)
    # 抽查任意同名组，参考实现直接按特征成员取平均，独立于reduceat。
    for name in ("Y_RNA", "RAB4B", merged.gene_name.iloc[-1]):
        column = int(merged.loc[merged.gene_name.eq(name), "merged_column"].iloc[0])
        indices = np.flatnonzero(model.genes.gene_name.eq(name))
        np.testing.assert_allclose(score[:, column], matrix[[0, -1]][:, indices].mean(1, dtype=np.float64), atol=1e-14)
    # 合成QA数据只检验汇总和图形；不会写正式统计目录。
    synthetic = []
    for m in range(1, 21):
        for tissue in model.tissues.itertuples(index=False):
            available = tissue.tissue_id in reference["pathways"]
            for u in range(64):
                q = .01 if (u + m) % 4 == 0 else .3
                synthetic.append({"model_number": m, "source_tissue_id": tissue.tissue_id,
                                  "source_tissue": tissue.tissue_name, "neuron_index": u,
                                  "reference_available": available, "q_matched_global": q if available else np.nan,
                                  "NES": np.sin((u + 1) * int(tissue.tissue_id[-2:]) / 35) * 3 if available else np.nan,
                                  "significant": available and q < .05})
    fake = pd.DataFrame(synthetic)
    a, b = stability_tables(fake)
    permuted = fake.copy()
    permuted.neuron_index = (permuted.neuron_index * 7 + permuted.model_number) % 64
    aa, bb = stability_tables(permuted.sample(frac=1, random_state=629))
    pd.testing.assert_frame_equal(a, aa)
    pd.testing.assert_frame_equal(b, bb)
    require(b.loc[~b.reference_available, "detection_frequency"].isna().all(), "缺失参考不应成为零稳定性。")
    plot_heatmap(fake, model.tissues, folder / "synthetic_heatmap", title_prefix="QA SYNTHETIC - ")
    write_json(report, {"status": "passed", "n_source_sets": 511, "n_reference_sets": 53,
                        "n_merged_genes": len(merged), "n_unmatched_genes": 190,
                        "signed_merge_checked": True, "planned_BH_and_missing_values_checked": True,
                        "neuron_permutation_invariance": True, "all_job_seeds_unique": True,
                        "binary_roundtrip": True, "display_tissue_selection": selected.to_dict("records"),
                        "top100_linear_tpm_duplicate_mean_sample_alignment_ties_shortfall": True,
                        "reference_selection": REFERENCE_SELECTION,
                        "script_sha256": sha(CODE), "R_fgsea_executed": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="Step5 reference, rankings and GSEA summaries")
    parser.add_argument("stage", choices=("check", "reference", "prepare", "finalize", "self-test"))
    parser.add_argument("--model-start", type=int, default=1)
    parser.add_argument("--model-end", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require(1 <= args.model_start <= args.model_end <= 20, "模型范围必须在1–20。")
    require(sklearn.__version__ == "1.9.0", "需要scikit-learn==1.9.0。")
    os.chdir(TRIAL_ROOT)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / f"logs/{args.stage}.log", encoding="utf-8"), logging.StreamHandler()])
    if args.stage == "self-test":
        self_test(ROOT)
        return
    if args.stage == "finalize":
        require(args.model_start == 1 and args.model_end == 20, "统一BH必须覆盖20模型。")
        finalize(ROOT, args.overwrite)
        return
    baseline = load_model(1)
    merged, features = make_gene_map(baseline.genes)
    reference = build_reference(merged, baseline.tissues)
    require(len(merged) == 58581, "合并基因总数与固定输入不匹配。")
    if args.stage in ("reference", "prepare"):
        reference = select_reference_top100(baseline, reference)
        write_reference(ROOT, baseline, reference, args.overwrite)
        if args.stage == "reference":
            LOGGER.info("Top100 reference complete: %d selected memberships; %d tissues with a shortfall",
                        reference["selection_metadata"]["n_selected_members"], len(reference["selection_metadata"]["shortfalls"]))
            return
    summaries = []
    for number in range(args.model_start, args.model_end + 1):
        model = baseline if number == 1 else load_model(number)
        pd.testing.assert_frame_equal(model.genes, baseline.genes)
        pd.testing.assert_frame_equal(model.tissues, baseline.tissues)
        for tissue in model.tissues.itertuples(index=False):
            if args.stage == "check":
                matrix, _ = checked_mean(model, tissue)
                del matrix
            else:
                prepare_tissue(ROOT, model, tissue, merged, features, args.overwrite)
        summaries.append({"model_number": number, "n_candidates": len(model.neurons), "n_tissues": 54,
                          "n_genes": len(merged), "mean_fingerprints_checked": 54})
        LOGGER.info("%s completed: %s", args.stage, model.tag)
    write_json(ROOT / "quality_control/input_check.json", {"status": "passed", "models": summaries,
               "reference_sets": 53, "source_sets": 511, "unmatched_genes": 190,
               "gene_score_input_bytes": sum(s["n_candidates"] * 54 * 58581 * 8 for s in summaries),
               "script_sha256": sha(CODE), "R_not_started": True})


if __name__ == "__main__":
    main()

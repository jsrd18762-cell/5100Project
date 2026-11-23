# Step7 共享函数，仅由本目录的R入口载入。
# Step5共享文件只提供读取、排序和fgsea计算函数；不执行其分析入口。
# 本文件重新约束写入函数，确保所有输出只进入step7。

local({
  here <- getwd()
  repeat {
    if (file.exists(file.path(here, "step7/code/step7_weight_analysis.py"))) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请从trial或其子目录启动R。")
    here <- parent
  }
  setwd(here)
})
if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
.libPaths(c("step5/environment/library", .libPaths()))
source("step5/code/_step5_common.R", local = TRUE, encoding = "UTF-8")

step7_output <- function(path) {
  ensure(startsWith(path, "step7/") && !grepl("(^|/)\\.\\.(/|$)", path), "输出必须位于step7。")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

write_json <- function(path, value) {
  step7_output(path)
  temporary <- paste0(path, ".tmp")
  jsonlite::write_json(value, temporary, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
  if (file.exists(path)) ensure(file.remove(path), paste("无法替换", path))
  ensure(file.rename(temporary, path), paste("无法写入", path))
}

write_table <- function(value, path) {
  step7_output(path)
  data.table::fwrite(value, path, na = "NA", quote = "auto")
}

save_session <- function(path) {
  step7_output(path)
  writeLines(capture.output(utils::sessionInfo()), path, useBytes = TRUE)
}

step7_environment <- function() {
  check_environment()  # 使用按numeric_version比较的已锁定环境，不安装或升级R包。
  ensure(identical(sha256("step5/environment/r_environment_lock.json"),
                   sha256("step7/environment/r_environment_lock.json")), "Step7版本锁快照与当前环境不同。")
  ensure(isTRUE(utils::packageVersion("fgsea") == numeric_version("1.38.0")), "需要fgsea 1.38.0。")
}

step7_propagation_design <- function(design) {
  # 核对带符号入口与完整后续层，拒绝跳过第一组BN/LeakyReLU的评分。
  ensure(identical(design$mode, "eval") && identical(design$grad_enabled, FALSE) &&
         identical(design$entry_formula, "10000 * decoder.0.weight.T") &&
         identical(design$propagation_entry, "decoder[1:]") &&
         identical(design$bn_statistics, "trained running statistics; decoder.1 and decoder.5 each called once") &&
         identical(design$dropout, "disabled by eval; p=0.3 modules retained") &&
         isTRUE(design$propagation_batch_size == 64L) && isTRUE(design$propagation_passes == 1L) &&
         identical(design$propagation_seed_formula, "629 + 7100000 + (model_number - 1)"),
         "需要eval模式、带符号入口及decoder[1:]的64行单次传播输入；请先重建Python prepare及其后续结果。")
}

step7_reference <- function() {
  path <- "step7/quality_control/reference.json"
  step7_propagation_design(read_json(path)$configuration$design)
  record <- check_record(path, sources = TRUE)
  genes <- read_table("step7/reference/gene_map.csv")
  sets <- read_table("step7/reference/reference_sets.csv")
  pathways <- lapply(read_json("step7/reference/pathways.json"), unlist, use.names = FALSE)
  design <- read_json("step7/reference/analysis_design.json")
  selection <- read_json("step7/reference/reference_selection.json")
  check_top100_reference(sets, pathways, selection, design)
  step7_propagation_design(design)
  ensure(isTRUE(all.equal(record$configuration$design, design, check.attributes = FALSE)), "参考快照与配置中的设计不同。")
  expected <- list(minSize = 15, maxSize = 500, eps = 0, scoreType = "std", gseaParam = 1, sampleSize = 101, nPermSimple = 1000)
  ensure(nrow(genes) == 58581L && !anyDuplicated(genes$gene_name) &&
         identical(as.integer(genes$merged_column), 0:58580), "基因顺序错误。")
  ensure(nrow(sets) == 53L && identical(names(pathways), sets$reference_tissue_id) &&
         all(lengths(pathways) == sets$n_matched), "参考集合错误。")
  ensure(isTRUE(all.equal(design$parameters, expected, check.attributes = FALSE)), "GSEA参数不匹配。")
  list(genes = genes$gene_name, sets = sets, pathways = pathways, parameters = expected, design = design,
       record = record, record_path = path)
}

step7_prepared <- function(model, reference) {
  path <- sprintf("step7/quality_control/prepared/model_%02d.json", model)
  step7_propagation_design(read_json(path)$configuration$design)
  record <- check_record(path, sources = TRUE)
  ensure(isTRUE(all.equal(record$configuration$design, reference$design, check.attributes = FALSE)) &&
         isTRUE(record$configuration$propagation_seed == 629 + 7100000 + model - 1) &&
         identical(record$audit$training_mode, FALSE) &&
         isTRUE(record$audit$propagation$seed == record$configuration$propagation_seed), "权重传播模式或种子记录不匹配。")
  # eval传播不能更新任何BN缓冲区；检查两个BN均有审计记录。
  updates <- record$audit$propagation$bn_buffer_updates
  calls <- record$audit$propagation$module_call_counts
  ensure(setequal(names(calls), paste0("decoder.", 0:8)) &&
           isTRUE(calls[["decoder.0"]] == 0L) &&
           all(vapply(calls[paste0("decoder.", 1:8)], function(x) isTRUE(x == 1L), logical(1))),
         "实际传播必须经过两组BN/LeakyReLU各一次，不能跳层或重复第一层Linear。")
  ensure(setequal(names(updates), c("decoder.1", "decoder.5")) &&
         all(vapply(updates, function(x) isTRUE(x$batches_before == x$batches_after) &&
           isTRUE(x$running_mean_max_abs_change == 0) && isTRUE(x$running_var_max_abs_change == 0), logical(1))),
         "BN缓冲区审计不符合评估模式。")
  ensure(record$model_number == model && record$n_neurons == 64L && record$n_genes == 58581L &&
         record$binary_dtype == "little-endian float64", "权重输入形状或格式错误。")
  ensure(identical(record$source_sha256[[reference$record_path]], sha256(reference$record_path)), "参考库来源改变。")
  folder <- sprintf("step7/rank_inputs/model_%02d", model)
  jobs <- read_table(paste0(folder, "/jobs.csv"))
  ensure(nrow(jobs) == 64L && all(jobs$model_number == model) &&
         identical(as.integer(jobs$matrix_row), 0:63) && identical(as.integer(jobs$neuron_index), 0:63) &&
         all(jobs$unit_id == sprintf("unit_%05d", 0:63)) &&
         all(jobs$seed == 629 + 7000000 + (model - 1) * 64 + jobs$neuron_index), "任务行或种子错误。")
  values <- read_scores(paste0(folder, "/scores.f64.bin"), 64L, 58581L)
  list(jobs = jobs, values = values, record = record, record_path = path)
}

step7_fit <- function(scores, job, reference) {
  # 每个神经元只运行一次，其53个组织集合一次性评估；不添加来源组织维度。
  ranked <- rank_scores(scores, reference$genes)
  if (all(ranked == 0)) {
    table <- data.frame(pathway = reference$sets$reference_tissue_id, pval = NA_real_, padj = NA_real_,
      ES = NA_real_, NES = NA_real_, log2err = NA_real_, size = reference$sets$n_matched,
      leadingEdge = "", estimate_status = "all_zero_scores")
    warning_text <- "All scores are zero; GSEA unavailable."
  } else {
    fit <- fgsea_one(ranked, reference$pathways, job$seed, reference$parameters)
    table <- fit$table
    ensure(nrow(table) == 53L && !anyDuplicated(table$pathway) && setequal(table$pathway, names(reference$pathways)),
           "fgsea返回的53个参考集合不完整。")
    table <- table[match(reference$sets$reference_tissue_id, table$pathway), , drop = FALSE]
    ensure(all(table$size == reference$sets$n_matched), "fgsea集合大小错误。")
    table$leadingEdge <- vapply(table$leadingEdge, paste, character(1), collapse = ";")
    table$estimate_status <- ifelse(!is.finite(table$ES), "es_unavailable",
      ifelse(!is.finite(table$NES), "nes_unavailable", ifelse(!is.finite(table$pval), "pvalue_unavailable", "ok")))
    warning_text <- fit$warnings
  }
  for (column in c("ES", "NES", "pval", "padj", "log2err")) {
    ensure(!any(is.infinite(table[[column]])), paste("无穷大结果：", column))
    table[[column]][is.nan(table[[column]])] <- NA_real_
  }
  table$model_number <- job$model_number
  table$neuron_index <- job$neuron_index
  table$unit_id <- job$unit_id
  table$seed <- job$seed
  table$reference_tissue_id <- table$pathway
  table$reference_tissue <- reference$sets$reference_tissue
  table$score_source <- "weight"
  table$ranking_scope <- "tissue_independent"
  table$pathway <- NULL
  list(table = table, warnings = warning_text)
}

step7_curve <- function(ranked, members) {
  # 使用fgsea 1.38.0计算阶段相同的缩放/取整权重，保留带符号排序顺序。
  weights <- suppressWarnings(getFromNamespace("prepareStats", "fgsea")(ranked, scoreType = "std", gseaParam = 1))
  ensure(identical(names(weights), names(ranked)), "prepareStats改变了基因顺序。")
  positions <- which(names(weights) %in% members)
  ensure(length(positions) == length(members), "曲线成员不完整。")
  result <- fgsea::calcGseaStat(weights, positions, gseaParam = 1, returnAllExtremes = TRUE, scoreType = "std")
  ensure(abs(running_es(weights, members) - result$res) < 1e-10, "累计曲线与ES公式不一致。")
  list(curve = data.frame(rank = c(0, as.vector(rbind(positions - 1L, positions)), length(weights)),
                          ES = c(0, as.vector(rbind(result$bottoms, result$tops)), 0)),
       ticks = data.frame(rank = positions), ES = result$res)
}

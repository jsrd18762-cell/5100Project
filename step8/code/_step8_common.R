# Step8 共享函数：只读取上游参考库和软件锁，所有写入在 step8。
# 评分是每模型、每神经元一份完整带符号排序；组织仅作为参考集合。
source("step5/code/_step5_common.R", encoding = "UTF-8")
step5_root()

step8_output <- function(path) {
  ensure(grepl("^step8/", path) && !grepl("(^|/)\\.\\.(/|$)", path), "输出必须位于 step8。")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

# 保存上游纯序列化函数，增加本步骤的写入边界；不修改上游函数文件。
step8_json_writer <- write_json
step8_table_writer <- write_table
write_json <- function(path, value) {
  step8_output(path)
  step8_json_writer(path, value)
}
write_table <- function(value, path) {
  step8_output(path)
  step8_table_writer(value, path)
}

step8_environment <- function() {
  check_environment()
  ensure(identical(sha256("step8/environment/r_environment_lock.json"),
                   sha256("step7/environment/r_environment_lock.json")), "Step8 R 版本锁与 Step7 不同。")
  ensure(identical(sha256("step8/environment/r_environment_lock.json"),
                   sha256("step5/environment/r_environment_lock.json")), "共享函数使用的 R 版本锁与 Step8 不同。")
  data.table::setDTthreads(1L)
  options(mc.cores = 1L)
  invisible(TRUE)
}

step8_reference <- function(condition) {
  folder <- paste0("step8/reference/", condition$input_scope)
  genes <- read_table(paste0(folder, "/gene_map.csv"))
  sets <- read_table(paste0(folder, "/reference_sets.csv"))
  pathways <- lapply(read_json(paste0(folder, "/pathways.json")), unlist, use.names = FALSE)
  ensure(!anyDuplicated(genes$gene_name) && identical(as.integer(genes$merged_column), seq_len(nrow(genes))-1L), "基因顺序错误。")
  ensure(nrow(sets) == 53L && identical(names(pathways), sets$reference_tissue_id) &&
           all(lengths(pathways) == sets$n_matched) && all(sets$n_matched >= 15L) &&
           all(sets$n_matched <= 500L), "53组织集合不符合 GSEA 大小限制。")
  ensure(all(vapply(pathways, function(x) !anyDuplicated(x) && all(x %in% genes$gene_name), logical(1))), "集合含未知或重复基因。")
  list(genes = genes$gene_name, sets = sets, pathways = pathways,
       paths = c(paste0(folder, "/gene_map.csv"), paste0(folder, "/reference_sets.csv"), paste0(folder, "/pathways.json")))
}

step8_prepared <- function(task, design) {
  cid <- task$condition_id
  condition <- design$conditions[[cid]]
  reference <- step8_reference(condition)
  report_path <- sprintf("step8/quality_control/prepared/%s/model_%02d.json", cid, task$model_number)
  report <- check_record(report_path, sources = TRUE)
  ensure(identical(report$condition_id, cid) && report$model_number == task$model_number &&
           report$n_genes == length(reference$genes) && report$n_neurons == 64L &&
           identical(report$binary_dtype, "little-endian float64") &&
           identical(report$configuration$design_sha256, sha256("step8/metadata/design.json")), "prepare记录的条件、模型或设计错误。")
  folder <- sprintf("step8/rank_inputs/%s/model_%02d", cid, task$model_number)
  jobs <- read_table(paste0(folder, "/jobs.csv"))
  expected <- 629 + 9000000 + condition$condition_index * 100000 + (task$model_number-1L)*64 + 0:63
  ensure(nrow(jobs) == 64L && identical(as.integer(jobs$neuron_index), 0:63) &&
           all(jobs$condition_id == cid) && all(jobs$model_number == task$model_number) &&
           identical(as.integer(jobs$matrix_row), 0:63) && all(jobs$seed == expected) &&
           all(jobs$unit_id == sprintf("unit_%05d", 0:63)), "任务主键或派生种子不匹配。")
  list(reference = reference, report = report, report_path = report_path, jobs = jobs,
       values = read_scores(paste0(folder, "/scores.f64.bin"), 64L, length(reference$genes)))
}

step8_fit <- function(scores, job, reference, parameters) {
  ranked <- rank_scores(scores, reference$genes)
  if (all(ranked == 0)) {
    table <- data.frame(pathway = reference$sets$reference_tissue_id,
      pval = NA_real_, padj = NA_real_, ES = NA_real_, NES = NA_real_, log2err = NA_real_,
      size = reference$sets$n_matched, leadingEdge = "", estimate_status = "all_zero_scores")
    messages <- "All-zero ranking: ES/NGCI remain undefined."
  } else {
    fitted <- fgsea_one(ranked, reference$pathways, job$seed, parameters)
    table <- fitted$table
    ensure(nrow(table) == 53L && !anyDuplicated(table$pathway) &&
             setequal(table$pathway, names(reference$pathways)), "fgsea 未返回全部53集合。")
    table <- table[match(reference$sets$reference_tissue_id, table$pathway), , drop = FALSE]
    ensure(all(table$size == reference$sets$n_matched), "GSEA集合大小改变。")
    table$leadingEdge <- vapply(table$leadingEdge, paste, character(1), collapse = ";")
    table$estimate_status <- ifelse(!is.finite(table$ES), "es_unavailable",
      ifelse(!is.finite(table$pval), "pvalue_unavailable", "ok"))
    messages <- fitted$warnings
  }
  for (name in c("ES", "NES", "pval", "padj", "log2err")) {
    ensure(!any(is.infinite(table[[name]])), paste("无穷大：", name))
    table[[name]][is.nan(table[[name]])] <- NA_real_
  }
  table$reference_tissue_id <- table$pathway
  table$reference_tissue <- reference$sets$reference_tissue
  table$condition_id <- job$condition_id
  table$model_number <- job$model_number
  table$neuron_index <- job$neuron_index
  table$unit_id <- job$unit_id
  table$seed <- job$seed
  table$score_source <- "weight"
  table$pathway <- NULL
  list(table = table, warnings = messages)
}

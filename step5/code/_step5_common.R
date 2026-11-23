# Step5共享函数。本文件由00、01、02脚本source，不需要单独运行。
# 所有数据路径相对于trial。不会调用Python、训练模型或更改上游文件。

step5_root <- function() {
  # 支持在trial或其子目录启动；不使用机器专属的盘符和安装目录。
  here <- getwd()
  repeat {
    if (file.exists(file.path(here, "step5/code/step5_gsea.py"))) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请先将R工作目录设为trial或其子目录。")
    here <- parent
  }
  setwd(here)
  library_dir <- "step5/environment/library"
  if (dir.exists(library_dir)) .libPaths(c(library_dir, .libPaths()))
  invisible(here)
}

ensure <- function(ok, message) {
  if (length(ok) != 1L || is.na(ok) || !ok) stop(message, call. = FALSE)
}

read_json <- function(path) jsonlite::fromJSON(path, simplifyVector = FALSE)

write_json <- function(path, value) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  jsonlite::write_json(value, temporary, auto_unbox = TRUE, pretty = TRUE,
                       digits = NA, na = "null", null = "null")
  # Windows不保证rename覆盖已存在文件，先仅删除本次要替换的单一文件。
  if (file.exists(path)) ensure(file.remove(path), paste("不能替换", path))
  ensure(file.rename(temporary, path), paste("不能保存", path))
}

sha256 <- function(path) {
  ensure(file.exists(path), paste("文件缺失：", path))
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

hash_files <- function(paths) {
  paths <- unique(paths)
  setNames(lapply(paths, sha256), paths)
}

check_hashes <- function(hashes) {
  ensure(length(hashes) > 0L && !is.null(names(hashes)), "指纹清单不能为空。")
  for (path in names(hashes)) {
    ensure(identical(sha256(path), hashes[[path]]), paste("文件指纹已改变：", path))
  }
  invisible(TRUE)
}

check_record <- function(path, sources = FALSE) {
  value <- read_json(path)
  ensure(identical(value$status, "complete"), paste("阶段尚未完成：", path))
  check_hashes(value$output_sha256)
  if (sources) check_hashes(value$source_sha256)
  value
}

check_environment <- function() {
  lock_path <- "step5/environment/r_environment_lock.json"
  ensure(file.exists(lock_path), "请先手动运行00_setup_environment.R建立版本锁。")
  lock <- read_json(lock_path)
  ensure(identical(as.character(getRversion()), lock$R), "R版本与锁定版本不一致。")
  ensure(identical(R.version$platform, lock$platform), "R平台与版本锁不一致。")
  for (package in names(lock$packages)) {
    # LinkingTo依赖可能只有头文件，无需强行加载其namespace。
    ensure(length(find.package(package, quiet = TRUE)) == 1L, paste("缺少锁定包：", package))
    ensure(isTRUE(utils::packageVersion(package) == numeric_version(lock$packages[[package]])),
           paste("包版本与版本锁不一致：", package))
  }
  required <- c("fgsea", "BiocParallel", "data.table", "jsonlite", "digest", "ggplot2")
  ensure(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)), "无法加载分析所需R包。")
  for (package in intersect(names(lock$packages), loadedNamespaces())) {
    # R版本号中的连字符与点号等价，例如0.2-20和0.2.20。
    # 比较解析后的版本，仍拒绝实际版本差异，避免字符串比较造成误报。
    ensure(isTRUE(numeric_version(as.character(getNamespaceVersion(package))) == numeric_version(lock$packages[[package]])),
           paste("当前会话仍加载了不同版本，请重启R：", package))
  }
  invisible(lock)
}

save_session <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(capture.output(utils::sessionInfo()), path, useBytes = TRUE)
}

read_table <- function(path) {
  # 不能把真实基因名"NA"当作缺失值；数值缺失由具体结果读取代码处理。
  data.table::fread(path, data.table = FALSE, encoding = "UTF-8", na.strings = NULL)
}

write_table <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(value, path, na = "NA", quote = "auto")
}

read_scores <- function(path, n_rows, n_genes) {
  # Python写出的格式：小端IEEE754 float64，无文件头，先候选行、后基因列。
  ensure(file.info(path)$size == as.double(n_rows) * n_genes * 8, "二进制矩阵长度不匹配。")
  connection <- file(path, "rb")
  on.exit(close(connection), add = TRUE)
  values <- readBin(connection, what = "double", n = n_rows * n_genes, size = 8, endian = "little")
  ensure(length(values) == n_rows * n_genes && all(is.finite(values)), "评分矩阵含无效数值。")
  matrix(values, nrow = n_rows, ncol = n_genes, byrow = TRUE)
}

rank_scores <- function(scores, genes) {
  ensure(length(scores) == length(genes) && !anyDuplicated(genes), "基因顺序或唯一性错误。")
  # 评分降序；完全并列时保留Python建立的UTF-8字节序，不随机抖动。
  index <- order(-scores, seq_along(scores), method = "radix")
  setNames(scores[index], genes[index])
}

check_top100_reference <- function(sets, pathways, selection, design) {
  # 两条路线调用同一验证函数；拒绝旧的组织并集及不完整的Top100快照。
  ensure(identical(selection$specification$id, "tissue_top100_mean_tpm_v1") &&
           isTRUE(selection$specification$target_size == 100L) &&
           isTRUE(all.equal(selection$specification, design$reference_selection, check.attributes = FALSE)),
         "需要同组织完整样本平均TPM的Top100参考库，请先运行Step5 prepare。")
  ensure(all(c("n_candidates_matched", "n_selected", "shortfall") %in% names(sets)), "缺少Top100选择记录。")
  ensure(nrow(sets) == 53L && identical(names(pathways), sets$reference_tissue_id) &&
           all(sets$n_selected == pmin(100L, sets$n_candidates_matched)) &&
           all(sets$n_matched == sets$n_selected) && all(lengths(pathways) == sets$n_selected) &&
           all(sets$shortfall == 100L - sets$n_selected) &&
           isTRUE(sum(sets$n_selected) == selection$n_selected_members), "Top100成员数或缺额记录不一致。")
  invisible(TRUE)
}

load_reference <- function() {
  path <- "step5/quality_control/reference.json"
  record <- check_record(path, sources = TRUE)
  gene_map <- read_table("step5/reference/gene_map.csv")
  sets <- read_table("step5/reference/reference_sets.csv")
  pathways <- lapply(read_json("step5/reference/pathways.json"), unlist, use.names = FALSE)
  design <- read_json("step5/reference/analysis_design.json")
  selection <- read_json("step5/reference/reference_selection.json")
  check_top100_reference(sets, pathways, selection, design)
  ensure(isTRUE(all.equal(record$configuration$design, design, check.attributes = FALSE)), "参考库设计与完成记录不同。")
  ensure(nrow(gene_map) == 58581L && !anyDuplicated(gene_map$gene_name) &&
           identical(as.integer(gene_map$merged_column), seq_len(nrow(gene_map)) - 1L), "基因映射错误。")
  ensure(nrow(sets) == 53L && identical(names(pathways), sets$reference_tissue_id), "参考集合顺序错误。")
  ensure(all(lengths(pathways) == sets$n_matched) && all(lengths(pathways) >= 15L) &&
           all(lengths(pathways) <= 500L), "有效集合大小错误。")
  ensure(all(vapply(pathways, function(x) !anyDuplicated(x) && all(x %in% gene_map$gene_name), logical(1))),
         "参考集合含重复成员或不属于基因空间的名称。")
  expected <- list(minSize = 15, maxSize = 500, eps = 0, scoreType = "std", gseaParam = 1,
                   sampleSize = 101, nPermSimple = 1000)
  ensure(identical(design$method, "fgseaMultilevel") &&
           isTRUE(all.equal(design$parameters, expected, check.attributes = FALSE)), "GSEA参数与批准配置不一致。")
  list(genes = gene_map$gene_name, sets = sets, pathways = pathways, design = design,
       record_path = path, record = record)
}

load_prepared <- function(model, tissue_id, reference) {
  tag <- sprintf("model_%02d", model)
  qc_path <- sprintf("step5/quality_control/prepared/%s/%s.json", tag, tissue_id)
  record <- check_record(qc_path, sources = TRUE)
  config <- record$configuration
  ensure(config$model_number == model && identical(config$source_tissue_id, tissue_id) &&
           config$n_genes == length(reference$genes) &&
           identical(config$binary_dtype, "little-endian float64"), "准备阶段的任务配置错误。")
  # 明确要求本批次prepare使用当前参考库，禁止将不同参考库的结果混合。
  ensure(identical(record$source_sha256[[reference$record_path]], sha256(reference$record_path)),
         "准备阶段使用的参考库已变化。")
  folder <- sprintf("step5/rank_inputs/%s/%s", tag, tissue_id)
  jobs <- read_table(paste0(folder, "/jobs.csv"))
  tissue_number <- as.integer(sub("tissue_", "", tissue_id, fixed = TRUE))
  seeds <- 629 + 5000000 + (model - 1) * 54 * 64 + (tissue_number - 1) * 64 + jobs$neuron_index
  ensure(nrow(jobs) == config$n_candidates && identical(as.integer(jobs$matrix_row), seq_len(nrow(jobs)) - 1L) &&
           all(jobs$model_number == model) && all(jobs$source_tissue_id == tissue_id) &&
           all(jobs$source_tissue == config$source_tissue) && all(jobs$seed == seeds) &&
           !anyDuplicated(jobs$neuron_index) && all(jobs$neuron_index %in% 0:63) &&
           all(jobs$unit_id == sprintf("unit_%05d", jobs$neuron_index)), "候选任务或种子映射错误。")
  values <- read_scores(paste0(folder, "/scores.f64.bin"), nrow(jobs), length(reference$genes))
  list(jobs = jobs, values = values, record = record, record_path = qc_path)
}

fgsea_one <- function(ranked, pathways, seed, parameters) {
  # 每个任务单独重设RNG；串行后端避免工作进程数量改变随机序列。
  RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
  set.seed(as.integer(seed))
  messages <- character()
  args <- c(list(pathways = pathways, stats = ranked, BPPARAM = BiocParallel::SerialParam()), parameters)
  result <- withCallingHandlers(do.call(fgsea::fgseaMultilevel, args), warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(table = as.data.frame(result), warnings = unique(messages))
}

running_es <- function(ranked, members) {
  # 独立计算带权ES，用于小型检查以及核对展示曲线。gseaParam固定为1。
  hit <- names(ranked) %in% members
  ensure(any(hit) && !all(hit), "ES需要非空且不覆盖全部基因的集合。")
  weight <- abs(ranked[hit])
  if (sum(weight) == 0) weight[] <- 1
  increments <- rep(-1 / sum(!hit), length(hit))
  increments[hit] <- weight / sum(weight)
  curve <- c(0, cumsum(increments))
  highest <- max(curve)
  lowest <- min(curve)
  # 标准双侧ES在正负峰绝对值完全相同时为0，与fgsea定义一致。
  if (highest == -lowest) return(0)
  if (highest > -lowest) highest else lowest
}

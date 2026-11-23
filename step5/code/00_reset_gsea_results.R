if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# 清空GSEA阶段输出，供全部20模型、54组织重新计算。仅在需要清空时运行。
# 先停止所有正在运行的GSEA进程，再在trial工作目录的R控制台执行：
# options(step5.reset_results=TRUE)
# source("step5/code/00_reset_gsea_results.R", encoding="UTF-8")
# 保留参考库、评分输入、版本锁和当前12进程检查报告；不处理Step1至Step4。

local({
  here <- getwd()
  repeat {
    common <- file.path(here, "step5/code/_step5_common.R")
    if (file.exists(common)) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请先将R工作目录设为trial或其子目录。")
    here <- parent
  }
  source(common, local = environment(), encoding = "UTF-8")
  step5_root()
  ensure(isTRUE(getOption("step5.reset_results", FALSE)),
         "清理会删除GSEA结果。需要清空时，先设置options(step5.reset_results=TRUE)。")
  options(step5.reset_results = FALSE)
  targets <- c("step5/gsea_results", "step5/statistics", "step5/figures",
    "step5/quality_control/fgsea", "step5/quality_control/finalize.json",
    "step5/quality_control/enrichment_plots.json", "step5/quality_control/r_self_test",
    "step5/quality_control/delivery_validation.json", "step5/quality_control/reset_targets.json",
    "step5/environment/sessionInfo_fgsea.txt",
    "step5/environment/sessionInfo_plots.txt", "step5/code/01_run_fgsea.R")
  # 同时清理任务日志；12进程自检的日志保留，便于查阅当前检查证据。
  log_dirs <- list.dirs("step5/logs", recursive = FALSE, full.names = TRUE)
  for (directory in log_dirs) {
    if (startsWith(basename(directory), "parallel_run_")) targets <- c(targets, directory)
    if (startsWith(basename(directory), "parallel_self-test_")) {
      path <- paste0(directory, "/workers.json")
      if (file.exists(path) && length(read_json(path)$workers) != 12L) targets <- c(targets, directory)
    }
  }
  targets <- unique(targets[file.exists(targets) | dir.exists(targets)])
  root <- normalizePath("step5", winslash = "/", mustWork = TRUE)
  # 在任何删除前逐一核对解析后的完整路径；只接受step5内部的明确目标。
  for (target in targets) {
    absolute <- normalizePath(target, winslash = "/", mustWork = TRUE)
    ensure(startsWith(tolower(absolute), paste0(tolower(root), "/")),
           paste("清理目标不在step5内部：", target))
  }
  preserved <- c("step5/reference/analysis_design.json", "step5/reference/pathways.json",
    "step5/quality_control/reference.json", "step5/environment/r_environment_lock.json",
    "step5/code/step5_gsea.py")
  fingerprints <- hash_files(preserved)
  for (target in targets) {
    ensure(unlink(target, recursive = TRUE, force = TRUE) == 0L,
           paste("未能清理，请确认GSEA已停止且文件未被占用：", target))
  }
  ensure(!any(file.exists(targets) | dir.exists(targets)), "仍有未删除的GSEA输出。")
  check_hashes(fingerprints)
  write_json("step5/quality_control/reset_for_12_workers.json", list(status = "complete",
    requested_workers = 12L, removed_paths = targets,
    reset_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    preserved_sha256 = fingerprints, full_recalculation_started = FALSE))
  message("GSEA输出与旧串行代码已清理。评分输入已保留；下一次run将计算全部所选组织。")
})

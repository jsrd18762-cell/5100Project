if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# Step7：12个外层进程，按模型分派；每个进程内部SerialParam及单线程。
# 在trial或其子目录启动。Python prepare全部完成后，在R中依次执行：
# options(step7.mode = "self-test")
# source("step7/code/01_run_fgsea_parallel.R", encoding = "UTF-8")
# options(step7.mode = "run")
# source("step7/code/01_run_fgsea_parallel.R", encoding = "UTF-8")
# 若需重算，只能显式options(step7.overwrite=TRUE)。不要并发启动两个入口。

local({
  here <- getwd()
  repeat {
    common <- file.path(here, "step7/code/_step7_common.R")
    if (file.exists(common)) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请从trial或其子目录启动R。")
    here <- parent
  }
  source(common, local = parent.env(environment()), encoding = "UTF-8")
})

STEP7_RUNNER <- "step7/code/01_run_fgsea_parallel.R"
STEP7_CODE <- c(STEP7_RUNNER, "step7/code/_step7_common.R", "step5/code/_step5_common.R")
STEP7_WORKERS <- 12L

step7_task_config <- function(model, prepared, reference) {
  list(model_number = model, parameters = reference$parameters, prepared_record_sha256 = sha256(prepared$record_path),
       reference_record_sha256 = sha256(reference$record_path), code_sha256 = hash_files(STEP7_CODE),
       environment_lock_sha256 = sha256("step7/environment/r_environment_lock.json"),
       RNG = "L'Ecuyer-CMRG/Inversion/Rejection", backend = "BiocParallel::SerialParam", workers = STEP7_WORKERS)
}

step7_run_model <- function(model, reference, overwrite) {
  prepared <- step7_prepared(model, reference)
  cfg <- step7_task_config(model, prepared, reference)
  qc_path <- sprintf("step7/quality_control/fgsea/model_%02d.json", model)
  out <- sprintf("step7/gsea_results/model_%02d.csv.gz", model)
  warn <- sprintf("step7/gsea_results/model_%02d_warnings.csv", model)
  incomplete <- FALSE
  if (file.exists(qc_path) && !overwrite) {
    previous <- read_json(qc_path)
    ensure(isTRUE(all.equal(previous$configuration, cfg, tolerance = 0, check.attributes = FALSE)), "R输入或配置改变，不能直接续跑。")
    if (identical(previous$status, "complete")) {
      check_record(qc_path, sources = TRUE)
      return(list(model_number = model, status = "skipped"))
    }
    ensure(previous$status %in% c("running", "failed"), "未知任务状态。")
    incomplete <- TRUE  # 仅重算未完整提交的模型，已完成模型继续保留。
  }
  ensure(overwrite || incomplete || !any(file.exists(c(out, warn))), "存在无完成记录的R输出。")
  sources <- hash_files(c(STEP7_CODE, prepared$record_path, names(prepared$record$output_sha256),
    reference$record_path, names(reference$record$output_sha256), "step7/environment/r_environment_lock.json"))
  started <- proc.time()[["elapsed"]]
  write_json(qc_path, list(status = "running", configuration = cfg, worker_pid = Sys.getpid()))
  tryCatch({
    tables <- vector("list", 64L)
    notices <- list()
    for (i in seq_len(64L)) {
      fitted <- step7_fit(prepared$values[i, ], prepared$jobs[i, , drop = FALSE], reference)
      tables[[i]] <- fitted$table
      if (length(fitted$warnings)) notices[[length(notices) + 1L]] <- data.frame(
        model_number = model, unit_id = prepared$jobs$unit_id[i], message = fitted$warnings)
      message(sprintf("%s | model_%02d / %s (%d/64)", Sys.time(), model, prepared$jobs$unit_id[i], i))
    }
    table <- do.call(rbind, tables)
    warning_table <- if (length(notices)) do.call(rbind, notices) else data.frame(model_number = integer(), unit_id = character(), message = character())
    write_table(table, out)
    write_table(warning_table, warn)
    write_json(qc_path, list(status = "complete", configuration = cfg, model_number = model,
      n_rankings = 64L, n_tests = nrow(table), n_unavailable_pvalues = sum(!is.finite(table$pval)),
      environment_lock_sha256 = sha256("step7/environment/r_environment_lock.json"),
      execution = list(workers = STEP7_WORKERS, worker_pid = Sys.getpid(), inner_backend = "SerialParam",
                       threads_per_worker = 1L, elapsed_seconds = proc.time()[["elapsed"]] - started),
      source_sha256 = sources, output_sha256 = hash_files(c(out, warn))))
    list(model_number = model, status = "complete")
  }, error = function(e) {
    write_json(qc_path, list(status = "failed", configuration = cfg, error = conditionMessage(e)))
    stop(e)
  })
}

step7_start_workers <- function(log_dir) {
  thread_vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
  do.call(Sys.setenv, as.list(setNames(rep("1", length(thread_vars)), thread_vars)))
  step7_output(paste0(log_dir, "/startup.log"))
  cluster <- parallel::makePSOCKcluster(STEP7_WORKERS, outfile = paste0(log_dir, "/startup.log"))
  ready <- FALSE
  on.exit(if (!ready) parallel::stopCluster(cluster), add = TRUE)
  workers <- parallel::clusterCall(cluster, function(root, log_dir) {
    setwd(root)
    # 在source含中文的文件之前设置编码；子进程不继承主进程的locale。
    if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
    options(step7.load_only = TRUE, mc.cores = 1L)
    source("step7/code/01_run_fgsea_parallel.R", local = .GlobalEnv, encoding = "UTF-8")
    step7_environment()
    data.table::setDTthreads(1L)
    path <- sprintf("%s/worker_%d.log", log_dir, Sys.getpid())
    connection <- file(path, open = "wt", encoding = "UTF-8")
    assign("step7_worker_log", connection, envir = .GlobalEnv)
    sink(connection, type = "output")
    sink(connection, type = "message")
    save_session(sprintf("%s/sessionInfo_%d.txt", log_dir, Sys.getpid()))
    list(pid = Sys.getpid(), log = path, data_table_threads = data.table::getDTthreads(), omp_threads = Sys.getenv("OMP_NUM_THREADS"))
  }, getwd(), log_dir)
  ensure(length(unique(vapply(workers, function(x) x$pid, integer(1)))) == STEP7_WORKERS, "工作进程数量错误。")
  ready <- TRUE
  list(cluster = cluster, workers = workers)
}

step7_parallel_self_test <- function(pool, reference) {
  # 12条真实权重排序向量仅写QA目录；正式结果由run独立生成。
  prepared <- step7_prepared(1L, reference)
  tasks <- lapply(1:12, function(i) list(scores = prepared$values[i, ], job = prepared$jobs[i, , drop = FALSE]))
  evaluate <- function(task, reference) step7_fit(task$scores, task$job, reference)
  serial <- lapply(tasks, evaluate, reference = reference)
  concurrent <- parallel::parLapply(pool$cluster, tasks, evaluate, reference = reference)
  ensure(identical(serial, concurrent), "串行与12进程结果不一致。")
  ensure(all(vapply(pool$workers, function(w) w$data_table_threads == 1L && w$omp_threads == "1", logical(1))), "内部线程限制未生效。")
  # 用与最终绘图相同的权重准备方式核验53集合的ES，不使用未取整权重代替。
  ranked <- rank_scores(tasks[[1]]$scores, reference$genes)
  expected <- vapply(reference$pathways, function(m) step7_curve(ranked, m)$ES, numeric(1))
  actual <- serial[[1]]$table$ES[match(names(expected), serial[[1]]$table$reference_tissue_id)]
  ensure(all(is.finite(actual)) && max(abs(actual - expected)) < 1e-8, "完整排序曲线ES自检失败。")
  folder <- "step7/quality_control/parallel_self_test"
  write_table(do.call(rbind, lapply(concurrent, function(x) x$table)), paste0(folder, "/real_input_comparison.csv"))
  write_json(paste0(folder, "/report.json"), list(status = "passed", workers = STEP7_WORKERS,
    n_rankings = 12L, n_tests = 636L, exact_serial_parallel_match = TRUE, max_ES_difference = max(abs(actual - expected)),
    source_sha256 = hash_files(c(STEP7_CODE, prepared$record_path, reference$record_path, "step7/environment/r_environment_lock.json"))))
  message("12进程检查通过。")
}

step7_main <- function() {
  step7_environment()
  mode <- getOption("step7.mode", "self-test")
  ensure(mode %in% c("self-test", "run"), "step7.mode只能为self-test或run。")
  first <- as.integer(getOption("step7.model_start", 1L))
  last <- as.integer(getOption("step7.model_end", 20L))
  ensure(first >= 1L && last <= 20L && first <= last, "模型范围必须在1-20。")
  reference <- step7_reference()
  if (mode == "run") {
    test <- read_json("step7/quality_control/parallel_self_test/report.json")
    ensure(test$status == "passed" && test$workers == 12L, "请先运行当前配置的self-test。")
    check_hashes(test$source_sha256)
  }
  log_dir <- paste0("step7/logs/R_", mode, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_", Sys.getpid())
  pool <- step7_start_workers(log_dir)
  on.exit(parallel::stopCluster(pool$cluster), add = TRUE)
  write_json(paste0(log_dir, "/workers.json"), list(workers = pool$workers))
  save_session(paste0(log_dir, "/sessionInfo_main.txt"))
  message("12个工作进程已启动，逐神经元进度见：", log_dir)
  if (mode == "self-test") {
    step7_parallel_self_test(pool, reference)
  } else {
    started <- proc.time()[["elapsed"]]
    completed <- parallel::parLapplyLB(pool$cluster, seq.int(first, last), step7_run_model,
      reference = reference, overwrite = isTRUE(getOption("step7.overwrite", FALSE)), chunk.size = 1L)
    write_json(paste0(log_dir, "/completion.json"), list(status = "complete", models = completed,
      elapsed_seconds = proc.time()[["elapsed"]] - started))
    message("所选模型已完成。20模型全部完成后运行Python finalize、display_muscle.py，再运行03_plot_muscle_enrichment.R。")
  }
}

# PSOCK子进程只加载函数，避免递归启动进程。
if (!isTRUE(getOption("step7.load_only", FALSE))) step7_main()

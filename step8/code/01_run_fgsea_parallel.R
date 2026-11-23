# Step8 GSEA：12个外层进程；内部 SerialParam，避免线程嵌套。
# 在 trial 中手动运行；先 options(step8.experiment="bn", step8.mode="self-test")，
# 然后 source 本文件。自检通过后将 mode 改为 "run"。
# 已完成且指纹一致的条件/模型自动跳过，不会重算基线或改变 Step1–7。
if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
local({
  here <- getwd()
  repeat {
    if (file.exists(file.path(here, "step8/code/_step8_common.R"))) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请从 trial 或其子目录启动 R。")
    here <- parent
  }
  setwd(here)
  source("step8/code/_step8_common.R", local = parent.env(environment()), encoding = "UTF-8")
})

STEP8_WORKERS <- 12L
STEP8_CODE <- c("step8/code/01_run_fgsea_parallel.R", "step8/code/_step8_common.R", "step5/code/_step5_common.R")

step8_run_task <- function(task, design) {
  prepared <- step8_prepared(task, design)
  cid <- task$condition_id
  number <- task$model_number
  cfg <- list(condition_id = cid, model_number = number, parameters = design$parameters,
    prepared_sha256 = sha256(prepared$report_path), design_sha256 = sha256("step8/metadata/design.json"),
    code_sha256 = hash_files(STEP8_CODE), lock_sha256 = sha256("step8/environment/r_environment_lock.json"),
    RNG = "L'Ecuyer-CMRG/Inversion/Rejection", backend = "SerialParam", workers = STEP8_WORKERS)
  report_path <- sprintf("step8/quality_control/fgsea/%s/model_%02d.json", cid, number)
  if (file.exists(report_path)) {
    previous <- read_json(report_path)
    ensure(isTRUE(all.equal(previous$configuration, cfg, tolerance = 0, check.attributes = FALSE)),
           "GSEA来源、代码或软件锁改变；不能将新旧结果混合。")
    if (identical(previous$status, "complete")) {
      check_record(report_path, sources = TRUE)
      return(list(condition_id = cid, model_number = number, status = "skipped"))
    }
  }
  output <- sprintf("step8/gsea_results/%s/model_%02d.csv.gz", cid, number)
  notices <- sprintf("step8/gsea_results/%s/model_%02d_warnings.csv", cid, number)
  sources <- hash_files(c(STEP8_CODE, "step8/metadata/design.json", prepared$report_path,
    names(prepared$report$output_sha256), prepared$reference$paths, "step8/environment/r_environment_lock.json"))
  started <- proc.time()[["elapsed"]]
  write_json(report_path, list(status = "running", configuration = cfg, worker_pid = Sys.getpid()))
  tryCatch({
    tables <- vector("list", 64L)
    warning_rows <- list()
    for (i in seq_len(64L)) {
      fitted <- step8_fit(prepared$values[i, ], prepared$jobs[i, , drop = FALSE], prepared$reference, design$parameters)
      tables[[i]] <- fitted$table
      if (length(fitted$warnings)) warning_rows[[length(warning_rows)+1L]] <- data.frame(
        condition_id = cid, model_number = number, unit_id = prepared$jobs$unit_id[i], message = fitted$warnings)
      message(sprintf("%s | %s/model_%02d/%s", Sys.time(), cid, number, prepared$jobs$unit_id[i]))
    }
    table <- do.call(rbind, tables)
    warn <- if (length(warning_rows)) do.call(rbind, warning_rows) else data.frame(
      condition_id = character(), model_number = integer(), unit_id = character(), message = character())
    write_table(table, output)
    write_table(warn, notices)
    write_json(report_path, list(status = "complete", configuration = cfg, n_tests = nrow(table), n_rankings = 64L,
      source_sha256 = sources, output_sha256 = hash_files(c(output, notices)),
      worker_pid = Sys.getpid(), elapsed_seconds = proc.time()[["elapsed"]]-started))
    list(condition_id = cid, model_number = number, status = "complete")
  }, error = function(e) {
    write_json(report_path, list(status = "failed", configuration = cfg, error = conditionMessage(e)))
    stop(e)
  })
}

step8_pool <- function(log_dir) {
  vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
  do.call(Sys.setenv, as.list(setNames(rep("1", length(vars)), vars)))
  step8_output(paste0(log_dir, "/startup.log"))
  cluster <- parallel::makePSOCKcluster(STEP8_WORKERS, outfile = paste0(log_dir, "/startup.log"))
  ready <- FALSE
  on.exit(if (!ready) parallel::stopCluster(cluster), add = TRUE)
  workers <- parallel::clusterCall(cluster, function(root, log_dir) {
    setwd(root)
    if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
    options(step8.load_only = TRUE)
    source("step8/code/01_run_fgsea_parallel.R", local = .GlobalEnv, encoding = "UTF-8")
    step8_environment()
    connection <- file(sprintf("%s/worker_%d.log", log_dir, Sys.getpid()), "wt", encoding = "UTF-8")
    assign("step8_worker_log", connection, envir = .GlobalEnv)
    sink(connection, type = "output"); sink(connection, type = "message")
    save_session(sprintf("%s/sessionInfo_%d.txt", log_dir, Sys.getpid()))
    list(pid = Sys.getpid(), threads = data.table::getDTthreads())
  }, getwd(), log_dir)
  ensure(length(unique(vapply(workers, function(x) x$pid, integer(1)))) == 12L, "外层进程不是12个。")
  ready <- TRUE
  list(cluster = cluster, workers = workers)
}

step8_parallel_test <- function(pool, task, design, experiment) {
  prepared <- step8_prepared(task, design)
  inputs <- lapply(1:12, function(i) list(scores = prepared$values[i, ], job = prepared$jobs[i, , drop = FALSE]))
  run <- function(task, reference, parameters) step8_fit(task$scores, task$job, reference, parameters)
  serial <- lapply(inputs, run, reference = prepared$reference, parameters = design$parameters)
  concurrent <- parallel::parLapply(pool$cluster, inputs, run, reference = prepared$reference, parameters = design$parameters)
  ensure(identical(serial, concurrent), "串行和12进程GSEA不一致。")
  ensure(all(vapply(pool$workers, function(w) w$threads == 1L, logical(1))), "内部线程限制未生效。")
  write_json(sprintf("step8/quality_control/parallel_self_test_%s.json", experiment),
    list(status = "passed", experiment = experiment, n_rankings = 12L, workers = 12L, exact_match = TRUE,
      source_sha256 = hash_files(c(STEP8_CODE, "step8/metadata/design.json", prepared$report_path,
                                   "step8/environment/r_environment_lock.json"))))
  message("12条真实排序的串行/并行一致性检查通过。")
}

step8_main <- function() {
  step8_environment()
  check_record("step8/quality_control/setup.json")
  design <- read_json("step8/metadata/design.json")
  experiment <- getOption("step8.experiment", "bn")
  mode <- getOption("step8.mode", "self-test")
  ensure(experiment %in% names(design$experiments) && mode %in% c("self-test", "run"), "子实验或运行模式错误。")
  ids <- setdiff(unlist(design$experiments[[experiment]], use.names = FALSE), "baseline")
  first <- as.integer(getOption("step8.model_start", 1L))
  last <- as.integer(getOption("step8.model_end", 20L))
  ensure(first >= 1L && last <= 20L && first <= last, "模型范围必须在1..20。")
  tasks <- unlist(lapply(ids, function(cid) lapply(seq.int(first, last), function(m)
    list(condition_id = cid, model_number = m))), recursive = FALSE)
  # 启动12个进程前检查准备范围；self-test只需要第一个模型的真实评分。
  needed <- if (mode == "self-test") tasks[1L] else tasks
  for (task in needed) {
    path <- sprintf("step8/quality_control/prepared/%s/model_%02d.json", task$condition_id, task$model_number)
    ensure(file.exists(path) && identical(read_json(path)$status, "complete"), paste("请先完成 prepare：", path))
  }
  if (mode == "run") {
    test <- read_json(sprintf("step8/quality_control/parallel_self_test_%s.json", experiment))
    ensure(identical(test$status, "passed"), "先执行本子实验的 self-test。")
    check_hashes(test$source_sha256)
  }
  # 同一时刻仅允许一个Step8 R主进程。目录创建具有排他性，防止两个会话覆盖结果。
  # 正常结束/报错均清理；硬终止后请先确认主进程及worker已结束，再移除该目录。
  lock <- "step8/control/fgsea.lock"
  dir.create("step8/control", recursive = TRUE, showWarnings = FALSE)
  ensure(dir.create(lock, showWarnings = FALSE),
    "Step8 GSEA锁已存在；请勿并发启动。若此前硬终止，确认所有R工作进程结束后再移除step8/control/fgsea.lock。")
  lock_owner <- paste0(lock, "/owner.json")
  on.exit({
    # 只清理本进程取得的、位置正确的空锁目录，不递归清理未知内容。
    expected <- paste0(normalizePath("step8/control", winslash = "/"), "/fgsea.lock")
    if (identical(normalizePath(lock, winslash = "/", mustWork = FALSE), expected)) {
      if (file.exists(lock_owner)) file.remove(lock_owner)
      if (length(list.files(lock, all.files = TRUE, no.. = TRUE)) == 0L) unlink(lock, recursive = TRUE)
    }
  }, add = TRUE)
  write_json(lock_owner, list(pid = Sys.getpid(), experiment = experiment, mode = mode,
                             started = as.character(Sys.time())))
  log_dir <- paste0("step8/logs/R_", experiment, "_", mode, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_", Sys.getpid())
  pool <- step8_pool(log_dir)
  on.exit(parallel::stopCluster(pool$cluster), add = TRUE, after = FALSE)
  write_json(paste0(log_dir, "/workers.json"), list(workers = pool$workers))
  save_session(paste0(log_dir, "/sessionInfo_main.txt"))
  if (mode == "self-test") {
    step8_parallel_test(pool, tasks[[1]], design, experiment)
  } else {
    completed <- parallel::parLapplyLB(pool$cluster, tasks, step8_run_task, design = design, chunk.size = 1L)
    write_json(paste0(log_dir, "/completion.json"), list(status = "complete", tasks = completed))
    message("GSEA已完成。返回 Python Notebook 运行 finalize。")
  }
}

if (!isTRUE(getOption("step8.load_only", FALSE))) step8_main()

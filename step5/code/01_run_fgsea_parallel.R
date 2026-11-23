if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# Step5：12个R工作进程，按模型×组织分派任务；每次fgsea仍使用SerialParam。
# 使用前先停止另一个正在运行的GSEA任务，避免两个入口同时写同一组织。
# 工作目录为trial时，在R控制台执行：
# options(step5.mode="self-test")
# source("step5/code/01_run_fgsea_parallel.R", encoding="UTF-8")
# 检查通过后设置options(step5.mode="run")，再source本文件。
# 计算参数、候选范围、排序、派生种子和结果表列保持一致。

local({
  here <- getwd()
  repeat {
    common <- file.path(here, "step5/code/_step5_common.R")
    if (file.exists(common)) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请先将R工作目录设为trial或其子目录。")
    here <- parent
  }
  # 函数同时供主进程和PSOCK工作进程使用，故载入当前source所在环境。
  source(common, local = parent.env(environment()), encoding = "UTF-8")
})

parallel_parameters <- function() {
  list(minSize = 15, maxSize = 500, eps = 0, scoreType = "std", gseaParam = 1,
       sampleSize = 101, nPermSimple = 1000)
}

parallel_script <- "step5/code/01_run_fgsea_parallel.R"
WORKER_COUNT <- 12L

analysis_signature <- function() {
  # 数值配置、软件版本和本入口的代码共同确定签名，只续接同一配置。
  digest::digest(list(parallel_parameters(), sha256("step5/environment/r_environment_lock.json"),
    sha256("step5/code/_step5_common.R"), sha256(parallel_script)), algo = "sha256")
}

task_configuration <- function(model, tissue_id, prepared, reference) {
  list(model_number = model, source_tissue_id = tissue_id, parameters = parallel_parameters(),
    analysis_signature = analysis_signature(), prepared_record_sha256 = sha256(prepared$record_path),
    reference_record_sha256 = sha256(reference$record_path), RNG = "L'Ecuyer-CMRG/Inversion/Rejection",
    backend = "BiocParallel::SerialParam",
    environment_lock_sha256 = sha256("step5/environment/r_environment_lock.json"))
}

check_task_state <- function(qc_path, configuration, overwrite = FALSE) {
  # 已完成项必须先验证输入、输出和数值配置；不改写其完成记录或结果。
  if (!file.exists(qc_path) || overwrite) return("compute")
  old <- read_json(qc_path)
  ensure(old$status %in% c("complete", "running"), paste("未知组织状态：", qc_path))
  ensure(isTRUE(all.equal(old$configuration, configuration, tolerance = 0, check.attributes = FALSE)),
         paste("输入、软件或数值配置已变化，不能直接续跑：", qc_path))
  if (identical(old$status, "complete")) {
    check_record(qc_path, sources = TRUE)
    return("skip")
  }
  # 状态running表示该组织没有完整提交；重算整个组织，不重算已完成组织。
  # 调用者必须先停止先前的GSEA运行，此状态本身不是进程存活检查。
  "restart_incomplete"
}

fit_one_job <- function(scores, job, reference) {
  # 每个任务独立调用fgsea_one；不引入外层随机数流分配。
  ranked <- rank_scores(scores, reference$genes)
  if (all(ranked == 0)) {
    result <- data.frame(pathway = reference$sets$reference_tissue_id, pval = NA_real_, padj = NA_real_,
      log2err = NA_real_, ES = NA_real_, NES = NA_real_, size = reference$sets$n_matched,
      leadingEdge = "", estimate_status = "all_zero_scores")
    warning_text <- "All ranking scores are zero; no GSEA estimate."
  } else {
    fitted <- fgsea_one(ranked, reference$pathways, job$seed, parallel_parameters())
    result <- fitted$table
    ensure(nrow(result) == 53L && !anyDuplicated(result$pathway) &&
             setequal(result$pathway, names(reference$pathways)), "fgsea结果集合不完整。")
    result <- result[match(reference$sets$reference_tissue_id, result$pathway), , drop = FALSE]
    ensure(all(result$size == reference$sets$n_matched), "fgsea集合大小不匹配。")
    result$leadingEdge <- vapply(result$leadingEdge, paste, character(1), collapse = ";")
    result$estimate_status <- ifelse(!is.finite(result$ES), "es_unavailable",
      ifelse(!is.finite(result$NES), "nes_unavailable", ifelse(!is.finite(result$pval), "pvalue_unavailable", "ok")))
    for (column in c("ES", "NES", "pval", "padj", "log2err")) {
      ensure(!any(is.infinite(result[[column]])), paste("fgsea返回无穷大：", column))
      result[[column]][is.nan(result[[column]])] <- NA_real_
    }
    for (column in c("pval", "padj")) {
      ensure(all(is.na(result[[column]]) | (result[[column]] >= 0 & result[[column]] <= 1)), "p或q越界。")
    }
    warning_text <- fitted$warnings
  }
  result$model_number <- job$model_number
  result$source_tissue_id <- job$source_tissue_id
  result$source_tissue <- job$source_tissue
  result$neuron_index <- job$neuron_index
  result$unit_id <- job$unit_id
  result$seed <- job$seed
  result$reference_tissue_id <- result$pathway
  result$reference_tissue <- reference$sets$reference_tissue
  result$pathway <- NULL
  result <- result[, c("model_number", "source_tissue_id", "source_tissue", "neuron_index", "unit_id", "seed",
    "reference_tissue_id", "reference_tissue", "ES", "NES", "pval", "padj", "size", "log2err", "leadingEdge", "estimate_status")]
  list(table = result, warnings = warning_text)
}

run_tissue_parallel <- function(task, reference, overwrite) {
  model <- task$model_number
  tissue_id <- task$source_tissue_id
  started <- proc.time()[["elapsed"]]
  prepared <- load_prepared(model, tissue_id, reference)
  jobs <- prepared$jobs
  out_path <- sprintf("step5/gsea_results/model_%02d/%s.csv.gz", model, tissue_id)
  warnings_path <- sprintf("step5/gsea_results/model_%02d/%s_warnings.csv", model, tissue_id)
  qc_path <- sprintf("step5/quality_control/fgsea/model_%02d/%s.json", model, tissue_id)
  configuration <- task_configuration(model, tissue_id, prepared, reference)
  state <- check_task_state(qc_path, configuration, overwrite)
  if (identical(state, "skip")) {
    message(sprintf("[%s] Skip completed: model_%02d / %s", Sys.time(), model, tissue_id))
    return(list(model_number = model, source_tissue_id = tissue_id, status = "skipped"))
  }
  ensure(overwrite || identical(state, "restart_incomplete") ||
    (!file.exists(out_path) && !file.exists(warnings_path)), "存在无完成记录的输出，请先核对来源。")
  sources <- hash_files(unique(c(prepared$record_path, reference$record_path,
    names(prepared$record$output_sha256), names(reference$record$output_sha256),
    "step5/environment/r_environment_lock.json", "step5/code/_step5_common.R", parallel_script)))
  write_json(qc_path, list(status = "running", configuration = configuration, worker_pid = Sys.getpid()))
  output <- vector("list", nrow(jobs))
  warnings <- list()
  for (i in seq_len(nrow(jobs))) {
    job <- jobs[i, , drop = FALSE]
    fitted <- fit_one_job(prepared$values[i, ], job, reference)
    output[[i]] <- fitted$table
    if (length(fitted$warnings)) warnings[[length(warnings) + 1L]] <- data.frame(
      model_number = model, source_tissue_id = tissue_id, unit_id = job$unit_id, message = fitted$warnings)
    message(sprintf("[%s] model_%02d / %s / %s (%d/%d)",
      Sys.time(), model, tissue_id, job$unit_id, i, nrow(jobs)))
  }
  output <- do.call(rbind, output)
  warning_table <- if (length(warnings)) do.call(rbind, warnings) else data.frame(
    model_number = integer(), source_tissue_id = character(), unit_id = character(), message = character())
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(out_path, ".tmp")
  data.table::fwrite(output, temporary, na = "NA", compress = "gzip")
  if (file.exists(out_path)) ensure(file.remove(out_path), "不能替换组织结果。")
  ensure(file.rename(temporary, out_path), "不能保存组织结果。")
  write_table(warning_table, warnings_path)
  elapsed <- proc.time()[["elapsed"]] - started
  write_json(qc_path, list(status = "complete", configuration = configuration,
    model_number = model, source_tissue_id = tissue_id, n_rankings = nrow(jobs), n_tests = nrow(output),
    n_unavailable_pvalues = sum(is.na(output$pval)), n_zero_pvalues = sum(output$pval == 0, na.rm = TRUE),
    n_warning_messages = nrow(warning_table), environment_lock_sha256 = configuration$environment_lock_sha256,
    execution = list(workers = WORKER_COUNT, outer_backend = "parallel::PSOCK", inner_backend = "BiocParallel::SerialParam",
      threads_per_worker = 1L, worker_pid = Sys.getpid(), elapsed_seconds = elapsed),
    source_sha256 = sources, output_sha256 = hash_files(c(out_path, warnings_path))))
  message(sprintf("[%s] SAVED model_%02d / %s: %.1f s", Sys.time(), model, tissue_id, elapsed))
  list(model_number = model, source_tissue_id = tissue_id, status = "complete", elapsed_seconds = elapsed)
}

start_workers <- function(root, log_dir) {
  # 环境变量在创建子进程前设置；对子进程生效后恢复主会话的设置。
  variables <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "R_DATATABLE_NUM_THREADS")
  previous <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    Sys.unsetenv(variables[is.na(previous)])
    present <- !is.na(previous)
    if (any(present)) do.call(Sys.setenv, as.list(previous[present]))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(setNames(rep("1", length(variables)), variables)))
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
  cluster <- parallel::makePSOCKcluster(WORKER_COUNT, outfile = paste0(log_dir, "/startup.log"))
  initialized <- FALSE
  on.exit(if (!initialized) parallel::stopCluster(cluster), add = TRUE)
  workers <- parallel::clusterCall(cluster, function(root, log_dir, runner) {
    Sys.setlocale("LC_CTYPE", ".UTF-8")
    setwd(root)
    .libPaths(c("step5/environment/library", .libPaths()))
    options(step5.parallel_load_only = TRUE, mc.cores = 1L)
    source(runner, local = .GlobalEnv, encoding = "UTF-8")
    check_environment()
    data.table::setDTthreads(1L)
    # 每个工作进程有独立日志和sessionInfo，不并发覆盖一个共享文件。
    log_path <- sprintf("%s/worker_%d.log", log_dir, Sys.getpid())
    connection <- file(log_path, open = "wt", encoding = "UTF-8")
    assign("step5_worker_log_connection", connection, envir = .GlobalEnv)
    sink(connection, type = "output")
    sink(connection, type = "message")
    save_session(sprintf("%s/sessionInfo_%d.txt", log_dir, Sys.getpid()))
    list(pid = Sys.getpid(), log = log_path, data_table_threads = data.table::getDTthreads(),
         omp_threads = Sys.getenv("OMP_NUM_THREADS"))
  }, root, log_dir, parallel_script)
  ensure(length(unique(vapply(workers, function(x) x$pid, integer(1)))) == WORKER_COUNT,
         sprintf("没有建立%d个独立进程。", WORKER_COUNT))
  initialized <- TRUE
  list(cluster = cluster, workers = workers)
}

parallel_self_test <- function(cluster, workers, reference) {
  # 真实输入小批检查：12个神经元各自的53集合，覆盖实际完整基因排序。
  # 输出只写QA目录，不改正式GSEA结果。每个任务仍由其固定seed独立控制。
  folder <- "step5/quality_control/parallel_self_test"
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  prepared <- load_prepared(1L, "tissue_01", reference)
  tasks <- lapply(seq_len(WORKER_COUNT), function(i) list(scores = prepared$values[i, ], job = prepared$jobs[i, , drop = FALSE]))
  evaluate <- function(task, reference) fit_one_job(task$scores, task$job, reference)
  serial_time <- system.time(serial <- lapply(tasks, evaluate, reference = reference))[["elapsed"]]
  parallel_time <- system.time(concurrent <- parallel::parLapply(cluster, tasks, evaluate, reference = reference))[["elapsed"]]
  ensure(identical(serial, concurrent), "串行与12进程结果不一致，不能启动正式并行分析。")
  # 数值相同之外，还检查每个进程确实只启用一个内部线程。
  ensure(all(vapply(workers, function(w) w$data_table_threads == 1L && w$omp_threads == "1", logical(1))),
         "工作进程的内部线程限制未生效。")
  # 独立的小型ES检查不依赖任何已保存GSEA结果。
  ranked <- setNames(seq(4, -4, length.out = 120L), sprintf("QA_G%03d", 1:120))
  pathways <- list(positive = names(ranked)[1:20], negative = names(ranked)[101:120],
                   middle = names(ranked)[45:64])
  fitted <- fgsea_one(ranked, pathways, 5000629L, parallel_parameters())$table
  expected <- vapply(pathways[fitted$pathway], function(x) running_es(ranked, x), numeric(1))
  ensure(max(abs(fitted$ES - expected)) < 1e-10 &&
    fitted$NES[fitted$pathway == "positive"] > 0 && fitted$NES[fitted$pathway == "negative"] < 0,
    "合成向量的ES或富集方向检查失败。")
  # 完整记录、未完成记录以及参数变化的分支均仅在QA文件中检查。
  config <- task_configuration(1L, "tissue_01", prepared, reference)
  qc_path <- paste0(folder, "/resume_fixture.json")
  fixture <- paste0(folder, "/fixture.txt")
  writeLines("QA only", fixture)
  write_json(qc_path, list(status = "complete", configuration = config,
    source_sha256 = hash_files(parallel_script), output_sha256 = hash_files(fixture)))
  ensure(check_task_state(qc_path, config) == "skip", "完成项续跑检查失败。")
  write_json(qc_path, list(status = "running", configuration = config))
  ensure(check_task_state(qc_path, config) == "restart_incomplete", "未完成组织恢复检查失败。")
  changed <- config
  changed$parameters$eps <- 1e-10
  rejected <- tryCatch({check_task_state(qc_path, changed); FALSE}, error = function(e) TRUE)
  ensure(rejected, "错误地接受了变化的GSEA参数。")
  result <- do.call(rbind, lapply(concurrent, function(x) x$table))
  write_table(result, paste0(folder, "/real_input_comparison.csv"))
  write_json(paste0(folder, "/report.json"), list(status = "passed", workers = WORKER_COUNT, n_rankings = length(tasks),
    n_tests = nrow(result), exact_serial_parallel_match = TRUE, independent_ES_and_direction_check = TRUE,
    resume_branches = TRUE, parameter_mismatch_rejected = TRUE, worker_details = workers,
    serial_seconds = serial_time, parallel_seconds = parallel_time,
    timing_note = "Small-batch timings exclude process startup; not a full-run speed guarantee.",
    environment_lock_sha256 = sha256("step5/environment/r_environment_lock.json"),
    source_sha256 = hash_files(c(parallel_script, "step5/code/_step5_common.R"))))
  message(sprintf("%d进程检查通过；%d条排序向量：串行%.1f秒，并行%.1f秒。",
                  WORKER_COUNT, length(tasks), serial_time, parallel_time))
}

step5_parallel_main <- function() {
  step5_root()
  check_environment()
  mode <- getOption("step5.mode", "self-test")
  first <- as.integer(getOption("step5.model_start", 1L))
  last <- as.integer(getOption("step5.model_end", 20L))
  overwrite <- isTRUE(getOption("step5.overwrite", FALSE))
  ensure(mode %in% c("self-test", "run"), "MODE只能为self-test或run。")
  ensure(first >= 1L && last <= 20L && first <= last, "模型范围必须在1至20。")
  reference <- load_reference()
  if (mode == "run") {
    test <- read_json("step5/quality_control/parallel_self_test/report.json")
    ensure(identical(test$status, "passed") && test$workers == WORKER_COUNT &&
      identical(test$environment_lock_sha256, sha256("step5/environment/r_environment_lock.json")),
      "请先运行当前版本的12进程self-test。")
    check_hashes(test$source_sha256)
  }
  # 主进程只调度；只有12个工作进程进行GSEA计算。
  stamp <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_", Sys.getpid())
  log_dir <- paste0("step5/logs/parallel_", mode, "_", stamp)
  pool <- start_workers(getwd(), log_dir)
  on.exit(parallel::stopCluster(pool$cluster), add = TRUE)
  message("已启动", WORKER_COUNT, "个R工作进程；每进程内部串行。日志目录：", log_dir)
  write_json(paste0(log_dir, "/workers.json"), list(workers = pool$workers, mode = mode))
  if (mode == "self-test") {
    parallel_self_test(pool$cluster, pool$workers, reference)
  } else {
    tissues <- read_table("step5/reference/tissue_map.csv")
    task_table <- expand.grid(source_tissue_id = tissues$tissue_id, model_number = seq.int(first, last),
                              stringsAsFactors = FALSE)
    tasks <- lapply(seq_len(nrow(task_table)), function(i) as.list(task_table[i, ]))
    message("按模型×组织动态分派；控制台等待期间，可查看各worker日志中的逐神经元进度。")
    started <- proc.time()[["elapsed"]]
    # chunk.size=1确保一个组织一项任务；快进程完成后立即领取下一组织。
    completed <- parallel::parLapplyLB(pool$cluster, tasks, run_tissue_parallel,
      reference = reference, overwrite = overwrite, chunk.size = 1L)
    write_json(paste0(log_dir, "/completion.json"), list(status = "complete", workers = WORKER_COUNT,
      model_start = first, model_end = last, elapsed_seconds = proc.time()[["elapsed"]] - started,
      tissues = completed))
    message("所选模型范围计算完成。全部20模型完成后，再运行Notebook的finalize。")
  }
}

# 子进程只载入函数，不递归创建进程。正常source会进入主入口。
if (!isTRUE(getOption("step5.parallel_load_only", FALSE))) step5_parallel_main()

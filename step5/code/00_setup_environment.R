if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# Step5 R环境：由用户手动运行一次，后续只核对锁定版本。
# 在trial工作目录执行：source("step5/code/00_setup_environment.R", encoding="UTF-8")
# 不运行GSEA。首次运行只安装缺失包；已有版本锁时不自动安装、升级或改锁。

local({
  # 仅靠当前工作目录向上寻找trial，支持RStudio在step5子目录启动。
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
  ensure(identical(as.character(getRversion()), "4.6.1"), "本步骤固定使用R 4.6.1。")
  dir.create("step5/environment/library", recursive = TRUE, showWarnings = FALSE)
  .libPaths(c("step5/environment/library", .libPaths()))
  lock_path <- "step5/environment/r_environment_lock.json"

  if (file.exists(lock_path)) {
    check_environment()
    save_session("step5/environment/sessionInfo_check.txt")
    message("版本锁核对通过；未修改已锁定版本。")
  } else {
    # 不猜测fgsea版本号：安装与当前R兼容的正式包，首次分析前冻结实际版本。
    options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = max(600, getOption("timeout")))
    required_cran <- c("jsonlite", "digest", "data.table", "ggplot2", "BiocManager")
    missing <- required_cran[!vapply(required_cran, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing)) utils::install.packages(missing, lib = "step5/environment/library")
    ensure(requireNamespace("BiocManager", quietly = TRUE), "BiocManager安装失败，请检查R控制台提示后重试。")
    required_bioc <- c("fgsea", "BiocParallel")
    missing <- required_bioc[!vapply(required_bioc, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing)) BiocManager::install(missing, lib = "step5/environment/library", ask = FALSE, update = FALSE)
    required <- unique(c(required_cran, required_bioc))
    ensure(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)), "仍有缺失R包，未写入版本锁。")
    expected_arguments <- c("pathways", "stats", "minSize", "maxSize", "eps", "scoreType", "gseaParam",
                            "sampleSize", "nPermSimple", "BPPARAM")
    ensure(all(expected_arguments %in% names(formals(fgsea::fgseaMultilevel))), "fgseaMultilevel接口不符合本步骤配置。")
    # 冻结直接依赖与递归依赖版本；系统自带R包的版本也保留。
    installed <- utils::installed.packages()
    dependencies <- tools::package_dependencies(required, db = installed,
      which = c("Depends", "Imports", "LinkingTo"), recursive = TRUE)
    packages <- sort(unique(c(required, unlist(dependencies, use.names = FALSE))))
    packages <- setdiff(packages, "R")
    versions <- setNames(lapply(packages, function(x) as.character(utils::packageVersion(x))), packages)
    lock <- list(schema_version = 1L, R = as.character(getRversion()), platform = R.version$platform,
      packages = versions, library = "step5/environment/library",
      bioconductor_version = as.character(BiocManager::version()),
      created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      policy = "freeze before first analysis; no automatic version changes")
    write_json(lock_path, lock)
    save_session("step5/environment/sessionInfo_setup.txt")
    check_environment()
    message("已冻结R/fgsea及依赖版本。下一步手动运行01_run_fgsea_parallel.R的self-test模式。")
    message("R = ", lock$R, "; fgsea = ", lock$packages$fgsea)
  }
})

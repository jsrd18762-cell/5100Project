if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# Step7：固定六组织的两路线富集曲线。正式绘图需Python finalize完成。
# source("step7/code/02_plot_enrichment.R", encoding="UTF-8")
# 只做合成版面检查：options(step7.plot_qa=TRUE)，结果仅写quality_control。

local({
  here <- getwd()
  repeat {
    common <- file.path(here, "step7/code/_step7_common.R")
    if (file.exists(common)) break
    parent <- dirname(here)
    if (identical(parent, here)) stop("请从trial或其子目录启动R。")
    here <- parent
  }
  source(common, local = environment(), encoding = "UTF-8")
  step7_environment()
  reference <- step7_reference()
  selection <- read_table("step7/reference/display_selection.csv")
  ensure(nrow(selection) == 6L && all(selection$model_number == 1L) && all(selection$neuron_index == 0L), "展示对象改变。")
  selection <- selection[order(selection$rank_in_group, match(selection$group, c("high", "low"))), , drop = FALSE]
  qa <- isTRUE(getOption("step7.plot_qa", FALSE))
  sources <- c("step7/code/02_plot_enrichment.R", "step7/code/_step7_common.R", "step5/code/_step5_common.R",
               reference$record_path, names(reference$record$output_sha256), "step7/environment/r_environment_lock.json")
  if (!qa) {
    final_path <- "step7/quality_control/finalize.json"
    final <- check_record(final_path, sources = TRUE)
    jac_final <- check_record("step5/quality_control/finalize.json")
    prepared <- step7_prepared(1L, reference)
    ranked_weight <- rank_scores(prepared$values[1, ], reference$genes)
    weight_table <- data.table::fread("step7/statistics/matched_gsea.csv", data.table = FALSE, na.strings = "NA")
    jac_table <- data.table::fread("step5/statistics/display_gsea.csv", data.table = FALSE, na.strings = "NA")
    sources <- c(sources, final_path, names(final$output_sha256), prepared$record_path,
                 names(prepared$record$output_sha256), "step5/quality_control/finalize.json", "step5/statistics/display_gsea.csv")
  }
  plots <- vector("list", 6L)
  checks <- list()
  colors <- c(Jacobian = "#b53e32", Weight = "#316a96")
  for (i in 1:6) {
    task <- selection[i, , drop = FALSE]
    if (qa) {
      # 合成输入的目的仅为测试曲线和版面，不充当任何正式统计结果。
      genes <- sprintf("QA_G%03d", 1:120)
      score <- seq(4, -4, length.out = 120)
      ranked_jac <- setNames(score, genes)
      ranked_weight <- setNames(score, genes[c(11:120, 1:10)])
      members <- genes[if (task$group == "high") 1:20 else 101:120]
      curves <- list(Jacobian = step7_curve(ranked_jac, members), Weight = step7_curve(ranked_weight, members))
      saved <- vapply(curves, function(x) x$ES, numeric(1))
    } else {
      jac_path <- sprintf("step5/quality_control/prepared/model_01/%s.json", task$source_tissue_id)
      jac <- check_record(jac_path)
      ensure(identical(jac$source_sha256[["step5/quality_control/reference.json"]],
                       reference$record$source_sha256[["step5/quality_control/reference.json"]]), "绘图两路线参考库不同。")
      folder <- sprintf("step5/rank_inputs/model_01/%s", task$source_tissue_id)
      jobs <- read_table(paste0(folder, "/jobs.csv"))
      values <- read_scores(paste0(folder, "/scores.f64.bin"), 64L, 58581L)
      index <- match(0L, jobs$neuron_index)
      ensure(!is.na(index), "Jacobian展示神经元缺失。")
      ranked_jac <- rank_scores(values[index, ], reference$genes)
      members <- reference$pathways[[task$source_tissue_id]]
      curves <- list(Jacobian = step7_curve(ranked_jac, members), Weight = step7_curve(ranked_weight, members))
      j <- jac_table[jac_table$model_number == 1L & jac_table$neuron_index == 0L & jac_table$source_tissue_id == task$source_tissue_id, ]
      w <- weight_table[weight_table$model_number == 1L & weight_table$neuron_index == 0L & weight_table$activation_tissue_id == task$source_tissue_id, ]
      ensure(nrow(j) == 1L && nrow(w) == 1L, "展示结果必须唯一。")
      saved <- c(Jacobian = j$ES, Weight = w$ES)
      sources <- c(sources, jac_path, names(jac$output_sha256))
    }
    lines <- list()
    ticks <- list()
    for (method in names(curves)) {
      difference <- abs(curves[[method]]$ES - saved[[method]])
      ensure(is.finite(difference) && difference < 1e-8, paste("曲线ES与计算结果不一致：", task$source_tissue_id, method))
      checks[[length(checks) + 1L]] <- list(model_number = 1L, neuron_index = 0L, tissue_id = task$source_tissue_id,
        score_source = method, curve_ES = curves[[method]]$ES, saved_ES = saved[[method]], absolute_difference = difference)
      lines[[method]] <- transform(curves[[method]]$curve, method = method)
      ticks[[method]] <- transform(curves[[method]]$ticks, method = method, lane = if (method == "Jacobian") -1.04 else -1.10)
    }
    line_table <- do.call(rbind, lines)
    tick_table <- do.call(rbind, ticks)
    title <- paste(strwrap(task$source_tissue, width = 40), collapse = "\n")
    subtitle <- sprintf("%s activation #%d | median %.3g\nES: Jacobian %.3f | Weight %.3f",
                        task$group, task$rank_in_group, task$median_activation, saved[["Jacobian"]], saved[["Weight"]])
    plots[[i]] <- ggplot2::ggplot(line_table, ggplot2::aes(x = rank, y = ES, colour = method)) +
      ggplot2::geom_hline(yintercept = 0, colour = "grey70", linewidth = .25) +
      ggplot2::geom_line(linewidth = .55) +
      ggplot2::geom_segment(data = tick_table, inherit.aes = FALSE,
        ggplot2::aes(x = rank, xend = rank, y = lane, yend = lane + .035, colour = method), linewidth = .18) +
      ggplot2::scale_colour_manual(values = colors) +
      ggplot2::coord_cartesian(ylim = c(-1.12, 1.02)) +
      ggplot2::labs(title = title, subtitle = subtitle, x = "Rank in full gene list", y = "Running ES", colour = NULL) +
      ggplot2::theme_classic(base_size = 10) +
      ggplot2::theme(plot.title = ggplot2::element_text(size = 11), plot.subtitle = ggplot2::element_text(size = 9),
                     legend.position = "top", legend.margin = ggplot2::margin(0, 0, 0, 0))
  }
  base <- if (qa) "step7/quality_control/plot_qa/enrichment_comparison" else "step7/figures/model_01/unit_00000_enrichment_comparison"
  paths <- if (qa) paste0(base, ".png") else paste0(base, c(".pdf", ".png"))
  ensure(qa || isTRUE(getOption("step7.overwrite", FALSE)) || !any(file.exists(paths)), "图形已存在，需显式step7.overwrite=TRUE。")
  for (path in paths) {
    step7_output(path)
    if (endsWith(path, ".pdf")) grDevices::pdf(path, width = 11, height = 12, useDingbats = FALSE)
    else grDevices::png(path, width = 2200, height = 2400, res = 200)
    tryCatch({
      grid::grid.newpage()
      grid::grid.text(if (qa) "QA SYNTHETIC" else "Tissue enrichment: Jacobian and weight propagation",
                      y = .98, gp = grid::gpar(fontsize = 14))
      grid::pushViewport(grid::viewport(y = .49, height = .92, layout = grid::grid.layout(3, 2)))
      for (i in 1:6) print(plots[[i]], newpage = FALSE, vp = grid::viewport(layout.pos.row = ceiling(i/2), layout.pos.col = 2L - i %% 2L))
      grid::popViewport()
      grid::grid.text("Two gene-hit tracks follow their respective rankings; all genes enter GSEA.",
                      y = .012, gp = grid::gpar(fontsize = 9))
    }, finally = grDevices::dev.off())
  }
  save_session(if (qa) "step7/quality_control/plot_qa/sessionInfo.txt" else "step7/environment/sessionInfo_plots.txt")
  write_json(if (qa) "step7/quality_control/plot_qa/report.json" else "step7/quality_control/enrichment_plots.json",
    list(status = if (qa) "qa_passed" else "complete", synthetic = qa, model_number = 1L, neuron_index = 0L,
         n_tissues = 6L, n_curves = 12L, es_checks = checks, source_sha256 = hash_files(sources), output_sha256 = hash_files(paths)))
  message("两路线富集曲线已写入step7。")
})

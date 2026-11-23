# Python记录中的固定展示对象：六组织Jacobian曲线及两路线对照。
# 先运行display_muscle.py，再从trial或其子目录手动source本文件。
# 只读取已保存评分和GSEA结果，不运行fgseaMultilevel，不更改统计表。
# options(muscle_display.validate_only=TRUE)只核对12条ES，不写文件。
if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")

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
  # 权重路径改变后不能仅刷新旧曲线；当前全模型统计必须先完成重算。
  step7_propagation_design(read_json("step7/reference/analysis_design.json"))
  final <- read_json("step7/quality_control/finalize.json")
  ensure(identical(final$status, "complete"), "请先完成Step7 finalize。")
  step7_propagation_design(final$configuration$design)
  check_hashes(final$configuration$code_sha256)
  folder <- "step7/display_muscle"
  record_path <- paste0(folder, "/quality_control/python.json")
  record <- check_record(record_path, sources = TRUE)
  selection <- read_table(paste0(folder, "/statistics/display_selection.csv"))
  ensure(nrow(selection) == 6L && !anyDuplicated(selection$source_tissue_id) &&
           all(selection$model_number == record$model_number) &&
           all(selection$neuron_index == record$neuron_index), "六组织与展示对象不一致。")
  # 左列为高激活H1-H3，右列为低激活L1-L3；两路线始终共用该顺序。
  selection <- selection[order(selection$rank_in_group, match(selection$group, c("high", "low"))), ]
  inputs <- read_table(paste0(folder, "/statistics/score_inputs.csv"))
  saved <- data.table::fread(paste0(folder, "/statistics/display_gsea.csv"), data.table = FALSE, na.strings = "NA")
  genes <- read_table("step5/reference/gene_map.csv")$gene_name
  pathways <- lapply(read_json("step5/reference/pathways.json"), unlist, use.names = FALSE)
  sources <- c(record_path, names(record$source_sha256), names(record$output_sha256),
    "step7/code/03_plot_muscle_enrichment.R", "step7/code/_step7_common.R", "step5/code/_step5_common.R",
    "step5/environment/r_environment_lock.json", "step7/environment/r_environment_lock.json")
  colors <- c(Jacobian = "#b53e32", Weight = "#316a96")
  curves <- vector("list", 6L)
  checks <- list()
  for (i in seq_len(nrow(selection))) {
    task <- selection[i, ]
    members <- pathways[[task$source_tissue_id]]
    curves[[i]] <- list()
    # 无参考集合时保持该组织的NA面板，不用别的组织替代。
    if (is.null(members)) next
    for (route in names(colors)) {
      tissue <- if (route == "Weight") "all_tissues" else task$source_tissue_id
      input <- inputs[inputs$route == route & inputs$source_tissue_id == tissue, ]
      result <- saved[saved$route == route & saved$source_tissue_id == task$source_tissue_id, ]
      ensure(nrow(input) == 1L && nrow(result) == 1L && input$unit_id == record$unit_id &&
               input$model_number == record$model_number && input$n_genes == length(genes), "评分/ES映射不唯一。")
      ranked <- rank_scores(read_scores(input$path, 1L, length(genes))[1, ], genes)
      # 复用已验证的fgsea1.38.0权重缩放与取整函数；保留带符号排序。
      curve <- step7_curve(ranked, members)
      difference <- abs(curve$ES - result$ES)
      ensure(is.finite(difference) && difference < 1e-8, paste("曲线ES不一致：", task$source_tissue_id, route))
      checks[[length(checks) + 1L]] <- list(model_number = record$model_number, unit_id = record$unit_id,
        tissue_id = task$source_tissue_id, route = route, curve_ES = curve$ES, saved_ES = result$ES,
        absolute_difference = difference)
      curves[[i]][[route]] <- curve
    }
  }
  ensure(length(checks) == 2L * sum(selection$source_tissue_id %in% names(pathways)), "ES核验条数不完整。")
  if (isTRUE(getOption("muscle_display.validate_only", FALSE))) {
    message(length(checks), "条ES核验通过；只读检查，未写图形。")
  } else {
    qc_path <- paste0(folder, "/quality_control/enrichment.json")
    write_json(qc_path, list(status = "running"))
    output_paths <- character()
    for (comparison in c(FALSE, TRUE)) {
      routes <- if (comparison) names(colors) else "Jacobian"
      plots <- vector("list", 6L)
      for (i in seq_len(6L)) {
        task <- selection[i, ]
        title <- paste(strwrap(task$source_tissue, width = 38), collapse = "\n")
        if (length(curves[[i]]) == 0L) {
          plots[[i]] <- ggplot2::ggplot() + ggplot2::theme_void() + ggplot2::labs(title = title) +
            ggplot2::annotate("text", x = 0, y = 0, label = "NA: no same-tissue reference")
          next
        }
        lines <- do.call(rbind, lapply(routes, function(route) transform(curves[[i]][[route]]$curve, method = route)))
        ticks <- do.call(rbind, lapply(routes, function(route) transform(curves[[i]][[route]]$ticks,
            method = route, lane = if (route == "Jacobian") -1.04 else -1.10)))
        results <- saved[saved$source_tissue_id == task$source_tissue_id, ]
        es_text <- paste(vapply(routes, function(route) sprintf("%s ES %.3f", route,
          results$ES[results$route == route]), character(1)), collapse = " | ")
        subtitle <- sprintf("%s activation #%d | median %.3f | n = %d\n%s",
          task$group, task$rank_in_group, task$median_activation, task$n_samples, es_text)
        plots[[i]] <- ggplot2::ggplot(lines, ggplot2::aes(x = rank, y = ES, colour = method)) +
          ggplot2::geom_hline(yintercept = 0, colour = "grey70", linewidth = .25) +
          ggplot2::geom_line(linewidth = .55) +
          ggplot2::geom_segment(data = ticks, inherit.aes = FALSE,
            ggplot2::aes(x = rank, xend = rank, y = lane, yend = lane + .035, colour = method), linewidth = .18) +
          ggplot2::scale_colour_manual(values = colors) + ggplot2::coord_cartesian(ylim = c(-1.12, 1.02)) +
          ggplot2::labs(title = title, subtitle = subtitle, x = "Rank in full gene list", y = "Running ES", colour = NULL) +
          ggplot2::theme_classic(base_size = 10) +
          ggplot2::theme(plot.title = ggplot2::element_text(size = 11),
            plot.subtitle = ggplot2::element_text(size = 9), legend.position = if (comparison) "top" else "none",
            legend.margin = ggplot2::margin(0, 0, 0, 0))
      }
      suffix <- if (comparison) "enrichment_comparison" else "jacobian_enrichment"
      paths <- paste0(record$figure_prefix, "_", suffix, c(".pdf", ".png"))
      # 本目录只存当前展示；重跑可安全替换同名图，不触及计算结果目录。
      for (path in paths) {
        step7_output(path)
        if (endsWith(path, ".pdf")) grDevices::pdf(path, width = 11, height = 12, useDingbats = FALSE)
        else grDevices::png(path, width = 2200, height = 2400, res = 200)
        tryCatch({
          grid::grid.newpage()
          grid::grid.text(if (comparison) "Tissue enrichment: Jacobian and weight propagation" else "Tissue enrichment: Jacobian",
            y = .98, gp = grid::gpar(fontsize = 14))
          grid::pushViewport(grid::viewport(y = .49, height = .92, layout = grid::grid.layout(3, 2)))
          for (i in seq_len(6L)) print(plots[[i]], newpage = FALSE,
            vp = grid::viewport(layout.pos.row = ceiling(i / 2), layout.pos.col = 2L - i %% 2L))
          grid::popViewport()
          grid::grid.text("Full gene rankings; same-tissue Top100 reference sets; signed ES.", y = .012, gp = grid::gpar(fontsize = 9))
        }, finally = grDevices::dev.off())
      }
      output_paths <- c(output_paths, paths)
    }
    session_path <- paste0(folder, "/quality_control/sessionInfo_plots.txt")
    save_session(session_path)
    write_json(qc_path, list(status = "complete", model_number = record$model_number, unit_id = record$unit_id,
      n_tissues = 6L, n_curves = length(checks), es_checks = checks,
      source_sha256 = hash_files(sources), output_sha256 = hash_files(c(output_paths, session_path))))
    message("骨骼肌最高激活对象的富集曲线已写入", folder, "/figures。")
  }
})

if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", ".UTF-8")
# Step5富集曲线；在全部R分析及Python finalize完成后手动运行。
# source("step5/code/02_plot_enrichment.R", encoding="UTF-8")
# 每个选定展示神经元绘制激活中位数最高、最低的前三个组织，共六个面板。
# 每个面板均为该来源组织对同组织参考集合的完整排序富集，不使用Top200替代排序。

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
  check_environment()

  enrichment_curve_data <- function(ranked, members) {
    # fgsea 1.38.0的prepareStats先保留带符号评分的降序，再把绝对权重
    # 缩放、取整。GSEA使用这些权重，但plotEnrichment使用未取整权重。
    # 在当前版本锁下调用相同的准备函数，使曲线与已保存ES使用相同权重。
    # 取整权重不再排序，避免把基因顺序变成绝对评分的顺序。
    preparation_warnings <- character()
    weights <- withCallingHandlers(
      getFromNamespace("prepareStats", "fgsea")(ranked, scoreType = "std", gseaParam = 1),
      warning = function(w) {
        preparation_warnings <<- c(preparation_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    ensure(identical(names(weights), names(ranked)), "fgsea权重准备改变了基因顺序。")
    positions <- which(names(weights) %in% members)
    ensure(length(positions) == length(members), "曲线的参考成员与基因空间不一致。")
    score <- fgsea::calcGseaStat(weights, positions, gseaParam = 1,
      returnAllExtremes = TRUE, scoreType = "std")
    # 独立累积和用于核验公式，不以放宽ES阈值绕过数值差异。
    independent_es <- running_es(weights, members)
    ensure(abs(independent_es - score$res) < 1e-10, "加权累计曲线与fgsea ES公式不一致。")
    curve <- data.frame(rank = c(0, as.vector(rbind(positions - 1L, positions)), length(weights)),
                        ES = c(0, as.vector(rbind(score$bottoms, score$tops)), 0))
    list(curve = curve, ticks = data.frame(rank = positions), ES = score$res,
         posES = max(score$tops), negES = min(score$bottoms),
         spreadES = max(score$tops) - min(score$bottoms),
         unrounded_ES = running_es(ranked, members), independent_ES = independent_es,
         preparation_warnings = unique(preparation_warnings))
  }

  reference <- load_reference()
  final_path <- "step5/quality_control/finalize.json"
  final <- check_record(final_path, sources = TRUE)
  selection <- read_table("step5/reference/display_selection.csv")
  table <- data.table::fread("step5/statistics/display_gsea.csv", data.table = FALSE, na.strings = "NA")
  ensure(nrow(table) == nrow(selection), "展示结果与预先选定的组织数量不匹配。")
  outputs <- character()
  es_checks <- list()
  input_paths <- c(final_path, names(final$output_sha256), reference$record_path,
    names(reference$record$output_sha256), "step5/environment/r_environment_lock.json",
    "step5/code/_step5_common.R", "step5/code/02_plot_enrichment.R")
  overwrite <- isTRUE(getOption("step5.overwrite", FALSE))

  # 同一神经元的high1/low1、high2/low2、high3/low3并排，避免跨面板错读。
  for (unit in sort(unique(selection$neuron_index))) {
    selected <- selection[selection$neuron_index == unit, , drop = FALSE]
    selected <- selected[order(selected$rank_in_group, match(selected$group, c("high", "low"))), , drop = FALSE]
    ensure(nrow(selected) == 6L && all(selected$model_number == 1L), "展示选择必须是model_01的六个组织。")
    plots <- vector("list", 6L)
    for (i in 1:6) {
      task <- selected[i, , drop = FALSE]
      row <- table[table$model_number == task$model_number & table$neuron_index == unit &
                     table$source_tissue_id == task$source_tissue_id, , drop = FALSE]
      ensure(nrow(row) == 1L, "展示结果不唯一。")
      prepared <- load_prepared(task$model_number, task$source_tissue_id, reference)
      input_paths <- unique(c(input_paths, prepared$record_path, names(prepared$record$output_sha256)))
      index <- match(unit, prepared$jobs$neuron_index)
      ensure(!is.na(index), "展示神经元缺少评分向量。")
      ranked <- rank_scores(prepared$values[index, ], reference$genes)
      members <- reference$pathways[[task$source_tissue_id]]
      title <- paste(strwrap(task$source_tissue, width = 52), collapse = "\n")
      label <- sprintf("%s activation #%d | median = %.4g", toupper(task$group), task$rank_in_group, task$median_activation)
      if (is.null(members) || !is.finite(row$ES)) {
        # 缺少集合或ES时保留所选组织，明确写NA，不临时替换另一个组织。
        reason <- if (is.null(members)) "No matching reference set (NA)" else "ES unavailable (NA)"
        plots[[i]] <- ggplot2::ggplot() + ggplot2::annotate("text", x = 0, y = 0, label = reason, size = 4) +
          ggplot2::labs(title = title, subtitle = label) + ggplot2::theme_void(base_size = 11)
      } else {
        curve <- enrichment_curve_data(ranked, members)
        difference <- curve$ES - row$ES
        ensure(abs(difference) < 1e-8,
          sprintf("%s / %s：绘图ES=%.17g，保存ES=%.17g，差值=%+.3g。",
                  task$source_tissue_id, task$unit_id, curve$ES, row$ES, difference))
        es_checks[[length(es_checks) + 1L]] <- list(model_number = task$model_number,
          neuron_index = unit, source_tissue_id = task$source_tissue_id,
          reference_tissue_id = row$reference_tissue_id, saved_ES = row$ES,
          curve_ES = curve$ES, independent_ES = curve$independent_ES,
          unrounded_ES = curve$unrounded_ES, absolute_difference = abs(difference),
          preparation_warnings = curve$preparation_warnings)
        q_label <- if (is.finite(row$q_matched_global)) format(row$q_matched_global, digits = 3) else "NA"
        subtitle <- paste0(label, "\nES = ", format(row$ES, digits = 4), " | NES = ",
          format(row$NES, digits = 4), " | global q = ", q_label)
        # 按命中位置及其前后累计值绘制阶跃点之间的折线；背景基因对应线性下降。
        plots[[i]] <- ggplot2::ggplot(curve$curve, ggplot2::aes(x = rank, y = ES)) +
          ggplot2::geom_line(color = "green") +
          ggplot2::geom_segment(data = curve$ticks, inherit.aes = FALSE,
            mapping = ggplot2::aes(x = rank, xend = rank),
            y = -curve$spreadES / 16, yend = curve$spreadES / 16, linewidth = .2) +
          ggplot2::geom_hline(yintercept = c(curve$posES, curve$negES), color = "red", linetype = "dashed") +
          ggplot2::geom_hline(yintercept = 0, color = "black") +
          ggplot2::labs(title = title, subtitle = subtitle,
            x = "Rank in full gene list (descending signed mean gradient)", y = "Running enrichment score") +
          ggplot2::theme_bw(base_size = 10) +
          ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", size = 11),
            plot.subtitle = ggplot2::element_text(size = 9), panel.grid.minor = ggplot2::element_blank())
      }
    }
    folder <- "step5/figures/model_01"
    dir.create(folder, recursive = TRUE, showWarnings = FALSE)
    base <- sprintf("%s/unit_%05d_enrichment", folder, unit)
    paths <- paste0(base, c(".pdf", ".png"))
    ensure(overwrite || !any(file.exists(paths)), "富集图已存在，明确设置options(step5.overwrite=TRUE)后可覆盖。")
    render <- function(path, pdf) {
      if (pdf) grDevices::pdf(path, width = 15, height = 12, useDingbats = FALSE)
      else grDevices::png(path, width = 2700, height = 2160, res = 180)
      on.exit(grDevices::dev.off(), add = TRUE)
      grid::grid.newpage()
      grid::grid.text(sprintf("model_01 / unit_%05d | source tissue = reference tissue", unit),
        y = .983, gp = grid::gpar(fontsize = 16, fontface = "bold"))
      grid::pushViewport(grid::viewport(y = .482, height = .92, layout = grid::grid.layout(3, 2)))
      for (i in 1:6) print(plots[[i]], newpage = FALSE, vp = grid::viewport(
        layout.pos.row = ceiling(i / 2), layout.pos.col = ifelse(i %% 2 == 1, 1, 2)))
      grid::popViewport()
      grid::grid.text("Tissues selected by activation median; curves use all 58,581 ranked genes.",
        y = .012, gp = grid::gpar(fontsize = 10))
    }
    render(paths[1], TRUE)
    render(paths[2], FALSE)
    outputs <- c(outputs, paths)
  }
  session_path <- "step5/environment/sessionInfo_plots.txt"
  save_session(session_path)
  write_json("step5/quality_control/enrichment_plots.json", list(status = "complete",
    environment_lock_sha256 = sha256("step5/environment/r_environment_lock.json"),
    curve_weight_method = "fgsea::prepareStats from the locked namespace; preserve signed-score gene order",
    fgsea_version = as.character(utils::packageVersion("fgsea")),
    es_absolute_tolerance = 1e-8, es_checks = es_checks,
    source_sha256 = hash_files(input_paths), output_sha256 = hash_files(c(outputs, session_path))))
  message("富集曲线已保存至step5/figures/model_01。")
})

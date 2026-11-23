"""
Step 1 - compare candidate log transforms for the GTEx expression matrix.

Purpose
-------
The source expression file supplied for this reproduction contains values that
match the original project's stored ``log2(TPM + 0.001)`` scale.  The original
training code reverses that transform and then applies ``log2(TPM + 1)``.  This
script makes that assumption explicit and compares four offsets on the same
underlying approximate TPM values:

    log2(TPM + 0.001)   (the stored/input scale)
    log2(TPM + 0.01)
    log2(TPM + 0.1)
    log2(TPM + 1)

The full 3.7 GB text matrix is read in row chunks.  Only a deterministic sample
of genes from each chunk is retained for distribution summaries and figures;
the script therefore does not claim that the plotted distributions are exact
full-matrix distributions.  The sampling seed is 629, as required for this
reproduction.  The expression matrix itself is never modified by this script.

Run from the ``trial`` directory, for example:

    python step1/code/compare_log_offsets.py

All default paths below are relative paths from the ``trial`` directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reproduction settings
# ---------------------------------------------------------------------------
# These values are deliberately kept in one place so that the notebook and the
# command-line execution use exactly the same settings.
DEFAULT_SEED = 629
DEFAULT_EXPRESSION_PATH = Path("dataset/gtex_RSEM_gene_tpm")
DEFAULT_RESULTS_DIR = Path("step1/results")
DEFAULT_FIGURES_DIR = Path("step1/figures")
DEFAULT_LOGS_DIR = Path("step1/logs")

# A row chunk contains a manageable number of genes while retaining all 7,862
# samples.  Four evenly spaced genes per chunk provide coverage across the file
# without loading the complete matrix into memory.
DEFAULT_CHUNK_ROWS = 256
DEFAULT_SAMPLE_ROWS_PER_CHUNK = 4

# Matplotlib does not need millions of points to estimate a histogram.  This
# cap keeps plotting and boxplot generation responsive while preserving a
# deterministic subset of the sampled values.
DEFAULT_MAX_PLOT_VALUES = 300_000

# Insertion order is intentional: it controls the order in CSV files and plots.
OFFSETS: Mapping[float, str] = {
    0.001: "log2_TPM_plus_0.001",
    0.01: "log2_TPM_plus_0.01",
    0.1: "log2_TPM_plus_0.1",
    1.0: "log2_TPM_plus_1",
}


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure a file-and-console logger for a reproducible run."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step1_log_offset_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def read_expression_header(expression_path: Path) -> List[str]:
    """Read and validate the sample IDs without loading expression values."""

    if not expression_path.exists():
        raise FileNotFoundError(f"Expression file not found: {expression_path}")

    with expression_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")

    fields = first_line.split("\t")
    if len(fields) < 2:
        raise ValueError("The expression header must contain a gene column and samples.")
    if fields[0].strip().lower() != "sample":
        raise ValueError(
            f"Expected the first expression header field to be 'sample', got {fields[0]!r}."
        )

    sample_ids = fields[1:]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("The expression header contains an empty sample ID.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("The expression header contains duplicate sample IDs.")
    return sample_ids


def iter_expression_chunks(
    expression_path: Path, chunk_rows: int
) -> Iterable[pd.DataFrame]:
    """Yield gene-row chunks while keeping sample columns in their source order."""

    # The first column is a string gene identifier; all remaining columns are
    # numeric expression values.  The dtype mapping is explicit because a
    # scalar ``dtype=np.float32`` would also be applied to the gene-ID column by
    # some pandas versions.  Reading chunks avoids a several-gigabyte in-memory
    # DataFrame.
    header = pd.read_csv(expression_path, sep="\t", nrows=0).columns.tolist()
    value_dtype = {column: np.float32 for column in header[1:]}
    return pd.read_csv(
        expression_path,
        sep="\t",
        index_col=0,
        dtype=value_dtype,
        chunksize=chunk_rows,
    )


def select_evenly_spaced_rows(
    chunk: pd.DataFrame, rows_to_keep: int
) -> pd.DataFrame:
    """Select deterministic rows spanning one expression chunk."""

    if rows_to_keep <= 0:
        raise ValueError("rows_to_keep must be positive")
    if len(chunk) <= rows_to_keep:
        return chunk

    indices = np.linspace(0, len(chunk) - 1, rows_to_keep, dtype=int)
    return chunk.iloc[np.unique(indices)]


def recover_approximate_tpm(stored_values: np.ndarray) -> np.ndarray:
    """Recover approximate TPM from stored ``log2(TPM + 0.001)`` values.

    The small non-negative clip protects against tiny negative values caused by
    decimal rounding in the text file.  It does not change valid TPM values.
    """

    stored_float64 = np.asarray(stored_values, dtype=np.float64)
    approximate_tpm = np.exp2(stored_float64) - 0.001
    return np.maximum(approximate_tpm, 0.0)


def apply_log_transform(approximate_tpm: np.ndarray, offset: float) -> np.ndarray:
    """Apply the base-2 log transform used by the original Python code."""

    if offset <= 0:
        raise ValueError("A log offset must be greater than zero.")
    return np.log2(np.asarray(approximate_tpm, dtype=np.float64) + offset)


def thin_values(values: np.ndarray, max_values: int) -> np.ndarray:
    """Deterministically reduce a value vector for plotting."""

    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    if len(finite) <= max_values:
        return finite

    indices = np.linspace(0, len(finite) - 1, max_values, dtype=int)
    return finite[np.unique(indices)]


def summarize_values(
    series_name: str,
    formula: str,
    offset: float | None,
    values: np.ndarray,
) -> Dict[str, float | int | str | None]:
    """Return distribution statistics for one sampled series."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError(f"No finite values were available for {series_name}.")

    quantiles = np.quantile(finite, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "series": series_name,
        "formula": formula,
        "offset": offset,
        "n_sampled_values": int(len(finite)),
        "minimum": float(np.min(finite)),
        "q01": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
    }


def _finite_plot_limits(series: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Choose shared robust x limits so long tails do not hide the center."""

    joined = np.concatenate(
        [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in series]
    )
    low, high = np.quantile(joined, [0.001, 0.999])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(joined)), float(np.max(joined))
    padding = max((high - low) * 0.05, 1e-6)
    return float(low - padding), float(high + padding)


def create_figures(
    sampled_values: Mapping[str, np.ndarray],
    figures_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Create density and boxplot figures analogous to the report's Figure 1."""

    # Import only when figures are requested.  This keeps metadata-only uses of
    # the script independent of Matplotlib, while providing a useful error when
    # the user asks for plots without the dependency installed.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Matplotlib is required to create figures. Install it with "
            "'python -m pip install matplotlib'."
        ) from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    input_name = "stored_log2_TPM_plus_0.001"
    ordered_series = [input_name, *OFFSETS.values()]
    colors = {
        input_name: "#4C8A8A",
        "log2_TPM_plus_0.001": "#D95F5F",
        "log2_TPM_plus_0.01": "#E79F3F",
        "log2_TPM_plus_0.1": "#5B8CCB",
        "log2_TPM_plus_1": "#6A51A3",
    }
    display_names = {
        input_name: "Stored input: log2(TPM + 0.001)",
        "log2_TPM_plus_0.001": "Candidate: log2(TPM + 0.001)",
        "log2_TPM_plus_0.01": "Candidate: log2(TPM + 0.01)",
        "log2_TPM_plus_0.1": "Candidate: log2(TPM + 0.1)",
        "log2_TPM_plus_1": "Candidate: log2(TPM + 1)",
    }

    # One row per candidate offset, with the stored input as the reference.
    before_after_path = figures_dir / "before_after_by_offset.png"
    fig, axes = plt.subplots(len(OFFSETS), 2, figsize=(14, 18), squeeze=False)
    reference = sampled_values[input_name]
    for row_index, candidate_name in enumerate(OFFSETS.values()):
        candidate = sampled_values[candidate_name]
        x_min, x_max = _finite_plot_limits([reference, candidate])

        density_ax = axes[row_index, 0]
        density_ax.hist(
            reference,
            bins=120,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[input_name],
            label=display_names[input_name],
        )
        density_ax.hist(
            candidate,
            bins=120,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[candidate_name],
            label=display_names[candidate_name],
        )
        density_ax.set_xlim(x_min, x_max)
        density_ax.set_ylabel("Density")
        density_ax.set_title(f"Distribution comparison: {candidate_name}")
        density_ax.grid(axis="y", alpha=0.25)
        density_ax.legend(fontsize=8, loc="best")

        box_ax = axes[row_index, 1]
        box_ax.boxplot(
            [reference, candidate],
            orientation="horizontal",
            tick_labels=["Stored input", "Candidate"],
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": colors[candidate_name]},
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": colors[candidate_name]},
            capprops={"color": colors[candidate_name]},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
        )
        box_ax.set_xlim(x_min, x_max)
        box_ax.set_xlabel("Transformed expression value")
        box_ax.set_title(f"Boxplot comparison: {candidate_name}")
        box_ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "GTEx expression preprocessing: stored input versus candidate log offsets",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(before_after_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overview density plot: all candidates against the stored input.
    density_path = figures_dir / "density_overview.png"
    fig, ax = plt.subplots(figsize=(12, 7))
    for series_name in ordered_series:
        values = sampled_values[series_name]
        ax.hist(
            values,
            bins=140,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colors[series_name],
            label=display_names[series_name],
        )
    ax.set_xlabel("Transformed expression value")
    ax.set_ylabel("Density")
    ax.set_title("GTEx expression distribution under candidate log offsets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(density_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overview boxplot uses the same sampled values and makes the scale shift
    # between offsets directly visible.
    boxplot_path = figures_dir / "boxplot_overview.png"
    fig, ax = plt.subplots(figsize=(12, 7))
    boxplot_values = [sampled_values[name] for name in ordered_series]
    ax.boxplot(
        boxplot_values,
        tick_labels=[
            "Stored\n+0.001",
            "Candidate\n+0.001",
            "Candidate\n+0.01",
            "Candidate\n+0.1",
            "Candidate\n+1",
        ],
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#4C4C4C"},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#4C4C4C"},
        capprops={"color": "#4C4C4C"},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.2},
    )
    ax.set_ylabel("Transformed expression value")
    ax.set_title("GTEx expression boxplots under candidate log offsets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    outputs = [before_after_path, density_path, boxplot_path]
    for path in outputs:
        logger.info("Saved figure: %s", path)
    return outputs


def run_comparison(
    expression_path: Path = DEFAULT_EXPRESSION_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    seed: int = DEFAULT_SEED,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    sample_rows_per_chunk: int = DEFAULT_SAMPLE_ROWS_PER_CHUNK,
    max_plot_values: int = DEFAULT_MAX_PLOT_VALUES,
) -> pd.DataFrame:
    """Run the complete chunked comparison and return the summary table."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if sample_rows_per_chunk <= 0:
        raise ValueError("sample_rows_per_chunk must be positive")
    if max_plot_values <= 0:
        raise ValueError("max_plot_values must be positive")

    # The selected rows are deterministic, but seeding NumPy here records the
    # experiment-wide seed and protects reproducibility if sampling is extended.
    np.random.seed(seed)
    logger = configure_logging(logs_dir / "compare_log_offsets.log")
    logger.info("Starting step 1 log-offset comparison with seed=%d", seed)
    logger.info("Expression path: %s", expression_path)

    sample_ids = read_expression_header(expression_path)
    logger.info("Detected %d unique sample IDs", len(sample_ids))

    collected: Dict[str, List[np.ndarray]] = {
        "stored_log2_TPM_plus_0.001": []
    }
    collected.update({series_name: [] for series_name in OFFSETS.values()})
    gene_rows_seen = 0
    chunks_seen = 0

    for chunk in iter_expression_chunks(expression_path, chunk_rows):
        chunks_seen += 1
        if list(chunk.columns) != sample_ids:
            raise ValueError(
                "A chunk's sample columns do not match the expression header order."
            )
        if chunk.index.isna().any():
            raise ValueError("The expression matrix contains an empty gene identifier.")

        selected = select_evenly_spaced_rows(chunk, sample_rows_per_chunk)
        stored_values = selected.to_numpy(dtype=np.float64, copy=False)
        collected["stored_log2_TPM_plus_0.001"].append(stored_values.reshape(-1))

        approximate_tpm = recover_approximate_tpm(stored_values)
        for offset, series_name in OFFSETS.items():
            transformed = apply_log_transform(approximate_tpm, offset)
            collected[series_name].append(transformed.reshape(-1))
        gene_rows_seen += len(chunk)

        if chunks_seen % 20 == 0:
            logger.info("Processed %d gene rows", gene_rows_seen)

    if gene_rows_seen == 0:
        raise ValueError("No gene rows were read from the expression matrix.")

    sampled_values = {
        series_name: thin_values(np.concatenate(parts), max_plot_values)
        for series_name, parts in collected.items()
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, float | int | str | None]] = []
    summary_rows.append(
        summarize_values(
            "stored_log2_TPM_plus_0.001",
            "stored input scale: log2(TPM + 0.001)",
            0.001,
            sampled_values["stored_log2_TPM_plus_0.001"],
        )
    )
    for offset, series_name in OFFSETS.items():
        summary_rows.append(
            summarize_values(
                series_name,
                f"candidate scale: log2(TPM + {offset:g})",
                offset,
                sampled_values[series_name],
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / "log_offset_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "seed": seed,
        "expression_path": str(expression_path),
        "sample_count": len(sample_ids),
        "gene_count": gene_rows_seen,
        "chunks_processed": chunks_seen,
        "chunk_rows": chunk_rows,
        "sample_rows_per_chunk": sample_rows_per_chunk,
        "max_plot_values_per_series": max_plot_values,
        "input_scale_assumption": "stored values are log2(TPM + 0.001)",
        "inverse_transform": "approximate_TPM = maximum(exp2(stored_value) - 0.001, 0)",
        "candidate_transforms": [f"log2(TPM + {offset:g})" for offset in OFFSETS],
        "note": "Summaries and figures use deterministic sampled values, not the full matrix.",
    }
    metadata_path = results_dir / "log_offset_comparison_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Read %d gene rows across %d chunks", gene_rows_seen, chunks_seen)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved metadata: %s", metadata_path)
    create_figures(sampled_values, figures_dir, logger)
    logger.info("Step 1 log-offset comparison completed")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments while keeping all defaults relative."""

    parser = argparse.ArgumentParser(
        description="Compare log2(TPM + offset) preprocessing scales for GTEx data."
    )
    parser.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--sample-rows-per-chunk",
        type=int,
        default=DEFAULT_SAMPLE_ROWS_PER_CHUNK,
    )
    parser.add_argument(
        "--max-plot-values",
        type=int,
        default=DEFAULT_MAX_PLOT_VALUES,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    summary = run_comparison(
        expression_path=args.expression,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        logs_dir=args.logs_dir,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
        sample_rows_per_chunk=args.sample_rows_per_chunk,
        max_plot_values=args.max_plot_values,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

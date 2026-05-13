import csv
import re
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


def slugify(text):
    """Return a filesystem-safe label for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def build_output_path(output_dir, sample_name, stem, suffix, run_label=None, sample_prefix=None):
    """Create a consistent output path for notebook exports."""
    prefix = slugify(sample_prefix or sample_name).replace("_", "")
    parts = [prefix]
    if run_label:
        parts.append(slugify(run_label))
    parts.append(slugify(stem))
    return Path(output_dir) / f"{'__'.join(parts)}{suffix}"


def save_current_figure(output_dir, sample_name, stem, run_label=None, sample_prefix=None):
    """Save the current matplotlib figure as a PDF."""
    path = build_output_path(
        output_dir=output_dir,
        sample_name=sample_name,
        sample_prefix=sample_prefix,
        stem=stem,
        suffix=".pdf",
        run_label=run_label,
    )
    plt.gcf().savefig(path, format="pdf", bbox_inches="tight")
    return path


def save_dataframe(df, output_dir, sample_name, stem, index=False, run_label=None, sample_prefix=None):
    """Save a DataFrame as CSV and return the output path."""
    path = build_output_path(
        output_dir=output_dir,
        sample_name=sample_name,
        sample_prefix=sample_prefix,
        stem=stem,
        suffix=".csv",
        run_label=run_label,
    )
    df.to_csv(path, index=index)
    return path


def chr_to_num(chrom):
    """Convert chromosome labels like 'chr1' or 'X' into sortable integers."""
    c = str(chrom).strip().replace("chr", "").replace("CHR", "")
    if c in ("X", "x"):
        return 23
    if c in ("Y", "y"):
        return 24
    try:
        n = int(c)
        if 1 <= n <= 24:
            return n
    except ValueError:
        pass
    return None


def resolve_wes_sample_id(sample_name, wes_path):
    """Resolve a user-facing sample name to a sampleID in the WES table."""
    with Path(wes_path).open(newline="") as handle:
        wes_reader = csv.DictReader(handle)
        wes_ids = sorted({row["sampleID"] for row in wes_reader})

    if sample_name in wes_ids:
        return sample_name

    patient_state = str(sample_name)
    if "_" in patient_state:
        patient, state = patient_state.split("_", 1)
        state_map = {"BL": "0", "K2": "2", "V2": "2", "OP": "OP"}
        mapped_state = state_map.get(state)
        if mapped_state is not None:
            candidate = f"UE-2971-{patient}-{mapped_state}"
            if candidate in wes_ids:
                return candidate

    raise ValueError(
        f"Could not resolve sample_name={sample_name!r} to a WES sampleID. "
        f"Example WES IDs: {wes_ids[:6]}"
    )


def load_wes_segments(path, sample_id):
    """Load WES segments for one sample into a normalized list of dictionaries."""
    rows = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sampleID"] != sample_id:
                continue
            chr_num = chr_to_num(row["chrom"])
            if chr_num is None:
                continue
            try:
                start = int(float(row["loc.start"]))
                end = int(float(row["loc.end"]))
                seg_mean = float(row["seg.mean"])
                copy_number = float(row["C"])
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            rows.append(
                {
                    "chrom": str(row["chrom"]),
                    "chr_num": chr_num,
                    "start": start,
                    "end": end,
                    "seg_mean": seg_mean,
                    "C": copy_number,
                }
            )
    rows.sort(key=lambda item: (item["chr_num"], item["start"]))
    return rows


def load_gene_cnv_values(path, value_col="CNV_value"):
    """Load gene-level CNV values from a sample-specific CSV or TSV file."""
    rows = []
    with Path(path).open(newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if ("\t" in first_line and first_line.count("\t") > first_line.count(",")) else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"gene", "chromosome", "start", "end", value_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing in situ columns: {sorted(missing)}")

        for row in reader:
            chr_num = chr_to_num(row["chromosome"])
            if chr_num is None:
                continue
            try:
                start = int(float(row["start"]))
                end = int(float(row["end"]))
                value = float(row[value_col])
            except (TypeError, ValueError, KeyError):
                continue
            if end < start:
                start, end = end, start
            rows.append(
                {
                    "gene": row["gene"],
                    "chrom": str(row["chromosome"]),
                    "chr_num": chr_num,
                    "start": start,
                    "end": end,
                    "mid": 0.5 * (start + end),
                    "value": value,
                }
            )
    rows.sort(key=lambda item: (item["chr_num"], item["mid"]))
    return rows


def load_bin_cnv_values(path, value_col="mean_cnv"):
    """Load one pre-binned CNV profile and normalize its schema."""
    rows = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"bin_index", "chromosome", value_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {Path(path).name}: {sorted(missing)}")

        for row in reader:
            chr_num = chr_to_num(row["chromosome"])
            if chr_num is None:
                continue
            try:
                bin_index = int(float(row["bin_index"]))
                value = float(row[value_col])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "chromosome": str(row["chromosome"]),
                    "chr_num": chr_num,
                    "bin_index": bin_index,
                    "value": value,
                }
            )
    rows.sort(key=lambda item: (item["chr_num"], item["bin_index"]))
    return rows


def build_genome_axis(*series_list):
    """Create one concatenated genome axis shared across genomic series."""
    chr_max_end = {}
    for series in series_list:
        for row in series:
            chr_max_end[row["chr_num"]] = max(chr_max_end.get(row["chr_num"], 0), row["end"])

    chromosomes = sorted(chr_max_end)
    offsets = {}
    running = 0
    for chrom in chromosomes:
        offsets[chrom] = running
        running += chr_max_end[chrom]

    def genome_x(chr_num, pos):
        return offsets[chr_num] + pos

    chr_midpoints = []
    for chrom in chromosomes:
        label = str(chrom) if chrom < 23 else ("X" if chrom == 23 else "Y")
        chr_midpoints.append((offsets[chrom] + chr_max_end[chrom] / 2.0, label))

    return chr_max_end, chromosomes, offsets, running, genome_x, chr_midpoints


def overlap_len(a0, a1, b0, b1):
    """Return the inclusive overlap length between two genomic intervals."""
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0, right - left + 1)


def weighted_segment_mean_for_interval(segments, start, end):
    """Average segment values over an interval using overlap length as the weight."""
    weighted_sum = 0.0
    total = 0
    for segment in segments:
        overlap = overlap_len(start, end, segment["start"], segment["end"])
        if overlap > 0:
            weighted_sum += segment["seg_mean"] * overlap
            total += overlap
    if total == 0:
        return None
    return weighted_sum / total


def build_wes_insitu_comparison_table(wes, insitu, bin_size_bp=None, min_genes_per_bin=5):
    """Build one standardized comparison table for WES versus gene-level in situ CNV values."""
    wes_by_chr = {}
    for row in wes:
        wes_by_chr.setdefault(row["chr_num"], []).append(row)

    insitu_by_chr = {}
    for row in insitu:
        insitu_by_chr.setdefault(row["chr_num"], []).append(row)

    chr_max_end, chromosomes, offsets, genome_length, genome_x, chr_midpoints = build_genome_axis(wes, insitu)
    rows = []

    if bin_size_bp is None:
        for gene in insitu:
            wes_value = weighted_segment_mean_for_interval(
                wes_by_chr.get(gene["chr_num"], []),
                int(gene["start"]),
                int(gene["end"]),
            )
            if wes_value is None:
                continue
            rows.append(
                {
                    "mode": "raw",
                    "chr_num": gene["chr_num"],
                    "start": gene["start"],
                    "end": gene["end"],
                    "x": genome_x(gene["chr_num"], gene["mid"]),
                    "wes_segmean": wes_value,
                    "insitu_value": gene["value"],
                    "n_genes": 1,
                    "gene": gene["gene"],
                }
            )
    else:
        for chrom in chromosomes:
            chr_len = chr_max_end[chrom]
            start = 1
            while start <= chr_len:
                end = min(start + bin_size_bp - 1, chr_len)
                wes_value = weighted_segment_mean_for_interval(wes_by_chr.get(chrom, []), start, end)
                insitu_values = [
                    gene["value"]
                    for gene in insitu_by_chr.get(chrom, [])
                    if start <= gene["mid"] <= end
                ]
                if wes_value is not None and len(insitu_values) >= min_genes_per_bin:
                    rows.append(
                        {
                            "mode": "binned",
                            "chr_num": chrom,
                            "start": start,
                            "end": end,
                            "x": genome_x(chrom, 0.5 * (start + end)),
                            "wes_segmean": wes_value,
                            "insitu_value": mean(insitu_values),
                            "n_genes": len(insitu_values),
                        }
                    )
                start = end + 1

    metric_df = pd.DataFrame(rows)
    axis = {
        "chr_max_end": chr_max_end,
        "chromosomes": chromosomes,
        "offsets": offsets,
        "genome_length": genome_length,
        "genome_x": genome_x,
        "chr_midpoints": chr_midpoints,
    }
    return metric_df, axis


def build_binwise_profile_comparison_table(truth_rows, pred_rows, truth_label="truth", pred_label="prediction"):
    """Inner-join two pre-binned CNV profiles and build one comparison table."""
    truth_df = pd.DataFrame(truth_rows).rename(columns={"value": "truth_value"})
    pred_df = pd.DataFrame(pred_rows).rename(columns={"value": "pred_value"})
    metric_df = truth_df.merge(
        pred_df,
        on=["chromosome", "chr_num", "bin_index"],
        how="inner",
    )
    if metric_df.empty:
        raise ValueError("No overlapping bins found between the two input CSV files.")

    metric_df = metric_df.sort_values(["chr_num", "bin_index"]).reset_index(drop=True)
    metric_df["x"] = np.arange(len(metric_df), dtype=float)

    chr_midpoints = []
    offsets = {}
    for chrom, subset in metric_df.groupby("chr_num", sort=True):
        offsets[chrom] = int(subset.index.min())
        label = str(chrom) if chrom < 23 else ("X" if chrom == 23 else "Y")
        chr_midpoints.append((0.5 * (subset.index.min() + subset.index.max()), label))

    metric_df["truth_label"] = truth_label
    metric_df["pred_label"] = pred_label
    axis = {"offsets": offsets, "chr_midpoints": chr_midpoints}
    return metric_df, axis


def safe_auc(y_true, y_score):
    """Compute ROC AUC safely for one-vs-rest labels."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_div(a, b):
    """Return a / b, or 0.0 when the denominator is zero."""
    return float(a) / float(b) if b else 0.0


def classify_symmetric(values, threshold):
    """Classify values into loss, neutral, and gain using symmetric thresholds."""
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=int)
    out[values >= threshold] = 1
    out[values <= -threshold] = -1
    return out


def select_validation_thresholds(
    metric_df,
    truth_col,
    pred_col,
    truth_threshold_name="truth_threshold",
    pred_threshold_name="pred_threshold",
    truth_min_threshold=0.005,
    pred_min_threshold=0.0001,
    max_threshold=0.20,
    min_events=5,
):
    """Select thresholds using the manuscript's two-stage quantitative validation rule."""
    truth_values = metric_df[truth_col].to_numpy(dtype=float)
    pred_values = metric_df[pred_col].to_numpy(dtype=float)

    truth_candidates = np.sort(np.unique(np.abs(truth_values).round(6)))
    pred_candidates = np.sort(np.unique(np.abs(pred_values).round(6)))

    if len(truth_candidates) > 0:
        truth_candidates = truth_candidates[
            (truth_candidates >= truth_min_threshold) & (truth_candidates <= min(max_threshold, truth_candidates.max()))
        ]
    if len(pred_candidates) > 0:
        pred_candidates = pred_candidates[
            (pred_candidates >= pred_min_threshold) & (pred_candidates <= min(max_threshold, pred_candidates.max()))
        ]

    if len(truth_candidates) == 0:
        truth_candidates = np.sort(np.unique(np.abs(truth_values)))
    if len(pred_candidates) == 0:
        pred_candidates = np.sort(np.unique(np.abs(pred_values)))

    truth_rows = []
    for threshold in truth_candidates:
        truth_labels = classify_symmetric(truth_values, float(threshold))
        n_gain = int(np.sum(truth_labels == 1))
        n_loss = int(np.sum(truth_labels == -1))
        if n_gain < min_events or n_loss < min_events:
            continue
        auc_gain = safe_auc((truth_labels == 1).astype(int), pred_values)
        auc_loss = safe_auc((truth_labels == -1).astype(int), -pred_values)
        truth_rows.append(
            {
                truth_threshold_name: float(threshold),
                "auc_gain": auc_gain,
                "auc_loss": auc_loss,
                "auc_mean": float(np.nanmean([auc_gain, auc_loss])),
                "n_truth_gain": n_gain,
                "n_truth_loss": n_loss,
            }
        )

    truth_grid = (
        pd.DataFrame(truth_rows)
        .sort_values(["auc_mean", "auc_gain", "auc_loss"], ascending=False)
        .reset_index(drop=True)
    )
    if truth_grid.empty:
        raise ValueError("No valid truth thresholds found after AUC filtering.")

    truth_thr = float(truth_grid.iloc[0][truth_threshold_name])
    truth_labels = classify_symmetric(truth_values, truth_thr)

    pred_rows = []
    for threshold in pred_candidates:
        pred_labels = classify_symmetric(pred_values, float(threshold))
        tp_gain = np.sum((truth_labels == 1) & (pred_labels == 1))
        fp_gain = np.sum((truth_labels != 1) & (pred_labels == 1))
        fn_gain = np.sum((truth_labels == 1) & (pred_labels != 1))
        tp_loss = np.sum((truth_labels == -1) & (pred_labels == -1))
        fp_loss = np.sum((truth_labels != -1) & (pred_labels == -1))
        fn_loss = np.sum((truth_labels == -1) & (pred_labels != -1))
        f1_gain = 0.0 if (2 * tp_gain + fp_gain + fn_gain) == 0 else (2 * tp_gain) / (2 * tp_gain + fp_gain + fn_gain)
        f1_loss = 0.0 if (2 * tp_loss + fp_loss + fn_loss) == 0 else (2 * tp_loss) / (2 * tp_loss + fp_loss + fn_loss)
        pred_rows.append(
            {
                pred_threshold_name: float(threshold),
                "f1_gain": float(f1_gain),
                "f1_loss": float(f1_loss),
                "f1_mean": float(np.mean([f1_gain, f1_loss])),
                "accuracy_3class": float(np.mean(pred_labels == truth_labels)),
            }
        )

    pred_grid = (
        pd.DataFrame(pred_rows)
        .sort_values(["f1_mean", "f1_gain", "f1_loss", "accuracy_3class"], ascending=False)
        .reset_index(drop=True)
    )
    pred_thr = float(pred_grid.iloc[0][pred_threshold_name])
    return truth_thr, pred_thr, truth_grid, pred_grid


def compute_validation_metrics(
    metric_df,
    truth_thr,
    pred_thr,
    truth_col,
    pred_col,
    metadata=None,
):
    """Compute summary metrics at the selected thresholds and return truth/pred labels."""
    truth_values = metric_df[truth_col].to_numpy(dtype=float)
    pred_values = metric_df[pred_col].to_numpy(dtype=float)
    truth_labels = classify_symmetric(truth_values, truth_thr)
    pred_labels = classify_symmetric(pred_values, pred_thr)

    def precision_for(cls):
        tp = np.sum((truth_labels == cls) & (pred_labels == cls))
        fp = np.sum((truth_labels != cls) & (pred_labels == cls))
        return safe_div(tp, tp + fp)

    def recall_for(cls):
        tp = np.sum((truth_labels == cls) & (pred_labels == cls))
        fn = np.sum((truth_labels == cls) & (pred_labels != cls))
        return safe_div(tp, tp + fn)

    def f1_for(cls):
        precision = precision_for(cls)
        recall = recall_for(cls)
        return safe_div(2 * precision * recall, precision + recall)

    auc_gain = safe_auc((truth_labels == 1).astype(int), pred_values)
    auc_loss = safe_auc((truth_labels == -1).astype(int), -pred_values)

    row = {
        "n_rows": int(len(metric_df)),
        "truth_threshold": float(truth_thr),
        "pred_threshold": float(pred_thr),
        "auc_gain": auc_gain,
        "auc_loss": auc_loss,
        "auc_mean": float(np.nanmean([auc_gain, auc_loss])),
        "precision_gain": precision_for(1),
        "precision_loss": precision_for(-1),
        "precision_mean": float(np.mean([precision_for(1), precision_for(-1)])),
        "recall_gain": recall_for(1),
        "recall_loss": recall_for(-1),
        "recall_mean": float(np.mean([recall_for(1), recall_for(-1)])),
        "f1_gain": f1_for(1),
        "f1_loss": f1_for(-1),
        "f1_mean": float(np.mean([f1_for(1), f1_for(-1)])),
    }
    if metadata:
        row = {**metadata, **row}
    return pd.DataFrame([row]), truth_labels, pred_labels


def rename_threshold_columns(metrics_df, truth_name, pred_name):
    """Rename generic threshold columns to dataset-specific names for notebook outputs."""
    return metrics_df.rename(
        columns={
            "truth_threshold": truth_name,
            "pred_threshold": pred_name,
        }
    )


def build_confusion_tables(truth_labels, pred_labels):
    """Build confusion tables with axes arranged to match the threshold scatter.

    Columns represent the truth axis used on the scatter plot x-axis:
    loss on the left, then neutral, then gain on the right.

    Rows represent the prediction axis used on the scatter plot y-axis:
    gain at the top, then neutral, then loss at the bottom.
    """
    label_names = {-1: "loss", 0: "neutral", 1: "gain"}
    pred_order = (1, 0, -1)
    truth_order = (-1, 0, 1)
    rows = []
    for pred in pred_order:
        row = {}
        for truth in truth_order:
            row[label_names[truth]] = int(np.sum((truth_labels == truth) & (pred_labels == pred)))
        rows.append(pd.Series(row, name=label_names[pred]))

    counts_df = pd.DataFrame(rows)[["loss", "neutral", "gain"]]
    pct_df = counts_df.div(counts_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    return counts_df, pct_df


def _genome_segments(offsets, x_values):
    """Return visible chromosome segments for genome-profile plots."""
    x_values = np.asarray(x_values, dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    starts = sorted(float(value) for value in offsets.values())

    segments = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else x_max
        visible_start = max(start, x_min)
        visible_end = min(end, x_max)
        if visible_end > visible_start:
            segments.append((visible_start, visible_end))
    return x_min, x_max, segments


def _style_genome_axis(ax, offsets, chr_midpoints, x_values, show_xticklabels=True):
    """Tighten the genome x-axis and add a chromosome boundary track underneath."""
    x_min, x_max, segments = _genome_segments(offsets, x_values)
    ax.set_xlim(x_min, x_max)
    ax.grid(False)

    for chrom in sorted(offsets):
        boundary = offsets[chrom]
        if x_min <= boundary <= x_max:
            ax.axvline(boundary, color="0.75", linewidth=0.8, zorder=0)

    track_y = -0.016
    track_colors = ("0.25", "0.65")
    for i, (start, end) in enumerate(segments):
        ax.plot(
            [start, end],
            [track_y, track_y],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            color=track_colors[i % 2],
            linewidth=7.2,
            solid_capstyle="butt",
        )

    visible_midpoints = [(midpoint, label) for midpoint, label in chr_midpoints if x_min <= midpoint <= x_max]
    ax.set_xticks([midpoint for midpoint, _ in visible_midpoints])
    if show_xticklabels:
        ax.set_xticklabels([label for _, label in visible_midpoints])
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=15)


def plot_profile_comparison(
    metric_df,
    offsets,
    chr_midpoints,
    truth_col,
    pred_col,
    truth_label,
    pred_label,
    title,
    ylabel="CNV signal",
    output_dir=None,
    sample_name=None,
    sample_prefix=None,
    run_label=None,
    stem="comparison_plot",
):
    """Plot two genomic profiles in stacked filled-area panels."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 8.1), sharex=True)
    fig.subplots_adjust(hspace=0.20, top=0.90, bottom=0.12)
    series_specs = [
        (truth_col, truth_label, axes[0]),
        (pred_col, pred_label, axes[1]),
    ]

    x = metric_df["x"].to_numpy(dtype=float)
    for value_col, panel_title, ax in series_specs:
        y = metric_df[value_col].to_numpy(dtype=float)
        ax.fill_between(x, 0, y, where=(y >= 0), color="firebrick", alpha=0.35, interpolate=True)
        ax.fill_between(x, 0, y, where=(y <= 0), color="royalblue", alpha=0.35, interpolate=True)
        ax.plot(x, y, color="black", linewidth=1.2, alpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(panel_title, fontsize=20, pad=2)

    _style_genome_axis(axes[0], offsets, chr_midpoints, x, show_xticklabels=False)
    _style_genome_axis(axes[1], offsets, chr_midpoints, x, show_xticklabels=True)
    axes[-1].set_xlabel("Chromosome", fontsize=18)
    fig.suptitle(title, fontsize=20)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            sample_prefix=sample_prefix,
            stem=stem,
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_profile_comparison_lines(
    metric_df,
    offsets,
    chr_midpoints,
    truth_col,
    pred_col,
    truth_label,
    pred_label,
    title,
    ylabel="CNV signal",
    output_dir=None,
    sample_name=None,
    sample_prefix=None,
    run_label=None,
    stem="comparison_plot_line",
):
    """Plot two genomic profiles in a single line-based panel."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 4.5), constrained_layout=True)
    ax.plot(metric_df["x"], metric_df[truth_col], color="black", linewidth=1.8, alpha=0.95, label=truth_label)
    ax.plot(metric_df["x"], metric_df[pred_col], color="darkorange", linewidth=1.4, alpha=0.95, label=pred_label)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    _style_genome_axis(ax, offsets, chr_midpoints, metric_df["x"].to_numpy(dtype=float), show_xticklabels=True)
    ax.set_xlabel("Chromosome", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title)
    ax.legend(loc="upper right")

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            sample_prefix=sample_prefix,
            stem=stem,
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_threshold_scatter(
    metric_df,
    truth_labels,
    truth_thr,
    pred_thr,
    truth_col,
    pred_col,
    truth_label,
    pred_label,
    output_dir=None,
    sample_name=None,
    sample_prefix=None,
    run_label=None,
    stem="threshold_scatter",
):
    """Plot truth versus prediction values with the selected thresholds overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), constrained_layout=True)
    colors = {1: "tab:red", 0: "gray", -1: "tab:blue"}
    labels = {1: "truth=gain", 0: "truth=neutral", -1: "truth=loss"}

    for cls in (1, 0, -1):
        mask = truth_labels == cls
        if np.any(mask):
            ax.scatter(
                metric_df.loc[mask, truth_col],
                metric_df.loc[mask, pred_col],
                s=18,
                alpha=0.65,
                c=colors[cls],
                edgecolors="none",
                label=labels[cls],
            )

    ax.axvline(truth_thr, color="firebrick", linestyle="--", linewidth=1.0)
    ax.axvline(-truth_thr, color="firebrick", linestyle="--", linewidth=1.0)
    ax.axhline(pred_thr, color="darkorange", linestyle="--", linewidth=1.0)
    ax.axhline(-pred_thr, color="darkorange", linestyle="--", linewidth=1.0)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_xlabel(f"{truth_label} value")
    ax.set_ylabel(f"{pred_label} value")
    ax.set_title(f"{truth_label} vs {pred_label} threshold selection")
    ax.legend(loc="best")

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            sample_prefix=sample_prefix,
            stem=stem,
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_confusion_matrix(
    truth_labels,
    pred_labels,
    truth_label,
    pred_label,
    output_dir=None,
    sample_name=None,
    sample_prefix=None,
    run_label=None,
    stem="confusion_matrix",
):
    """Plot a row-normalized confusion matrix and return count/pct tables."""
    counts_df, pct_df = build_confusion_tables(truth_labels, pred_labels)
    cmap = LinearSegmentedColormap.from_list("cnv_cmp_soft_yellow", ["#ffffff", "#fff7cc", "#f3df8a"])

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.8), constrained_layout=True)
    sns.heatmap(
        pct_df,
        annot=counts_df.astype(str) + "\n(" + (100 * pct_df).round(1).astype(str) + "%)",
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Truth-class fraction"},
        linewidths=0.0,
        ax=ax,
    )
    ax.grid(False)
    ax.set_xlabel(f"Truth class ({truth_label})")
    ax.set_ylabel(f"Predicted class ({pred_label})")
    ax.set_title("Confusion matrix")

    if output_dir is not None and sample_name is not None:
        save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            sample_prefix=sample_prefix,
            stem=stem,
            run_label=run_label,
        )
    plt.show()
    return counts_df, pct_df


def plot_truth_overlay(
    metric_df,
    offsets,
    chr_midpoints,
    truth_col,
    pred_col,
    truth_label,
    pred_label,
    output_dir=None,
    sample_name=None,
    sample_prefix=None,
    run_label=None,
    stem="truth_overlay",
):
    """Plot both profiles and highlight bins classified as truth gain or loss."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 4.8), constrained_layout=True)
    x = metric_df["x"].to_numpy(dtype=float)
    truth_values = metric_df[truth_col].to_numpy(dtype=float)
    pred_values = metric_df[pred_col].to_numpy(dtype=float)
    truth_classes = metric_df["truth"].to_numpy(dtype=int)

    ax.plot(x, truth_values, color="black", linewidth=1.8, alpha=0.95, label=truth_label)
    ax.plot(x, pred_values, color="darkorange", linewidth=1.3, alpha=0.95, label=pred_label)
    ax.fill_between(x, truth_values.min(), truth_values.max(), where=(truth_classes == 1), color="tab:red", alpha=0.10)
    ax.fill_between(x, truth_values.min(), truth_values.max(), where=(truth_classes == -1), color="tab:blue", alpha=0.10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    _style_genome_axis(ax, offsets, chr_midpoints, x, show_xticklabels=True)
    ax.set_xlabel("Chromosome", fontsize=18)
    ax.set_ylabel("CNV signal", fontsize=18)
    ax.set_title("Profiles with truth gain/loss regions highlighted")
    ax.legend(loc="upper right")

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            sample_prefix=sample_prefix,
            stem=stem,
            run_label=run_label,
        )
    plt.show()
    return output_path


def extract_numeric_parameter(path_or_label, pattern):
    """Extract the first integer captured by a regex from a filename or label."""
    match = re.search(pattern, str(path_or_label), re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_gene_profile_runs(results_path, sample_name=None, pattern=None, parameter_name="parameter_value", label_prefix="run"):
    """Discover saved gene-level CNV CSVs for a parameter sweep and sort them by parameter value."""
    sample_tokens = {slugify(sample_name), slugify(sample_name).replace("_", "")} if sample_name else set()
    runs = []
    for path in sorted(Path(results_path).glob("*.csv")):
        if sample_tokens:
            name_tokens = {slugify(path.stem), slugify(path.name)}
            if not any(token in name for token in sample_tokens for name in name_tokens):
                continue
        parameter_value = extract_numeric_parameter(path.name, pattern) if pattern else None
        run_label = f"{label_prefix}{parameter_value}" if parameter_value is not None else path.stem
        runs.append(
            {
                "path": path,
                parameter_name: parameter_value,
                "run_label": run_label,
            }
        )

    runs.sort(
        key=lambda run: (
            run.get(parameter_name) is None,
            run.get(parameter_name) if run.get(parameter_name) is not None else run["run_label"],
        )
    )
    return runs


def evaluate_wes_insitu_parameter_sweep(
    runs,
    wes,
    sample_name,
    wes_sample_id,
    bin_size_bp,
    min_genes_per_bin,
    parameter_name,
):
    """Run the quantitative validation framework across a list of saved in situ profiles."""
    run_results = []
    metrics_frames = []
    truth_grid_frames = []
    pred_grid_frames = []

    for run in runs:
        insitu = load_gene_cnv_values(run["path"], value_col="CNV_value")
        if not insitu:
            continue

        metric_df, axis = build_wes_insitu_comparison_table(
            wes=wes,
            insitu=insitu,
            bin_size_bp=bin_size_bp,
            min_genes_per_bin=min_genes_per_bin,
        )
        if metric_df.empty:
            continue

        truth_thr, pred_thr, truth_grid_df, pred_grid_df = select_validation_thresholds(
            metric_df=metric_df,
            truth_col="wes_segmean",
            pred_col="insitu_value",
            truth_threshold_name="wes_threshold",
            pred_threshold_name="insitu_threshold",
        )

        metrics_df, truth_labels, pred_labels = compute_validation_metrics(
            metric_df=metric_df,
            truth_thr=truth_thr,
            pred_thr=pred_thr,
            truth_col="wes_segmean",
            pred_col="insitu_value",
            metadata={
                "sample_name": sample_name,
                "wes_sample_id": wes_sample_id,
                "run_label": run["run_label"],
                parameter_name: run.get(parameter_name),
                "display_label": run.get("display_label", run["run_label"]),
                "insitu_file": run["path"].name,
                "mode": "raw" if bin_size_bp is None else "binned",
                "bin_size_bp": bin_size_bp,
                "min_genes_per_bin": min_genes_per_bin,
            },
        )

        metric_df = metric_df.copy()
        metric_df["truth"] = truth_labels
        metric_df["pred"] = pred_labels

        truth_grid_df = truth_grid_df.copy()
        truth_grid_df["run_label"] = run["run_label"]
        truth_grid_df[parameter_name] = run.get(parameter_name)

        pred_grid_df = pred_grid_df.copy()
        pred_grid_df["run_label"] = run["run_label"]
        pred_grid_df[parameter_name] = run.get(parameter_name)

        run_results.append(
            {
                **run,
                "metric_df": metric_df,
                "axis": axis,
                "truth_thr": truth_thr,
                "pred_thr": pred_thr,
                "truth_labels": truth_labels,
                "pred_labels": pred_labels,
                "metrics_df": rename_threshold_columns(metrics_df, "wes_threshold", "insitu_threshold"),
            }
        )
        metrics_frames.append(rename_threshold_columns(metrics_df, "wes_threshold", "insitu_threshold"))
        truth_grid_frames.append(truth_grid_df)
        pred_grid_frames.append(pred_grid_df)

    all_metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    all_truth_grid_df = pd.concat(truth_grid_frames, ignore_index=True) if truth_grid_frames else pd.DataFrame()
    all_pred_grid_df = pd.concat(pred_grid_frames, ignore_index=True) if pred_grid_frames else pd.DataFrame()
    return run_results, all_metrics_df, all_truth_grid_df, all_pred_grid_df


def plot_parameter_metric_summary(
    metrics_df,
    parameter_name,
    parameter_label,
    output_dir=None,
    sample_name=None,
    run_label="all_runs",
    stem=None,
):
    """Plot mean AUC and mean F1 against one sweep parameter."""
    if metrics_df.empty:
        return None

    plot_df = metrics_df.sort_values(parameter_name).copy()
    x = plot_df[parameter_name].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x, plot_df["auc_mean"], marker="o", linewidth=1.8)
    axes[0].set_title(f"Mean AUC vs {parameter_label}")
    axes[0].set_xlabel(parameter_label)
    axes[0].set_ylabel("auc_mean")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, plot_df["f1_mean"], marker="o", linewidth=1.8, color="darkorange")
    axes[1].set_title(f"Mean F1 vs {parameter_label}")
    axes[1].set_xlabel(parameter_label)
    axes[1].set_ylabel("f1_mean")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            stem=stem or f"{parameter_name}_metric_summary",
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_parameter_metric_bars(
    metrics_df,
    parameter_name,
    parameter_label,
    output_dir=None,
    sample_name=None,
    run_label="all_runs",
    stem=None,
):
    """Plot grouped bar charts for AUC, precision, recall, and F1 across a parameter sweep."""
    if metrics_df.empty:
        return None

    metric_columns = [
        "auc_gain",
        "auc_loss",
        "auc_mean",
        "precision_gain",
        "precision_loss",
        "precision_mean",
        "recall_gain",
        "recall_loss",
        "recall_mean",
        "f1_gain",
        "f1_loss",
        "f1_mean",
    ]
    plot_df = metrics_df.sort_values(parameter_name).copy()
    plot_df["parameter_label"] = plot_df[parameter_name].astype(int).astype(str)

    tidy_metrics = plot_df.melt(
        id_vars=["sample_name", "run_label", parameter_name, "parameter_label", "wes_threshold", "insitu_threshold"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    tidy_metrics[["metric_family", "state"]] = tidy_metrics["metric"].str.split("_", n=1, expand=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    family_order = ["auc", "precision", "recall", "f1"]
    state_order = ["gain", "loss", "mean"]
    hue_order = list(plot_df["parameter_label"])

    for ax, family in zip(axes.flat, family_order):
        family_df = tidy_metrics.loc[tidy_metrics["metric_family"] == family]
        sns.barplot(
            data=family_df,
            x="state",
            y="value",
            hue="parameter_label",
            order=state_order,
            hue_order=hue_order,
            palette="viridis",
            ax=ax,
        )
        ax.set_title(f"{family.upper()} across {parameter_label}")
        ax.set_xlabel("")
        ax.set_ylabel(family)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(title=parameter_label)
        if family == "auc":
            ax.set_ylim(0.5, 1.0)
        else:
            ax.set_ylim(0.0, 1.0)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            stem=stem or f"{parameter_name}_metric_bars",
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_parameter_threshold_curves(
    metrics_df,
    truth_grid_df,
    pred_grid_df,
    parameter_name,
    parameter_label,
    output_dir=None,
    sample_name=None,
    run_label="all_runs",
    stem=None,
):
    """Plot the selected threshold values and their optimization curves across a sweep."""
    if metrics_df.empty or truth_grid_df.empty or pred_grid_df.empty:
        return None

    plot_df = metrics_df.sort_values(parameter_name).copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    axes[0, 0].plot(plot_df[parameter_name], plot_df["wes_threshold"], marker="o", linewidth=1.8)
    axes[0, 0].set_title(f"Selected WES thresholds vs {parameter_label}")
    axes[0, 0].set_xlabel(parameter_label)
    axes[0, 0].set_ylabel("wes_threshold")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(plot_df[parameter_name], plot_df["insitu_threshold"], marker="o", linewidth=1.8, color="darkorange")
    axes[0, 1].set_title(f"Selected in situ thresholds vs {parameter_label}")
    axes[0, 1].set_xlabel(parameter_label)
    axes[0, 1].set_ylabel("insitu_threshold")
    axes[0, 1].grid(alpha=0.25)

    for _, subset in truth_grid_df.groupby("run_label"):
        x_value = subset[parameter_name].iloc[0]
        best_row = subset.sort_values("auc_mean", ascending=False).iloc[0]
        axes[1, 0].scatter(x_value, best_row["auc_mean"], s=60)
        axes[1, 0].text(x_value, best_row["auc_mean"], str(int(x_value)), fontsize=9, ha="center", va="bottom")
    axes[1, 0].set_title(f"Best mean AUC after WES threshold search")
    axes[1, 0].set_xlabel(parameter_label)
    axes[1, 0].set_ylabel("auc_mean")
    axes[1, 0].set_ylim(0.5, 1.0)
    axes[1, 0].grid(alpha=0.25)

    for _, subset in pred_grid_df.groupby("run_label"):
        x_value = subset[parameter_name].iloc[0]
        best_row = subset.sort_values("f1_mean", ascending=False).iloc[0]
        axes[1, 1].scatter(x_value, best_row["f1_mean"], s=60, color="darkorange")
        axes[1, 1].text(x_value, best_row["f1_mean"], str(int(x_value)), fontsize=9, ha="center", va="bottom")
    axes[1, 1].set_title("Best mean F1 after in situ threshold search")
    axes[1, 1].set_xlabel(parameter_label)
    axes[1, 1].set_ylabel("f1_mean")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(alpha=0.25)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            stem=stem or f"{parameter_name}_threshold_curves",
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_multi_run_profile_lines(
    run_results,
    parameter_name,
    parameter_label,
    output_dir=None,
    sample_name=None,
    run_label="all_runs",
    stem=None,
):
    """Overlay the in situ profiles from multiple runs against the shared WES baseline."""
    if not run_results:
        return None

    sorted_runs = sorted(run_results, key=lambda run: run.get(parameter_name))
    first_run = sorted_runs[0]
    metric_df = first_run["metric_df"]
    axis = first_run["axis"]

    fig, ax = plt.subplots(1, 1, figsize=(18, 4.8), constrained_layout=True)
    ax.plot(metric_df["x"], metric_df["wes_segmean"], color="black", linewidth=2.0, alpha=0.95, label="WES")

    palette = sns.color_palette("viridis", n_colors=len(sorted_runs))
    for color, run in zip(palette, sorted_runs):
        label_value = run.get("display_label", run.get(parameter_name))
        label = f"{parameter_label}={label_value}"
        ax.plot(
            run["metric_df"]["x"],
            run["metric_df"]["insitu_value"],
            linewidth=1.2,
            alpha=0.95,
            color=color,
            label=label,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for chrom in sorted(axis["offsets"]):
        ax.axvline(axis["offsets"][chrom], color="lightgray", linewidth=0.5)
    ax.set_xticks([midpoint for midpoint, _ in axis["chr_midpoints"]])
    ax.set_xticklabels([label for _, label in axis["chr_midpoints"]])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("CNV signal")
    ax.set_title(f"WES vs in situ profiles across {parameter_label}")
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            stem=stem or f"{parameter_name}_comparison_lines",
            run_label=run_label,
        )
    plt.show()
    return output_path


def plot_multi_run_profile_stack(
    run_results,
    parameter_name,
    parameter_label,
    output_dir=None,
    sample_name=None,
    run_label="all_runs",
    stem=None,
):
    """Plot WES on top and each in situ run in its own panel below."""
    if not run_results:
        return None

    sorted_runs = sorted(run_results, key=lambda run: run.get(parameter_name))
    first_run = sorted_runs[0]
    n_panels = len(sorted_runs) + 1

    fig_height = max(4.8, 2.0 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, fig_height), sharex=True)
    fig.subplots_adjust(hspace=0.34, top=0.96, bottom=0.08)

    wes_ax = axes[0]
    wes_df = first_run["metric_df"]
    wes_axis = first_run["axis"]
    wes_x = wes_df["x"].to_numpy(dtype=float)
    wes_y = wes_df["wes_segmean"].to_numpy(dtype=float)
    wes_ax.fill_between(wes_x, 0, wes_y, where=(wes_y >= 0), color="firebrick", alpha=0.30, interpolate=True)
    wes_ax.fill_between(wes_x, 0, wes_y, where=(wes_y <= 0), color="royalblue", alpha=0.30, interpolate=True)
    wes_ax.plot(wes_x, wes_y, color="black", linewidth=1.2, alpha=0.95)
    wes_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    wes_ax.set_ylabel("WES", fontsize=15)
    wes_ax.set_title("WES seg.mean", fontsize=18, pad=4)
    _style_genome_axis(wes_ax, wes_axis["offsets"], wes_axis["chr_midpoints"], wes_x, show_xticklabels=False)

    for ax, run in zip(axes[1:], sorted_runs):
        metric_df = run["metric_df"]
        axis = run["axis"]
        x = metric_df["x"].to_numpy(dtype=float)
        y = metric_df["insitu_value"].to_numpy(dtype=float)
        value = run.get(parameter_name)
        label_value = run.get("display_label")
        if label_value is None:
            label_value = int(value) if value is not None else run["run_label"]

        ax.fill_between(x, 0, y, where=(y >= 0), color="firebrick", alpha=0.30, interpolate=True)
        ax.fill_between(x, 0, y, where=(y <= 0), color="royalblue", alpha=0.30, interpolate=True)
        ax.plot(x, y, color="black", linewidth=1.0, alpha=0.95)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{parameter_label}={label_value}", fontsize=13)
        ax.set_title(f"in situ mean | {parameter_label}={label_value}", fontsize=15, pad=3)
        _style_genome_axis(ax, axis["offsets"], axis["chr_midpoints"], x, show_xticklabels=ax is axes[-1])

    axes[-1].set_xlabel("Chromosome", fontsize=18)
    fig.suptitle(f"Comparative WES and in situ profiles across {parameter_label}", fontsize=20, y=0.995)

    output_path = None
    if output_dir is not None and sample_name is not None:
        output_path = save_current_figure(
            output_dir=output_dir,
            sample_name=sample_name,
            stem=stem or f"{parameter_name}_comparison_stack",
            run_label=run_label,
        )
    plt.show()
    return output_path

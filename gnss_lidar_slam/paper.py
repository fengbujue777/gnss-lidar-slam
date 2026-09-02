"""Recreate paper result tables and figures from saved experiment results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DATASET_NAMES = {
    "DATASET-001": "RTK-SLAM", "DATASET-002": "M2DGR", "DATASET-003": "i2Nav-Robot"
}
METHOD_NAMES = {"baseline": "KISS-SLAM", "candidate": "Proposed method"}
PRIMARY_METRIC = {"DATASET-001": "checkpoint_3d_rmse", "DATASET-002": "ate_rmse", "DATASET-003": "ate_rmse"}
SEQUENCE_ORDER = {
    "DATASET-001": ["stadtgarten_seq1", "stadtgarten_seq2", "construction_seq1", "construction_seq2"],
    "DATASET-002": ["door_01", "door_02", "street_01"],
    "DATASET-003": ["building00", "building01", "building02", "parking00", "playground00", "street00"],
}
COLORS = {"KISS-SLAM": "#4C78A8", "RTAB-Map": "#F58518", "Proposed method": "#54A24B"}


def _metric(result: dict, name: str) -> float | None:
    for item in result.get("metrics", []):
        if item.get("name") == name and isinstance(item.get("value"), (int, float)):
            return float(item["value"])
    return None


def load_sequence_values(root: Path) -> list[dict]:
    manifests = root / "stage_6/runs/manifests"
    results = root / "stage_6/runs/results"
    rows: dict[tuple[str, str], dict] = {}
    for result_path in sorted(results.glob("RUN-*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        manifest_path = manifests / result_path.name
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_id, variant = manifest.get("dataset_id"), manifest.get("variant")
        if dataset_id not in PRIMARY_METRIC or variant not in METHOD_NAMES:
            continue
        value = _metric(result, PRIMARY_METRIC[dataset_id])
        if value is None:
            continue
        key = (dataset_id, manifest["sequence_id"])
        rows.setdefault(key, {"dataset_id": dataset_id, "sequence": manifest["sequence_id"]})
        rows[key][METHOD_NAMES[variant]] = value

    # RTAB-Map results use their source manifest to recover dataset and sequence.
    rtab_root = root / "stage_6/comparisons/rtabmap"
    for result_path in sorted((rtab_root / "artifacts").glob("RTABMAP-*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        metadata_path = result_path.parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        dataset_id, sequence = metadata.get("dataset_id"), metadata.get("sequence_id")
        if dataset_id not in PRIMARY_METRIC or not sequence:
            continue
        value = _metric(result, PRIMARY_METRIC[dataset_id])
        if value is None:
            continue
        rows.setdefault((dataset_id, sequence), {"dataset_id": dataset_id, "sequence": sequence})["RTAB-Map"] = value
    def ordering(row):
        sequence_order = SEQUENCE_ORDER.get(row["dataset_id"], [])
        try:
            sequence_index = sequence_order.index(row["sequence"])
        except ValueError:
            sequence_index = len(sequence_order)
        return row["dataset_id"], sequence_index, row["sequence"]
    return sorted(rows.values(), key=ordering)


def write_table3(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = ["Dataset", "Sequence", "KISS-SLAM", "RTAB-Map", "Proposed method"]
    with (output / "table3_sequence_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        for row in rows:
            writer.writerow([DATASET_NAMES[row["dataset_id"]], row["sequence"], *[
                "" if row.get(method) is None else f"{row[method]:.2f}"
                for method in fields[2:]
            ]])
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Dataset & Sequence & KISS-SLAM & RTAB-Map & Proposed method \\\\", "\\midrule"]
    for row in rows:
        values = [row.get(method) for method in fields[2:]]
        rendered = ["--" if value is None else f"{value:.2f}" for value in values]
        finite = [value for value in values if value is not None]
        if finite:
            best = min(round(value, 2) for value in finite)
            rendered = [f"\\textbf{{{text}}}" if value is not None and round(value, 2) == best else text for text, value in zip(rendered, values)]
        sequence = row["sequence"].replace("_", "\\_")
        lines.append(f"{DATASET_NAMES[row['dataset_id']]} & {sequence} & " + " & ".join(rendered) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "table3_sequence_accuracy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_sequence_accuracy(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    datasets = list(DATASET_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [4.2, 3.2, 5.7]})
    methods, width = ["KISS-SLAM", "RTAB-Map", "Proposed method"], 0.24
    for axis, dataset_id in zip(axes, datasets):
        selected = [row for row in rows if row["dataset_id"] == dataset_id]
        positions = np.arange(len(selected), dtype=float)
        maximum = 0.0
        for offset, method in zip((-width, 0, width), methods):
            values = np.asarray([row.get(method, np.nan) for row in selected], dtype=float)
            bars = axis.bar(positions + offset, np.nan_to_num(values), width, color=COLORS[method], label=method)
            for bar, value in zip(bars, values):
                if np.isnan(value): bar.set_visible(False)
                else:
                    maximum = max(maximum, value)
                    axis.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        axis.set_title(DATASET_NAMES[dataset_id]); axis.set_xticks(positions, [row["sequence"] for row in selected], rotation=25, ha="right", fontsize=8)
        axis.set_ylabel("Checkpoint 3D RMSE (m)" if dataset_id == "DATASET-001" else "ATE RMSE (m)")
        axis.set_ylim(0, maximum * 1.18 if maximum else 1); axis.grid(axis="y", color="0.88"); axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.84, bottom=.24, wspace=.3)
    for suffix in ("pdf", "png"): fig.savefig(output / f"figure6_sequence_accuracy_bars.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    source = root / "stage_6/runtime_benchmark/aggregate.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload.get("complete"): raise RuntimeError("saved 1,200-frame runtime benchmark is incomplete")
    methods = ["KISS-SLAM", "RTAB-Map", "Proposed method"]
    summaries = {(item["dataset"], item["method"]): item for item in payload["summaries"]}
    datasets = list(dict.fromkeys(item["dataset"] for item in payload["summaries"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.25, 2.55), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        means = [np.mean([payload["frame_limit"] / seconds for seconds in summaries[(dataset, method)]["repetitions_seconds"]]) for method in methods]
        bars = axis.bar(np.arange(3), means, color=[COLORS[m] for m in methods], edgecolor="black", linewidth=.45)
        for bar, value in zip(bars, means): axis.text(bar.get_x()+bar.get_width()/2, value+max(means)*.035, f"{value:.2f}", ha="center", fontsize=8)
        axis.set_title(f"{dataset}\n{summaries[(dataset, methods[0])]['sequence']}"); axis.set_xticks(np.arange(3), ["KISS-SLAM", "RTAB-Map", "Proposed"], rotation=25, ha="right")
        axis.set_ylim(0, max(means)*1.2); axis.grid(axis="y", color="#d9d9d9"); axis.set_axisbelow(True)
    axes[0].set_ylabel("Average end-to-end throughput (FPS)")
    for suffix in ("pdf", "png"): fig.savefig(output / f"figure7_runtime_1200.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Directory containing stage_6 saved results")
    parser.add_argument("--output", type=Path, default=Path("paper_artifacts"))
    parser.add_argument("--skip-comparisons", action="store_true", help="Skip point-cloud Figures 3-5")
    parser.add_argument("--cloud-cache", type=Path, help="Saved point-cloud cache used by Figures 3-5")
    args = parser.parse_args(argv)
    root, output = args.results_root.expanduser().resolve(), args.output.expanduser().resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True); (output / "tables").mkdir(parents=True, exist_ok=True)
    rows = load_sequence_values(root)
    if not rows: raise RuntimeError(f"no completed saved results found below {root}")
    write_table3(rows, output / "tables"); plot_sequence_accuracy(rows, output / "figures"); plot_runtime(root, output / "figures")
    if not args.skip_comparisons:
        from .comparison_figures import main as comparisons
        command = ["--results-root", str(root), "--output", str(output / "figures")]
        if args.cloud_cache: command.extend(["--cloud-cache", str(args.cloud_cache)])
        comparisons(command)
    manifest = {"source": str(root), "generated": [str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file()]}
    (output / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run():
    main()


if __name__ == "__main__":
    main()

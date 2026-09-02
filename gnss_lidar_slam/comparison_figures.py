#!/usr/bin/env python3
"""Build publication figures from checksum-bound Stage 6 run artifacts.

The point clouds are reconstructed in each method's final optimized graph frame.
For KISS-SLAM and the proposed method, deterministic samples of the same raw
LiDAR scans are placed with the final per-frame graph vertices. This is
required because the proposed batch GNSS solve now acts on frame vertices
after local-map serialization. RTAB-Map clouds are exported from its database
at optimized poses by export_rtabmap_cloud.cpp.
Existing trajectory panels contain the final
optimized estimate and the method-specific reference overlays.  RTAB-Map's
stored geodetic priors are converted to local ENU and rigidly registered
without scale into the displayed reference frame for visualization only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path.cwd()
RUNS = ROOT / "stage_6/runs/artifacts"
RTAB = ROOT / "stage_6/comparisons/rtabmap/artifacts"
DATA = ROOT / "stage_8/draft/figures/comparisons/data"
OUT = ROOT / "paper_artifacts/figures"

DATASETS = {
    "rtkslam": [
        ("stadtgarten_seq1", "RUN-0001-001-0004", "RUN-0001-001-0001", "RTABMAP-0001"),
        ("stadtgarten_seq2", "RUN-0001-001-0010", "RUN-0001-001-0007", "RTABMAP-0004"),
        ("construction_seq1", "RUN-0001-001-0016", "RUN-0001-001-0013", "RTABMAP-0007"),
        ("construction_seq2", "RUN-0001-001-0022", "RUN-0001-001-0019", None),
    ],
    "m2dgr": [
        ("door_01", "RUN-0001-001-0028", "RUN-0001-001-0025", "RTABMAP-0013"),
        ("door_02", "RUN-0001-001-0034", "RUN-0001-001-0031", "RTABMAP-0016"),
        ("street_01", "RUN-0001-001-0040", "RUN-0001-001-0037", "RTABMAP-0019"),
    ],
    "i2nav": [
        ("building00", "RUN-0001-001-0046", "RUN-0001-001-0043", "RTABMAP-0022"),
        ("building01", "RUN-0001-001-0052", "RUN-0001-001-0049", None),
        ("building02", "RUN-0001-001-0058", "RUN-0001-001-0055", None),
        ("parking00", "RUN-0001-001-0064", "RUN-0001-001-0061", None),
        ("playground00", "RUN-0001-001-0082", "RUN-0001-001-0079", "RTABMAP-0040"),
        ("street00", "RUN-0001-001-0088", "RUN-0001-001-0085", "RTABMAP-0043"),
    ],
}


def read_binary_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        count = None
        properties = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                count = int(text.rsplit(" ", 1)[1])
            elif text.startswith("property "):
                properties.append(text)
            elif text == "end_header":
                break
        if count is None or properties[:3] != ["property float x", "property float y", "property float z"]:
            raise ValueError(f"Unexpected PLY layout: {path}")
        record = np.dtype([(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")])
        points = np.fromfile(handle, dtype=record, count=count)
    return np.column_stack((points["x"], points["y"], points["z"]))


def kiss_cloud(run_id: str, target_points: int = 220_000) -> np.ndarray:
    optimized_scan_cache = DATA / f"{run_id}-optimized-scan-cloud-v2.npy"
    if optimized_scan_cache.exists():
        return np.load(optimized_scan_cache)
    raise RuntimeError(
        f"Missing final-frame point-cloud cache for {run_id}; run "
        "with --cloud-cache pointing to the saved *-optimized-scan-cloud-v2.npy files"
    )


def rtab_cloud(run_id: str) -> np.ndarray:
    path = DATA / f"{run_id}-cloud.csv"
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def optimized_graph_xyz(path: Path, expected_count: int) -> np.ndarray:
    """Read the final fine-grained SE(3) graph positions in vertex-ID order."""
    vertices: dict[int, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] == "VERTEX_SE3:QUAT":
                vertices[int(fields[1])] = np.asarray(fields[2:5], dtype=np.float64)
    expected_ids = set(range(expected_count))
    if set(vertices) != expected_ids:
        missing = sorted(expected_ids - set(vertices))[:5]
        extra = sorted(set(vertices) - expected_ids)[:5]
        raise RuntimeError(
            f"Optimized graph {path} does not match its timestamp stream: "
            f"expected={expected_count}, vertices={len(vertices)}, "
            f"missing={missing}, extra={extra}"
        )
    return np.vstack([vertices[index] for index in range(expected_count)])


def plot_cloud(fig, ax, xyz: np.ndarray) -> None:
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    lo = np.percentile(xyz[:, :2], 0.5, axis=0)
    hi = np.percentile(xyz[:, :2], 99.5, axis=0)
    inside = np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), axis=1)
    xyz = xyz[inside]
    if len(xyz) > 180_000:
        xyz = xyz[np.linspace(0, len(xyz) - 1, 180_000, dtype=int)]
    zlo, zhi = np.percentile(xyz[:, 2], [2, 98])
    color = np.clip(xyz[:, 2], zlo, zhi)
    points = ax.scatter(
        xyz[:, 0], xyz[:, 1], c=color, cmap="viridis", vmin=zlo, vmax=zhi,
        s=0.08, linewidths=0, rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.75")
        spine.set_linewidth(0.5)


def interpolate(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (query >= times[0]) & (query <= times[-1])
    result = np.empty((len(query), values.shape[1]), dtype=np.float64)
    result[:] = np.nan
    for dim in range(values.shape[1]):
        result[valid, dim] = np.interp(query[valid], times, values[:, dim])
    return result, valid


def reference_records(source_run: str) -> tuple[np.ndarray, np.ndarray, bool]:
    manifest = json.loads((ROOT / f"stage_6/runs/manifests/{source_run}.json").read_text())
    path = ROOT / manifest["inputs"]["reference_path"]
    if manifest["dataset_id"] == "DATASET-001":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        times = np.asarray([float(row["timestamp"]) for row in rows])
        xyz = np.asarray([[float(row["easting"]), float(row["northing"]), float(row["height"])] for row in rows])
        return times, xyz, True
    rows = np.loadtxt(path, dtype=np.float64)
    return rows[:, 0], rows[:, 1:4], False


def rigid_display_alignment(
    estimate_times: np.ndarray,
    estimate_xyz: np.ndarray,
    reference_times: np.ndarray,
    reference_xyz: np.ndarray,
    gps_sow_policy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    match_times = estimate_times
    if gps_sow_policy:
        match_times = np.mod(estimate_times - 315964800.0 + 18.0, 604800.0)
    matched, valid = interpolate(match_times, estimate_xyz, reference_times)
    if np.count_nonzero(valid) < 3:
        raise RuntimeError("Fewer than three reference matches for trajectory figure")
    estimate = matched[valid]
    reference = reference_xyz[valid]
    ec = estimate.mean(axis=0)
    rc = reference.mean(axis=0)
    u, _, vt = np.linalg.svd((estimate - ec).T @ (reference - rc))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = rc - ec @ rotation
    return estimate_xyz @ rotation + translation, reference, rotation, translation


def stored_rtab_gps_records(database: Path) -> tuple[np.ndarray, np.ndarray]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT gps FROM Node WHERE gps IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    values = np.asarray([struct.unpack("6d", row[0]) for row in rows], dtype=np.float64)
    if not len(values):
        return np.empty(0), np.empty((0, 3))
    # Stored layout: stamp, longitude, latitude, altitude, accuracy, bearing.
    return values[:, 0], values[:, [2, 1, 3]]


def local_enu(lla: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude/altitude records to WGS84 local ENU."""
    if not len(lla):
        return np.empty((0, 3))
    semi_major = 6378137.0
    eccentricity_squared = 6.69437999014e-3
    lat = np.deg2rad(lla[:, 0])
    lon = np.deg2rad(lla[:, 1])
    altitude = lla[:, 2]
    prime_vertical = semi_major / np.sqrt(
        1.0 - eccentricity_squared * np.sin(lat) ** 2
    )
    ecef = np.column_stack((
        (prime_vertical + altitude) * np.cos(lat) * np.cos(lon),
        (prime_vertical + altitude) * np.cos(lat) * np.sin(lon),
        (prime_vertical * (1.0 - eccentricity_squared) + altitude) * np.sin(lat),
    ))
    lat0, lon0 = lat[0], lon[0]
    ecef_to_enu = np.asarray([
        [-np.sin(lon0), np.cos(lon0), 0.0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])
    return (ecef - ecef[0]) @ ecef_to_enu.T


def stored_rtab_gps(database: Path) -> tuple[np.ndarray, np.ndarray]:
    stamp, lla = stored_rtab_gps_records(database)
    return stamp, local_enu(lla)[:, :2]


def m2dgr_gps_display(
    database: Path, source_run: str, proposed_frame: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Map M2DGR RTAB GPS through the accepted final batch calibration.

    Fitting GPS independently to an already reference-aligned trajectory can
    select an incorrect reflection on a near-degenerate path.  M2DGR has a
    checksum-bound GNSS-to-map calibration from the proposed run, so use that
    physical frame chain for both methods instead: WGS84 -> ENU -> calibrated
    KISS map -> common displayed reference frame.
    """
    stamp, lla = stored_rtab_gps_records(database)
    if not len(stamp):
        return stamp, np.empty((0, 2))
    diagnostics = json.loads((RUNS / source_run / "gnss_diagnostics.json").read_text())
    calibration = diagnostics.get("batch_final_calibration", {})
    if not calibration.get("accepted"):
        return stamp, np.empty((0, 2))
    yaw = float(calibration["yaw_alignment_rad"])
    yaw_rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    gps_map = (
        local_enu(lla) @ yaw_rotation.T
        + np.asarray(calibration["map_translation_m"], dtype=np.float64)
    )
    displayed = gps_map @ proposed_frame["rotation"] + proposed_frame["translation"]
    return stamp, displayed[:, :2]


def align_planar(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
    rotation = u @ vt
    # Some dataset reference frames swap the displayed X/Y axis convention
    # relative to ENU. Permit the resulting orthogonal reflection here; forcing
    # det(R)=+1 made otherwise timestamp-matched GPS priors appear tens of
    # metres away from the RTAB-Map trajectory on i2Nav and M2DGR.
    return (source - sc) @ rotation + tc


def rtab_trajectory_data(run_id: str, source_run: str, proposed_frame: dict) -> dict:
    rows = np.genfromtxt(RTAB / run_id / "trajectory.csv", delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    times = np.asarray(rows["timestamp"], dtype=np.float64)
    xyz = np.column_stack((rows["x"], rows["y"], rows["z"]))
    ref_times, ref_xyz, checkpoints = reference_records(source_run)
    manifest = json.loads((ROOT / f"stage_6/runs/manifests/{source_run}.json").read_text())
    displayed, displayed_ref, rotation, translation = rigid_display_alignment(
        times, xyz, ref_times, ref_xyz, manifest["dataset_id"] == "DATASET-003"
    )
    if manifest["dataset_id"] == "DATASET-002":
        gps_times, calibrated_gps = m2dgr_gps_display(
            RTAB / run_id / "rtabmap.db", source_run, proposed_frame,
        )
    else:
        gps_times, gps_enu = stored_rtab_gps(RTAB / run_id / "rtabmap.db")
        calibrated_gps = np.empty((0, 2))
    gps_display = np.empty((0, 2))
    if len(gps_times):
        _, valid = interpolate(times, displayed[:, :2], gps_times)
        if manifest["dataset_id"] == "DATASET-002":
            gps_display = calibrated_gps[valid]
        else:
            paired, _ = interpolate(times, displayed[:, :2], gps_times)
            if np.count_nonzero(valid) >= 3:
                gps_display = align_planar(gps_enu[valid], paired[valid])

    return {
        "estimate": displayed,
        "reference": displayed_ref,
        "checkpoints": checkpoints,
        "gps": gps_display,
        "anchors": np.empty((0, 3)),
        "rotation": rotation,
        "translation": translation,
    }


def kiss_trajectory_data(run_id: str, proposed: bool) -> dict:
    result_dir = RUNS / run_id / "result_artifacts"
    tum_files = sorted(result_dir.glob("*_poses_tum.txt"))
    if not tum_files and proposed:
        # Final-GNSS-only replay deliberately removes the stale proposed TUM
        # export. Reuse timestamps from the paired KISS-SLAM run, which
        # processes the identical sensor-frame grid, while continuing to read
        # every displayed proposed position from its optimized graph.
        manifest = json.loads(
            (ROOT / f"stage_6/runs/manifests/{run_id}.json").read_text()
        )
        for candidate in sorted((ROOT / "stage_6/runs/manifests").glob("RUN*.json")):
            paired = json.loads(candidate.read_text())
            if (
                paired.get("variant") == "baseline"
                and paired.get("dataset_id") == manifest.get("dataset_id")
                and paired.get("sequence_id") == manifest.get("sequence_id")
                and paired.get("split") == manifest.get("split")
                and paired.get("repetition") == manifest.get("repetition")
            ):
                tum_files = sorted(
                    (RUNS / paired["run_id"] / "result_artifacts").glob(
                        "*_poses_tum.txt"
                    )
                )
                break
    if len(tum_files) != 1:
        raise RuntimeError(f"Expected exactly one timestamp-source TUM trajectory for {run_id}")
    rows = np.loadtxt(tum_files[0], dtype=np.float64)
    times = rows[:, 0]
    # SlamPipeline deliberately writes the TUM file from `pipeline.poses`,
    # which is the online-published stream. Accuracy, however, is evaluated
    # from `pipeline.optimized_poses`; its durable artifact is trajectory.g2o,
    # written after fine-grained graph optimization. Use the TUM file only for
    # timestamps and the graph vertices for every displayed estimate.
    xyz = optimized_graph_xyz(result_dir / "trajectory.g2o", len(times))
    ref_times, ref_xyz, checkpoints = reference_records(run_id)
    manifest = json.loads((ROOT / f"stage_6/runs/manifests/{run_id}.json").read_text())
    displayed, displayed_ref, rotation, translation = rigid_display_alignment(
        times, xyz, ref_times, ref_xyz, manifest["dataset_id"] == "DATASET-003"
    )
    anchors = np.empty((0, 3))
    diagnostics_path = RUNS / run_id / "gnss_diagnostics.json"
    if proposed and diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        values = [
            event["factor_position"]
            for event in diagnostics.get("anchor_events", [])
            if event.get("accepted") and event.get("factor_added") and event.get("factor_position")
        ]
        if values:
            anchors = np.asarray(values, dtype=np.float64) @ rotation + translation
    return {
        "estimate": displayed,
        "reference": displayed_ref,
        "checkpoints": checkpoints,
        "gps": np.empty((0, 2)),
        "anchors": anchors,
        "rotation": rotation,
        "translation": translation,
    }


def plot_trajectory(ax, data: dict) -> None:
    estimate = data["estimate"]
    reference = data["reference"]
    ax.plot(
        estimate[:, 0], estimate[:, 1], color="black", linewidth=0.9,
        label="Final optimized estimate",
    )
    if data["checkpoints"]:
        ax.scatter(
            reference[:, 0], reference[:, 1], s=10, color="#1f77b4",
            label="Surveyed checkpoints",
        )
    else:
        ax.plot(
            reference[:, 0], reference[:, 1], color="#1f77b4", linewidth=0.8,
            label="Ground truth",
        )
    if len(data["anchors"]):
        anchors = data["anchors"]
        ax.scatter(
            anchors[:, 0], anchors[:, 1], s=8, color="#d62728", marker="x",
            label="Reliable GNSS anchors",
        )
    if len(data["gps"]):
        gps = data["gps"]
        ax.scatter(
            gps[:, 0], gps[:, 1], s=6, color="#ff7f0e", alpha=0.8,
            label="All stored GPS priors",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=6)
    ax.set_ylabel("Y [m]", fontsize=6)
    ax.tick_params(labelsize=5)
    # Reserve empty space above the trajectory so the top-right legend does
    # not cover any trajectory or reference samples.
    ymin, ymax = ax.get_ylim()
    height = max(ymax - ymin, 1.0)
    ax.set_ylim(ymin, ymax + 0.24 * height)
    ax.legend(loc="upper right", fontsize=5.2, framealpha=0.95)


def aligned_cloud(run_id: str, data: dict, rtabmap: bool = False) -> np.ndarray:
    cloud = rtab_cloud(run_id) if rtabmap else kiss_cloud(run_id)
    return cloud @ data["rotation"] + data["translation"]


def make_dataset_plate(dataset: str, rows) -> None:
    headings = [
        "KISS-SLAM trajectory", "KISS-SLAM point cloud",
        "RTAB-Map trajectory", "RTAB-Map point cloud",
        "Proposed method trajectory", "Proposed method point cloud",
    ]
    # The dedicated first column keeps every sequence label on exactly the
    # same figure-space vertical line, independent of trajectory aspect ratio.
    fig, axes = plt.subplots(
        len(rows), 7, squeeze=False,
        gridspec_kw={"width_ratios": [0.22, 1.0, 1.55, 1.0, 1.55, 1.0, 1.55]},
        figsize=(20.5, max(8.5, 2.9 * len(rows))), constrained_layout=True,
    )
    for row, (sequence, proposed, kiss_slam, rtabmap) in enumerate(rows):
        label_ax = axes[row, 0]
        label_ax.axis("off")
        label_ax.text(
            0.5, 0.5, sequence, transform=label_ax.transAxes,
            ha="center", va="center", rotation=90,
            fontsize=8, fontweight="bold",
        )
        kiss_data = kiss_trajectory_data(kiss_slam, proposed=False)
        proposed_data = kiss_trajectory_data(proposed, proposed=True)
        plot_trajectory(axes[row, 1], kiss_data)
        plot_cloud(fig, axes[row, 2], aligned_cloud(kiss_slam, kiss_data))
        if rtabmap is not None:
            rtab_data = rtab_trajectory_data(rtabmap, proposed, proposed_data)
            plot_trajectory(axes[row, 3], rtab_data)
            plot_cloud(fig, axes[row, 4], aligned_cloud(rtabmap, rtab_data, rtabmap=True))
        else:
            axes[row, 3].axis("off")
            axes[row, 4].axis("off")
        plot_trajectory(axes[row, 5], proposed_data)
        plot_cloud(fig, axes[row, 6], aligned_cloud(proposed, proposed_data))
    # Axis aspect constraints give trajectory and point-cloud panels different
    # top edges. Figure-space headers use one fixed y coordinate so every
    # column label is aligned on exactly the same horizontal baseline.
    fig.canvas.draw()
    header_y = 0.995
    for col, heading in enumerate(headings):
        box = axes[0, col + 1].get_position()
        fig.text(
            (box.x0 + box.x1) / 2, header_y, heading,
            ha="center", va="top", fontsize=9, fontweight="bold",
        )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{dataset}_comparison.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / f"{dataset}_comparison.pdf", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    global ROOT, RUNS, RTAB, DATA, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path.cwd(),
                        help="Directory containing stage_6 saved results")
    parser.add_argument("--output", type=Path, default=Path("paper_artifacts/figures"))
    parser.add_argument("--cloud-cache", type=Path,
                        help="Directory containing saved *-cloud files")
    parser.add_argument(
        "datasets", nargs="*", choices=sorted(DATASETS),
        help="Dataset plates to rebuild; defaults to all datasets.",
    )
    args = parser.parse_args(argv)
    ROOT = args.results_root.expanduser().resolve()
    RUNS = ROOT / "stage_6/runs/artifacts"
    RTAB = ROOT / "stage_6/comparisons/rtabmap/artifacts"
    DATA = (args.cloud_cache or ROOT / "stage_8/draft/figures/comparisons/data").resolve()
    OUT = args.output.expanduser().resolve()
    selected = args.datasets or list(DATASETS)
    for dataset in selected:
        rows = DATASETS[dataset]
        make_dataset_plate(dataset, rows)


if __name__ == "__main__":
    main()

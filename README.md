# GNSS-LiDAR SLAM

This repository packages the paper's proposed method: KISS-SLAM followed by
reliability-gated, frame-associated GNSS factors and one guarded final graph
solve. It includes the original interactive KISS-SLAM/Open3D visualization and
scripts that rebuild all result-bearing paper artifacts (Figures 3–7 and Table
3) from saved results. Figures 1–2 and Tables 1–2 are intentionally excluded.

## Install

The native extension builds g2o and the other C++ dependencies through CMake.
On Ubuntu, install a compiler and Python headers first, then install the package:

```bash
sudo apt-get install build-essential cmake ninja-build python3-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[paper,test]'
```

## Run the proposed method

The first argument is the dataset name and the second is its root folder. When
the root is a directory, give the sequence name explicitly:

```bash
gnss-lidar-slam rtk-slam /data/rtk_slam_dataset \
  --sequence stadtgarten_seq1 --output results

gnss-lidar-slam m2dgr /data/M2DGR \
  --sequence door_01 --output results

gnss-lidar-slam i2nav-robot /data/i2Nav-Robot \
  --sequence building00 --output results
```

You can also pass one `*_euroc.zip` or `.bag` file as `ROOT`; its sequence name
is inferred from the filename. The interactive visualization is on by default:
space starts/pauses, `N` advances one frame, `C` recenters, and Esc exits. Use
`--no-visualize` on a headless machine and `--n-scans 1200` for a bounded run.

Supported input contracts:

- `rtk-slam`: extended-EuRoC ZIP containing `lidar0/data.csv` and
  `gps0/data_raw.csv`.
- `m2dgr`: ROS1 bag with `/velodyne_points` and `/ublox/fix`.
- `i2nav-robot`: ROS1 bag with `/hesai/at128/points` and `/ublox/f9p/fix`.

The CLI prints the resolved dataset, sequence, input file, and output root
before processing. Dataset-level profiles reproduce the paper settings,
including lever arms, covariance gates, anchor cadence, factor subsampling,
fixed RTK-SLAM leveling, and GNSS information scaling. Use
`--config config/example.yaml` for reviewed overrides.

## Recreate paper results

The saved-results root must contain the original result topology:

```text
ROOT/
├── stage_6/runs/{manifests,results,artifacts}/
├── stage_6/comparisons/rtabmap/artifacts/
├── stage_6/runtime_benchmark/aggregate.json
└── stage_8/draft/figures/comparisons/data/  # saved cloud caches
```

Run:

```bash
gnss-lidar-slam-paper --results-root /path/to/saved-results \
  --output paper_artifacts
```

This creates:

- `tables/table3_sequence_accuracy.csv` and `.tex`;
- Figures 3–5: per-dataset trajectory/point-cloud comparisons (`*_comparison`);
- Figure 6: `figure6_sequence_accuracy_bars`;
- Figure 7: `figure7_runtime_1200`;
- `reproduction_manifest.json`, listing every generated file.

Use `--skip-comparisons` when only the aggregate JSON/results are available and
the large trajectory/cloud caches have not been retained. Table 3 and Figures
6–7 still regenerate. The comparison figures use the final optimized graph,
not the online pose stream; KISS-SLAM and proposed clouds must be the saved
`*-optimized-scan-cloud-v2.npy` files, while RTAB-Map clouds are the saved CSVs.

## Method boundary

GNSS positions are converted to local ENU but the final ENU-to-graph planar
calibration is estimated only after the LiDAR frame graph is complete. Reliable
fixes are attached to timestamp-associated frame vertices. Strong temporary
vertical and attitude stabilizers constrain the correction to horizontal
translation. The transaction is rolled back if it produces non-finite poses,
excessive displacement, excessive vertical change, or attitude deformation.

The implementation derives from KISS-SLAM and retains its MIT license and
copyright notices. The new GNSS integration code is distributed under the same
license.


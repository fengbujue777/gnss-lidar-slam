"""Dataset discovery and adapters for the three datasets used in the paper."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .errors import DatasetError
from .geodesy import gnss_fix
from .ros1_standard import gnss_enabled_rosbag_dataset

ALIASES = {
    "rtk-slam": "rtk-slam", "rtkslam": "rtk-slam", "rtk_slam": "rtk-slam",
    "m2dgr": "m2dgr",
    "i2nav-robot": "i2nav-robot", "i2nav": "i2nav-robot", "i2nav_robot": "i2nav-robot",
}

ROS_TOPICS = {
    "m2dgr": ("/velodyne_points", "/ublox/fix"),
    "i2nav-robot": ("/hesai/at128/points", "/ublox/f9p/fix"),
}


def canonical_name(name: str) -> str:
    try:
        return ALIASES[name.strip().lower()]
    except KeyError as exc:
        raise DatasetError(
            f"unsupported dataset {name!r}; choose rtk-slam, m2dgr, or i2nav-robot"
        ) from exc


def resolve_sequence_file(dataset: str, root: Path, sequence: str | None) -> tuple[Path, str]:
    root = root.expanduser().resolve()
    if root.is_file():
        inferred = root.stem.removesuffix("_euroc")
        return root, sequence or inferred
    if not root.is_dir():
        raise DatasetError(f"dataset root does not exist: {root}")
    if not sequence:
        raise DatasetError("--sequence is required when ROOT is a directory")
    patterns = (
        [f"**/{sequence}_euroc.zip", f"**/{sequence}.zip"]
        if dataset == "rtk-slam"
        else [f"**/{sequence}.bag", f"**/{sequence}_*.bag", f"**/*{sequence}*.bag"]
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            break
    if len(matches) != 1:
        raise DatasetError(
            f"expected exactly one input for sequence {sequence!r} below {root}; "
            f"found {len(matches)}"
        )
    return matches[0], sequence


def _member(archive: ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise DatasetError(f"expected one {suffix} member, found {len(matches)}")
    return matches[0]


def _iter_rtk_fixes(path: Path):
    with ZipFile(path) as archive, archive.open(_member(archive, "gps0/data_raw.csv")) as raw:
        for row in csv.DictReader(line.decode("utf-8") for line in raw):
            fix = {
                "timestamp": int(row["timestamp"]) / 1e9,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "altitude": float(row["altitude"]),
                "horizontal_error": float(row["horizontal_error"]),
                "vertical_error": float(row["vertical_error"]),
                "n_sat": int(row["n_sat"]),
            }
            fix["valid"] = (
                all(math.isfinite(fix[key]) for key in (
                    "latitude", "longitude", "altitude", "horizontal_error", "vertical_error"
                ))
                and -90 <= fix["latitude"] <= 90 and -180 <= fix["longitude"] <= 180
                and (fix["latitude"] != 0 or fix["longitude"] != 0)
                and fix["horizontal_error"] > 0 and fix["vertical_error"] > 0
                and fix["n_sat"] > 0
            )
            yield fix


def _lidar_epoch_inventory(path: Path, fixes: list[dict], cache_dir: Path) -> list[int]:
    identity = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    cache = cache_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("source_identity") == identity:
            return [int(item) for item in payload["valid_fix_indices"]]
    times = [item["timestamp"] for item in fixes]
    if not times:
        raise DatasetError("RTK-SLAM archive contains no GNSS epochs")
    present = bytearray(len(times)); index = 0
    with ZipFile(path) as archive, archive.open(_member(archive, "lidar0/data.csv")) as raw:
        next(raw, None)
        for line in raw:
            stamp = int(line.split(b",", 1)[0]) / 1e9
            while index + 1 < len(times) and stamp >= (times[index] + times[index + 1]) / 2:
                index += 1
            upper = (times[index] + times[index + 1]) / 2 if index + 1 < len(times) else times[index] + 0.05
            if times[index] - 0.05 <= stamp < upper:
                present[index] = 1
    indices = [i for i, value in enumerate(present) if value]
    if not indices:
        raise DatasetError("RTK-SLAM archive contains no LiDAR epochs matched to GNSS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"source_identity": identity, "valid_fix_indices": indices}) + "\n")
    return indices


class RTKSlamDataset:
    """Sequential extended-EuRoC ZIP reader keyed by the receiver epoch grid."""

    def __init__(self, path: Path, sequence: str, config: dict, cache_dir: Path):
        self.path, self.sequence_id, self.data_dir = path, sequence, str(path)
        all_fixes = list(_iter_rtk_fixes(path))
        indices = _lidar_epoch_inventory(path, all_fixes, cache_dir)
        self.fixes = [all_fixes[index] for index in indices]
        self.timestamps = [item["timestamp"] for item in self.fixes]
        self.config = config
        valid = next((item for item in self.fixes if item["valid"]), None)
        if valid is None:
            raise DatasetError("RTK-SLAM sequence contains no valid GNSS fix")
        self.config["enu_origin_lla"] = [
            valid["latitude"], valid["longitude"], valid["altitude"]
        ]
        self._open()

    def _open(self):
        self._archive = ZipFile(self.path)
        raw = self._archive.open(_member(self._archive, "lidar0/data.csv"))
        self._text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        self._rows = csv.reader(self._text); next(self._rows, None)
        self._pending, self._next_index, self._returned_timestamps = None, 0, []

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        if self._next_index == 0 and index > 0:
            self._next_index = index
        if index != self._next_index:
            raise DatasetError("RTK-SLAM ZIP reader requires sequential access")
        center = self.timestamps[index]
        lower = (self.timestamps[index - 1] + center) / 2 if index else center - 0.05
        upper = (center + self.timestamps[index + 1]) / 2 if index + 1 < len(self) else center + 0.05
        points, absolute_times = [], []
        row = self._pending
        while True:
            if row is None:
                try:
                    row = next(self._rows)
                except StopIteration:
                    break
            timestamp = int(row[0]) / 1e9
            if timestamp >= upper:
                self._pending = row; break
            if timestamp >= lower:
                points.append([float(row[1]), float(row[2]), float(row[3])])
                absolute_times.append(timestamp)
            row = None
        self._next_index += 1; self._returned_timestamps.append(center)
        if not points:
            raise DatasetError(f"no LiDAR points matched GNSS epoch {index}")
        phases = np.clip((np.asarray(absolute_times) - lower) / (upper - lower), 0, 1)
        return np.asarray(points, dtype=np.float64), phases

    def get_frames_timestamps(self):
        return list(self._returned_timestamps)

    def get_frame_timestamp(self, index):
        return self.timestamps[index]

    def get_gnss_fix(self, index):
        item = self.fixes[index]
        if not item["valid"]:
            return None
        covariance = np.diag([
            item["horizontal_error"] ** 2, item["horizontal_error"] ** 2,
            item["vertical_error"] ** 2,
        ])
        return gnss_fix({**item, "covariance": covariance}, self.config)

    def reset(self):
        self._text.close(); self._archive.close(); self._open()


def make_dataset(name: str, root: Path, sequence: str | None, config: dict, cache_dir: Path):
    dataset = canonical_name(name)
    path, sequence = resolve_sequence_file(dataset, root, sequence)
    if dataset == "rtk-slam":
        if path.suffix.lower() != ".zip":
            raise DatasetError("RTK-SLAM input must be an extended-EuRoC ZIP")
        return RTKSlamDataset(path, sequence, config, cache_dir), dataset, sequence, path
    if path.suffix.lower() != ".bag":
        raise DatasetError(f"{dataset} input must be a ROS1 .bag")
    lidar_topic, gnss_topic = ROS_TOPICS[dataset]
    wrapped = gnss_enabled_rosbag_dataset(path, lidar_topic, gnss_topic, config)
    return wrapped, dataset, sequence, path


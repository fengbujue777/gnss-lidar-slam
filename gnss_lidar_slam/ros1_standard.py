"""ROS1 standard-message decoding used by M2DGR and i2Nav adapters."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np

from .errors import DatasetError


_POINT_FIELD_DTYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8",
}


def stamp_seconds(header) -> float:
    stamp = header.stamp
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def pointcloud_xyz(message) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    missing = {name for name in ("x", "y", "z") if name not in fields}
    if missing:
        raise DatasetError(f"PointCloud2 lacks fields: {', '.join(sorted(missing))}")
    endian = ">" if message.is_bigendian else "<"
    dtype_fields = []
    for name in ("x", "y", "z"):
        field = fields[name]
        code = _POINT_FIELD_DTYPES.get(field.datatype)
        if code not in {"f4", "f8"} or field.count != 1:
            raise DatasetError(f"PointCloud2 {name} must be one float32/float64 value")
        dtype_fields.append((name, endian + code, field.offset))
    dtype = np.dtype({"names": [x[0] for x in dtype_fields], "formats": [x[1] for x in dtype_fields], "offsets": [x[2] for x in dtype_fields], "itemsize": message.point_step})
    count = int(message.width) * int(message.height)
    records = np.frombuffer(message.data, dtype=dtype, count=count)
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64, copy=False)
    return points[np.all(np.isfinite(points), axis=1)]


def navsatfix_record(message) -> dict | None:
    # sensor_msgs/NavSatStatus.STATUS_NO_FIX is -1. Preserve it as missing.
    if int(message.status.status) < 0:
        return None
    values = (float(message.latitude), float(message.longitude), float(message.altitude))
    if not all(math.isfinite(value) for value in values):
        return None
    covariance = np.asarray(message.position_covariance, dtype=float).reshape(3, 3)
    covariance_value = covariance if np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0) else None
    return {"timestamp": stamp_seconds(message.header), "latitude": values[0], "longitude": values[1], "altitude": values[2], "covariance": covariance_value, "covariance_type": int(message.position_covariance_type), "frame_id": message.header.frame_id}


def iter_standard_messages(path: Path, lidar_topic: str, gnss_topic: str) -> Iterator[tuple[str, object]]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise DatasetError("rosbags is required to read M2DGR and i2Nav ROS1 bags") from exc
    with AnyReader([path]) as reader:
        selected = [connection for connection in reader.connections if connection.topic in {lidar_topic, gnss_topic}]
        found = {connection.topic: connection.msgtype for connection in selected}
        if lidar_topic not in found or gnss_topic not in found:
            raise DatasetError(f"bag topics do not match contract: found {found}")
        for connection, _, rawdata in reader.messages(connections=selected):
            yield connection.topic, reader.deserialize(rawdata, connection.msgtype)


def load_navsatfix_records(path: Path, topic: str) -> list[dict]:
    records = []
    for message_topic, message in iter_standard_messages(path, topic, topic):
        record = navsatfix_record(message)
        if record is not None: records.append(record)
    return records


def gnss_enabled_rosbag_dataset(path: Path, lidar_topic: str, gnss_topic: str, config: dict):
    from bisect import bisect_left
    from kiss_icp.datasets.rosbag import RosbagDataset
    from .geodesy import gnss_fix
    class Dataset(RosbagDataset):
        def __init__(self):
            super().__init__(path, lidar_topic)
            self.gnss_records = load_navsatfix_records(path, gnss_topic)
            if config.get("enu_origin_policy") == "first_valid_fix" and self.gnss_records:
                first = self.gnss_records[0]
                config["enu_origin_lla"] = [first["latitude"], first["longitude"], first["altitude"]]
            offset = float(config.get("timestamp_offset_s", 0.0))
            self.gnss_times = [item["timestamp"] + offset for item in self.gnss_records]
            self._used_gnss_indices = set()
        def get_gnss_fix(self, index):
            if not self.timestamps or not self.gnss_times: return None
            stamp = self.timestamps[-1]; location = bisect_left(self.gnss_times, stamp)
            candidates = [i for i in (location - 1, location) if 0 <= i < len(self.gnss_times) and i not in self._used_gnss_indices]
            if not candidates: return None
            nearest = min(candidates, key=lambda i: abs(self.gnss_times[i] - stamp))
            if abs(self.gnss_times[nearest] - stamp) > float(config["max_timestamp_offset_s"]): return None
            self._used_gnss_indices.add(nearest)
            return gnss_fix(self.gnss_records[nearest], config)
        def get_frame_timestamp(self, index):
            if not self.timestamps:
                raise DatasetError("ROS bag frame timestamp is unavailable after scan decoding")
            return float(self.timestamps[-1])
        def get_replay_gnss_series(self, frame_count=None):
            """Read timestamps and associate GNSS without decoding point arrays."""
            lidar_times = []
            for message_topic, message in iter_standard_messages(
                path, lidar_topic, lidar_topic
            ):
                if message_topic != lidar_topic:
                    continue
                lidar_times.append(stamp_seconds(message.header))
                if frame_count is not None and len(lidar_times) >= int(frame_count):
                    break
            used = set(); fixes = []; tolerance = float(config["max_timestamp_offset_s"])
            for stamp in lidar_times:
                location = bisect_left(self.gnss_times, stamp)
                candidates = [
                    item for item in (location - 1, location)
                    if 0 <= item < len(self.gnss_times) and item not in used
                ]
                if not candidates:
                    fixes.append(None); continue
                nearest = min(candidates, key=lambda item: abs(self.gnss_times[item] - stamp))
                if abs(self.gnss_times[nearest] - stamp) > tolerance:
                    fixes.append(None); continue
                used.add(nearest)
                fixes.append(gnss_fix(self.gnss_records[nearest], config))
            self.timestamps = list(lidar_times)
            return lidar_times, fixes
        def yaw_calibration_records(self):
            return list(self.gnss_records)
        def skip_prefix_scans(self, count):
            for index in range(int(count)):
                super(Dataset, self).__getitem__(index)
            self.timestamps = []
        def reset(self):
            super().reset(); self._used_gnss_indices.clear()
    return Dataset()

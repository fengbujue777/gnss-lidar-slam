from pathlib import Path

import pytest

from gnss_lidar_slam.datasets import canonical_name, resolve_sequence_file
from gnss_lidar_slam.errors import DatasetError
from gnss_lidar_slam.profiles import profile


def test_aliases_and_frozen_profiles_are_independent():
    assert canonical_name("RTK_SLAM") == "rtk-slam"
    first = profile("m2dgr")
    second = profile("m2dgr")
    first["lever_arm_body_m"][0] = 99
    assert second["lever_arm_body_m"][0] == pytest.approx(-0.09825)
    assert second["batch_final_calibration_enabled"] is True


def test_sequence_discovery(tmp_path: Path):
    bag = tmp_path / "nested" / "door_01.bag"
    bag.parent.mkdir(); bag.touch()
    path, sequence = resolve_sequence_file("m2dgr", tmp_path, "door_01")
    assert path == bag
    assert sequence == "door_01"


def test_directory_requires_sequence(tmp_path: Path):
    with pytest.raises(DatasetError, match="--sequence"):
        resolve_sequence_file("i2nav-robot", tmp_path, None)


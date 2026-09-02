# MIT License

# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from kiss_icp.config.config import (
    AdaptiveThresholdConfig,
    DataConfig,
    MappingConfig,
    RegistrationConfig,
)
from kiss_icp.config.parser import KISSConfig
from map_closures.config.config import MapClosuresConfig
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KissOdometryConfig(BaseModel):
    preprocessing: DataConfig = DataConfig()
    registration: RegistrationConfig = RegistrationConfig()
    mapping: MappingConfig = MappingConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


class LoopCloserConfig(BaseModel):
    detector: MapClosuresConfig = MapClosuresConfig()
    overlap_threshold: float = 0.4


class LocalMapperConfig(BaseModel):
    voxel_size: float = 0.5
    splitting_distance: float = 100.0


class OccupancyMapperConfig(BaseModel):
    free_threshold: float = 0.2
    occupied_threshold: float = 0.65
    resolution: float = 0.5
    max_range: Optional[float] = None
    z_min: float = 0.1
    z_max: float = 0.5


class PoseGraphOptimizerConfig(BaseModel):
    max_iterations: int = 10


class GnssConfig(BaseModel):
    """Reliability gate and map-frame publication settings for processed GNSS fixes."""

    enabled: bool = False
    max_hdop: float = Field(default=2.5, gt=0.0)
    max_timestamp_offset_s: float = Field(default=0.2, ge=0.0)
    queue_size: int = Field(default=12, ge=3)
    ransac_min_samples: int = Field(default=4, ge=2)
    ransac_iterations: int = Field(default=32, ge=1)
    ransac_residual_threshold_m: float = Field(default=2.0, gt=0.0)
    min_position_sigma_m: float = Field(default=0.5, gt=0.0)
    max_position_sigma_m: float = Field(default=2.5, gt=0.0)
    hdop_sigma_scale_m: float = Field(default=1.0, gt=0.0)
    horizontal_only: bool = True
    profile_id: str = "generic-horizontal-v1"
    horizontal_optimization_strategy: str = "constrained_xy_preserve_lidar_z_attitude"
    vertical_stabilizer_information: float = Field(default=1e5, gt=0.0)
    attitude_stabilizer_information: float = Field(default=1e5, gt=0.0)
    max_anchor_attitude_shift_deg: float = Field(default=0.1, gt=0.0)
    lever_arm_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outage_timeout_s: float = Field(default=2.0, gt=0.0)
    reentry_stable_fixes: int = Field(default=3, ge=1)
    reentry_max_translation_step_m: float = Field(default=0.5, gt=0.0)
    max_anchor_innovation_m: float = Field(default=10.0, gt=0.0)
    max_recovery_anchor_innovation_m: float = Field(default=50.0, gt=0.0)
    recovery_anchor_consistency_m: float = Field(default=3.0, gt=0.0)
    max_anchor_optimization_shift_m: float = Field(default=50.0, gt=0.0)
    max_anchor_vertical_shift_m: float = Field(default=1.0, gt=0.0)
    anchor_robust_kernel_delta: float = Field(default=2.0, gt=0.0)
    min_anchor_nodes_for_optimization: int = Field(default=3, ge=3)
    anchor_interval_s: float = Field(default=10.0, gt=0.0)
    anchor_min_travel_m: float = Field(default=5.0, gt=0.0)
    recursive_calibration_enabled: bool = False
    initial_yaw_alignment_rad: float = 0.0
    initial_map_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    recursive_calibration_min_anchors: int = Field(default=3, ge=3)
    recursive_calibration_min_baseline_m: float = Field(default=10.0, gt=0.0)
    recursive_calibration_residual_threshold_m: float = Field(default=3.0, gt=0.0)
    recursive_calibration_max_yaw_step_deg: float = Field(default=5.0, gt=0.0)
    recursive_calibration_max_translation_step_m: float = Field(default=10.0, gt=0.0)
    batch_final_calibration_enabled: bool = False
    batch_calibration_min_observations: int = Field(default=20, ge=3)
    batch_calibration_min_baseline_m: float = Field(default=10.0, gt=0.0)
    batch_calibration_residual_threshold_m: float = Field(default=3.0, gt=0.0)
    batch_max_position_factors: int = Field(default=0, ge=0)
    batch_factor_selection_policy: str = "uniform_trajectory_distance"
    set_lidar_roll_pitch: bool = False
    # KISS map -> ENU leveling angles. The ENU -> map GNSS transform uses
    # their inverse before applying the estimated planar yaw.
    lidar_roll_deg: float = 0.0
    lidar_pitch_deg: float = 0.0
    gnss_information_scale: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_ransac_window(self):
        if self.ransac_min_samples > self.queue_size:
            raise ValueError("ransac_min_samples must not exceed queue_size")
        if self.max_position_sigma_m < self.min_position_sigma_m:
            raise ValueError("max_position_sigma_m must not be below min_position_sigma_m")
        if self.max_recovery_anchor_innovation_m < self.max_anchor_innovation_m:
            raise ValueError("recovery innovation limit must not be below tracking limit")
        if self.horizontal_optimization_strategy not in {
            "constrained_xy_preserve_lidar_z_attitude",
            "unconstrained_se3",
        }:
            raise ValueError("unsupported horizontal GNSS optimization strategy")
        if self.batch_factor_selection_policy not in {
            "uniform_trajectory_distance",
        }:
            raise ValueError("unsupported batch GNSS factor selection policy")
        if not self.set_lidar_roll_pitch and (
            self.lidar_roll_deg != 0.0 or self.lidar_pitch_deg != 0.0
        ):
            raise ValueError(
                "lidar roll/pitch angles require set_lidar_roll_pitch=true"
            )
        return self


class KissSLAMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="kiss_slam_")
    out_dir: str = "slam_output"
    odometry: KissOdometryConfig = KissOdometryConfig()
    local_mapper: LocalMapperConfig = LocalMapperConfig()
    occupancy_mapper: OccupancyMapperConfig = OccupancyMapperConfig()
    loop_closer: LoopCloserConfig = LoopCloserConfig()
    pose_graph_optimizer: PoseGraphOptimizerConfig = PoseGraphOptimizerConfig()
    gnss: GnssConfig = GnssConfig()

    def kiss_icp_config(self) -> KISSConfig:
        return KISSConfig(
            out_dir=self.out_dir,
            data=self.odometry.preprocessing,
            registration=self.odometry.registration,
            mapping=self.odometry.mapping,
            adaptive_threshold=self.odometry.adaptive_threshold,
        )


class KissDumper(yaml.Dumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    data = None
    if config_file is not None:
        with open(config_file) as cfg_file:
            data = yaml.safe_load(cfg_file)
    return data or {}


def load_config(config_file: Optional[Path]) -> KissSLAMConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = KissSLAMConfig(**_yaml_source(config_file))

    # Use specified voxel size or compute one using the max range
    if config.odometry.mapping.voxel_size is None:
        config.odometry.mapping.voxel_size = float(config.odometry.preprocessing.max_range / 100.0)

    if config.occupancy_mapper.max_range is None:
        config.occupancy_mapper.max_range = config.odometry.preprocessing.max_range

    return config


def write_config(config: KissSLAMConfig = KissSLAMConfig(), filename: str = "kiss_slam.yaml"):
    with open(filename, "w") as outfile:
        yaml.dump(
            config.model_dump(),
            outfile,
            Dumper=KissDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=4,
        )

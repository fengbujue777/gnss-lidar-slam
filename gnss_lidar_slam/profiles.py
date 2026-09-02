"""Frozen dataset-level profiles used for the paper's proposed method."""
from __future__ import annotations

import copy

COMMON = {
    "enabled": True,
    "queue_size": 12,
    "ransac_min_samples": 4,
    "ransac_iterations": 12,
    "min_position_sigma_m": 0.5,
    "horizontal_only": True,
    "outage_timeout_s": 2.0,
    "reentry_stable_fixes": 3,
    "reentry_max_translation_step_m": 0.5,
    "max_anchor_innovation_m": 10.0,
    "max_recovery_anchor_innovation_m": 50.0,
    "recovery_anchor_consistency_m": 3.0,
    "max_anchor_optimization_shift_m": 50.0,
    "max_anchor_vertical_shift_m": 1.0,
    "min_anchor_nodes_for_optimization": 3,
    "recursive_calibration_enabled": False,
    "batch_final_calibration_enabled": True,
    "batch_calibration_min_observations": 20,
    "batch_calibration_min_baseline_m": 10.0,
    "batch_calibration_residual_threshold_m": 3.0,
    "batch_factor_selection_policy": "uniform_trajectory_distance",
    "horizontal_optimization_strategy": "constrained_xy_preserve_lidar_z_attitude",
    "vertical_stabilizer_information": 100000.0,
    "attitude_stabilizer_information": 100000.0,
    "max_anchor_attitude_shift_deg": 0.1,
}

PROFILES = {
    "rtk-slam": {
        "profile_id": "rtk-slam-surveyed-rtk-v1", "max_timestamp_offset_s": 0.06,
        "max_position_sigma_m": 2.5, "ransac_residual_threshold_m": 2.0,
        "anchor_robust_kernel_delta": 2.0, "anchor_interval_s": 10.0,
        "anchor_min_travel_m": 5.0, "batch_max_position_factors": 30,
        "set_lidar_roll_pitch": True, "lidar_roll_deg": -5.50,
        "lidar_pitch_deg": -15.11, "gnss_information_scale": 0.01,
        "lever_arm_body_m": [0.034, 0.0, 0.046],
    },
    "m2dgr": {
        "profile_id": "m2dgr-urban-covariance-v1", "max_timestamp_offset_s": 0.12,
        "max_position_sigma_m": 5.0, "ransac_residual_threshold_m": 3.0,
        "anchor_robust_kernel_delta": 1.0, "anchor_interval_s": 7.5,
        "anchor_min_travel_m": 4.0, "batch_max_position_factors": 10,
        "set_lidar_roll_pitch": False, "gnss_information_scale": 0.001,
        "lever_arm_body_m": [-0.09825, 0.00582, 0.72673],
    },
    "i2nav-robot": {
        "profile_id": "i2nav-robot-navsat-v1", "max_timestamp_offset_s": 0.12,
        "max_position_sigma_m": 2.5, "ransac_residual_threshold_m": 2.0,
        "anchor_robust_kernel_delta": 1.5, "anchor_interval_s": 5.0,
        "anchor_min_travel_m": 3.0, "batch_max_position_factors": 0,
        "set_lidar_roll_pitch": False, "gnss_information_scale": 1.0,
        "lever_arm_body_m": [-0.03917706, 0.05191214, 0.30239426],
    },
}


def profile(name: str) -> dict:
    return {**copy.deepcopy(COMMON), **copy.deepcopy(PROFILES[name])}


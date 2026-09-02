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
from collections import deque
from collections import Counter
from dataclasses import replace

import numpy as np
from kiss_icp.kiss_icp import KissICP
from kiss_icp.voxelization import voxel_down_sample

from kiss_slam.config import KissSLAMConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.loop_closer import LoopCloser
from kiss_slam.gnss import GnssFix, GnssReliabilityGate, SmoothReentryTracker
from kiss_slam.pose_graph_optimizer import PoseGraphOptimizer
from kiss_slam.voxel_map import VoxelMap


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class KissSLAM:
    def __init__(self, config: KissSLAMConfig):
        self.config = config
        self.odometry = KissICP(config.kiss_icp_config())
        self.closer = LoopCloser(config.loop_closer)
        local_map_config = self.config.local_mapper
        self.local_map_voxel_size = local_map_config.voxel_size
        self.voxel_grid = VoxelMap(self.local_map_voxel_size)
        self.local_map_graph = LocalMapGraph()
        self.local_map_splitting_distance = local_map_config.splitting_distance
        self.optimizer = PoseGraphOptimizer(config.pose_graph_optimizer)
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        self.optimizer.fix_variable(self.local_map_graph.last_id)
        self.closures = []
        self._initialize_gnss_state(config)

    def _initialize_gnss_state(self, config):
        """Initialize GNSS state independently of LiDAR front-end objects."""
        self.gnss_gate = GnssReliabilityGate(config.gnss)
        self.reentry_tracker = SmoothReentryTracker(config.gnss)
        # This interval buffer is intentionally independent of the short RANSAC
        # queue. Every reliability-approved fix contributes to one robust graph
        # observation instead of being silently dropped when the queue fills.
        self.pending_gnss_anchors = []
        self.gnss_anchor_ids = []
        self.gnss_anchor_events = []
        self.gnss_anchor_decision_reasons = Counter()
        self.staged_gnss_anchors = []
        self.gnss_anchor_graph_observable = False
        self.last_gnss_anchor_node_timestamp = None
        self.last_reliable_fix_timestamp = None
        self.published_poses = []
        self.gnss_epochs = 0
        self.gnss_missing_epochs = 0
        self.gnss_decision_reasons = Counter()
        self.gnss_calibration_yaw_rad = float(
            config.gnss.initial_yaw_alignment_rad
        )
        self.gnss_calibration_translation_m = np.asarray(
            config.gnss.initial_map_translation_m, dtype=float
        ).copy()
        self.gnss_calibration_history = [{
            "reason": "prefix_initialization",
            "yaw_alignment_rad": self.gnss_calibration_yaw_rad,
            "map_translation_m": self.gnss_calibration_translation_m.tolist(),
        }]
        self.batch_gnss_observations = []
        self.batch_gnss_optimization = {
            "enabled": bool(config.gnss.batch_final_calibration_enabled),
            "executed": False,
        }

    def get_closures(self):
        return self.closures

    def get_keyposes(self):
        return list(self.local_map_graph.keyposes())

    def process_scan(
        self, frame, timestamps, gnss_fix=None, scan_timestamp=None, force_node=False
    ):
        deskewed_frame, _ = self.odometry.register_frame(frame, timestamps)
        current_pose = self.odometry.last_pose
        mapping_frame = voxel_down_sample(deskewed_frame, self.local_map_voxel_size)
        self.voxel_grid.integrate_frame(mapping_frame, current_pose)
        self.local_map_graph.last_local_map.local_trajectory.append(current_pose)
        scan_timestamp = (
            float(scan_timestamp) if self.config.gnss.enabled and scan_timestamp is not None
            else self._scan_timestamp(timestamps) if self.config.gnss.enabled
            else None
        )
        raw_map_pose = self.local_map_graph.last_keypose @ current_pose
        dead_reckoned_map_pose = np.copy(raw_map_pose)
        if self.config.gnss.enabled:
            self.gnss_epochs += 1
            if gnss_fix is None:
                self.gnss_missing_epochs += 1
                self.gnss_decision_reasons["missing_fix"] += 1
            else:
                if isinstance(gnss_fix, dict):
                    gnss_fix = GnssFix(**gnss_fix)
                gnss_fix = self._apply_current_gnss_calibration(gnss_fix)
                decision = self.gnss_gate.evaluate(gnss_fix, scan_timestamp, raw_map_pose)
                self.gnss_decision_reasons[decision.reason] += 1
                if decision.accepted:
                    if (
                        self.last_reliable_fix_timestamp is not None
                        and decision.timestamp - self.last_reliable_fix_timestamp
                        > self.config.gnss.outage_timeout_s
                    ):
                        self._begin_gnss_recovery()
                    self.last_reliable_fix_timestamp = decision.timestamp
                    if self.config.gnss.batch_final_calibration_enabled:
                        self.batch_gnss_observations.append({
                            "frame_index": len(self.published_poses),
                            "timestamp": float(decision.timestamp),
                            "node_id": int(self.local_map_graph.last_id),
                            "relative_pose": np.asarray(current_pose, dtype=float).copy(),
                            "raw_enu_position": np.asarray(
                                decision.raw_position, dtype=float
                            ).copy(),
                            "information": np.asarray(
                                decision.information, dtype=float
                            ).copy(),
                        })
                    else:
                        self.pending_gnss_anchors.append(decision)
                        self.reentry_tracker.observe_fix(decision.timestamp)
        traveled_distance = np.linalg.norm(current_pose[:3, -1])
        gnss_node_due = self._gnss_anchor_node_due(
            scan_timestamp, traveled_distance
        )
        if (
            traveled_distance > self.local_map_splitting_distance
            or gnss_node_due
            or force_node
        ):
            self.generate_new_node(scan_timestamp)
            raw_map_pose = self.local_map_graph.last_keypose @ self.odometry.last_pose
        published_pose = (
            self.reentry_tracker.publish(
                raw_map_pose, scan_timestamp, dead_reckoned_map_pose
            )
            if self.config.gnss.enabled
            and not self.config.gnss.batch_final_calibration_enabled
            else np.copy(raw_map_pose)
        )
        self.published_poses.append(published_pose)
        return published_pose

    def _begin_gnss_recovery(self):
        # Never mix fixes collected before an outage with the newly reacquired
        # solution.  Their common median could manufacture an anchor that no
        # receiver epoch actually supports.
        if self.pending_gnss_anchors:
            self.gnss_anchor_decision_reasons["outage_buffer_reset"] += len(
                self.pending_gnss_anchors
            )
            self.pending_gnss_anchors.clear()
        if self.staged_gnss_anchors:
            for _, _, _, event in self.staged_gnss_anchors:
                event["accepted"] = False
                event["factor_added"] = False
                event["reason"] = "observability_reset"
            self.gnss_anchor_decision_reasons["observability_reset"] += len(
                self.staged_gnss_anchors
            )
            self.staged_gnss_anchors.clear()
        self.gnss_anchor_graph_observable = False

    def _gnss_anchor_node_due(self, timestamp, traveled_distance):
        if (
            not self.config.gnss.enabled
            or timestamp is None
            or not self.pending_gnss_anchors
            or traveled_distance < self.config.gnss.anchor_min_travel_m
        ):
            return False
        reference = self.last_gnss_anchor_node_timestamp
        if reference is None:
            reference = self.pending_gnss_anchors[0].timestamp
        return float(timestamp) - float(reference) >= self.config.gnss.anchor_interval_s

    @staticmethod
    def _scan_timestamp(timestamps):
        values = np.atleast_1d(np.asarray(timestamps, dtype=float))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("scan timestamps must contain at least one finite value")
        return float(finite[-1])

    def compute_closures(self, query_id, query):
        is_good, source_id, target_id, pose_constraint = self.closer.compute(
            query_id, query, self.local_map_graph
        )
        if is_good:
            self.closures.append((source_id, target_id))
            self.optimizer.add_factor(source_id, target_id, pose_constraint, np.eye(6))
            self.optimize_pose_graph()

    @staticmethod
    def _planar_rotation(yaw):
        cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
        return np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def _fixed_lidar_tilt_rotation(self):
        """Return the fixed ENU-level -> tilted KISS-map rotation.

        Configured roll/pitch describe the inverse map -> ENU leveling
        rotation obtained from reference alignment.  GNSS points travel in
        the opposite direction, hence the negated angles and inverse order.
        """
        config = self.config.gnss
        if not getattr(config, "set_lidar_roll_pitch", False):
            return np.eye(3)
        roll = np.radians(-float(config.lidar_roll_deg))
        pitch = np.radians(-float(config.lidar_pitch_deg))
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rotation_x = np.array([
            [1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr],
        ])
        rotation_y = np.array([
            [cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp],
        ])
        return rotation_x @ rotation_y

    def _gnss_rotation(self, yaw):
        """Compose estimated ENU yaw with fixed dataset-level LiDAR tilt."""
        return self._fixed_lidar_tilt_rotation() @ self._planar_rotation(yaw)

    @staticmethod
    def _wrapped_angle(value):
        return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)

    def _apply_current_gnss_calibration(self, fix):
        if (
            not self.config.gnss.recursive_calibration_enabled
            or fix.raw_position is None
        ):
            return fix
        raw = np.asarray(fix.raw_position, dtype=float)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            return fix
        rotation = self._gnss_rotation(self.gnss_calibration_yaw_rad)
        position = rotation @ raw + self.gnss_calibration_translation_m
        covariance = fix.covariance
        if covariance is not None:
            covariance = rotation @ np.asarray(covariance, dtype=float) @ rotation.T
        return replace(fix, position=position, covariance=covariance)

    def _fit_recursive_calibration(self, reference_estimates):
        config = self.config.gnss
        usable = [
            event for event in self.gnss_anchor_events
            if event.get("factor_added") and event.get("raw_enu_position") is not None
            and event.get("node_id") in reference_estimates
        ]
        if len(usable) < config.recursive_calibration_min_anchors:
            return None
        raw = np.asarray([event["raw_enu_position"][:2] for event in usable])
        lever = np.asarray(config.lever_arm_body_m, dtype=float)
        target = np.asarray([
            np.asarray(reference_estimates[event["node_id"]], dtype=float)[:3, 3]
            + np.asarray(reference_estimates[event["node_id"]], dtype=float)[:3, :3]
            @ lever
            for event in usable
        ])[:, :2]
        if max(np.ptp(raw, axis=0)) < config.recursive_calibration_min_baseline_m:
            return None

        inliers = np.ones(len(raw), dtype=bool)
        threshold = config.recursive_calibration_residual_threshold_m
        fitted_yaw = self.gnss_calibration_yaw_rad
        fitted_translation = self.gnss_calibration_translation_m[:2].copy()
        for _ in range(4):
            if np.count_nonzero(inliers) < config.recursive_calibration_min_anchors:
                return None
            source = raw[inliers]
            destination = target[inliers]
            source_center = np.mean(source, axis=0)
            destination_center = np.mean(destination, axis=0)
            left, _, right_t = np.linalg.svd(
                (source - source_center).T @ (destination - destination_center)
            )
            row_rotation = left @ right_t
            if np.linalg.det(row_rotation) < 0.0:
                left[:, -1] *= -1.0
                row_rotation = left @ right_t
            column_rotation = row_rotation.T
            fitted_yaw = float(
                np.arctan2(column_rotation[1, 0], column_rotation[0, 0])
            )
            fitted_translation = np.median(
                destination - source @ row_rotation, axis=0
            )
            residuals = np.linalg.norm(
                raw @ row_rotation + fitted_translation - target, axis=1
            )
            updated = residuals <= threshold
            if np.array_equal(updated, inliers):
                break
            inliers = updated

        yaw_delta = self._wrapped_angle(
            fitted_yaw - self.gnss_calibration_yaw_rad
        )
        yaw_limit = np.radians(config.recursive_calibration_max_yaw_step_deg)
        yaw_delta = float(np.clip(yaw_delta, -yaw_limit, yaw_limit))
        updated_yaw = self._wrapped_angle(
            self.gnss_calibration_yaw_rad + yaw_delta
        )
        updated_rotation = self._planar_rotation(updated_yaw)[:2, :2]
        requested_translation = np.median(
            target[inliers] - raw[inliers] @ updated_rotation.T, axis=0
        )
        translation_delta = (
            requested_translation - self.gnss_calibration_translation_m[:2]
        )
        translation_norm = float(np.linalg.norm(translation_delta))
        translation_limit = config.recursive_calibration_max_translation_step_m
        if translation_norm > translation_limit:
            translation_delta *= translation_limit / translation_norm
        updated_translation = self.gnss_calibration_translation_m.copy()
        updated_translation[:2] += translation_delta
        old_rotation = self._planar_rotation(self.gnss_calibration_yaw_rad)[:2, :2]
        old_residuals = np.linalg.norm(
            raw @ old_rotation.T + self.gnss_calibration_translation_m[:2] - target,
            axis=1,
        )
        new_residuals = np.linalg.norm(
            raw @ updated_rotation.T + updated_translation[:2] - target, axis=1
        )
        return {
            "yaw_alignment_rad": updated_yaw,
            "map_translation_m": updated_translation,
            "anchor_count": len(usable),
            "inlier_count": int(np.count_nonzero(inliers)),
            "yaw_step_deg": float(np.degrees(yaw_delta)),
            "translation_step_m": float(np.linalg.norm(translation_delta)),
            "horizontal_rmse_before_m": float(np.sqrt(np.mean(old_residuals**2))),
            "horizontal_rmse_after_m": float(np.sqrt(np.mean(new_residuals**2))),
        }

    def _fit_batch_calibration(
        self, reference_estimates, observation_pose_key="node_id"
    ):
        """Fit one robust global ENU-to-graph transform after LiDAR SLAM.

        ``node_id`` targets the coarse local-map graph and composes the stored
        frame-relative pose. ``frame_index`` targets the final fine-grained
        graph directly. The latter is used by batch-final optimization so that
        each reliable GNSS epoch constrains its timestamp-associated LiDAR
        frame rather than repeatedly constraining one local-map keypose.
        """
        config = self.config.gnss
        usable = [
            item for item in self.batch_gnss_observations
            if item[observation_pose_key] in reference_estimates
        ]
        if len(usable) < config.batch_calibration_min_observations:
            return None
        raw_xyz = np.asarray([item["raw_enu_position"] for item in usable])
        raw = raw_xyz[:, :2]
        lever = np.asarray(config.lever_arm_body_m, dtype=float)
        target = []
        for item in usable:
            graph_pose = np.asarray(
                reference_estimates[item[observation_pose_key]], dtype=float
            )
            body_pose = (
                graph_pose @ np.asarray(item["relative_pose"], dtype=float)
                if observation_pose_key == "node_id"
                else graph_pose
            )
            target.append(body_pose[:3, 3] + body_pose[:3, :3] @ lever)
        target_xyz = np.asarray(target)
        # Remove the fixed tilt before estimating the remaining planar yaw.
        # Translation remains free, so rotating the target frame preserves the
        # Procrustes objective while making the 2-D SVD physically valid.
        tilt = self._fixed_lidar_tilt_rotation()
        target_level = target_xyz @ tilt
        target = target_level[:, :2]
        if max(np.ptp(raw, axis=0)) < config.batch_calibration_min_baseline_m:
            return None

        inliers = np.ones(len(raw), dtype=bool)
        threshold = config.batch_calibration_residual_threshold_m
        yaw = self.gnss_calibration_yaw_rad
        translation = self.gnss_calibration_translation_m[:2].copy()
        for _ in range(6):
            if np.count_nonzero(inliers) < config.batch_calibration_min_observations:
                return None
            source = raw[inliers]
            destination = target[inliers]
            source_center = np.mean(source, axis=0)
            destination_center = np.mean(destination, axis=0)
            left, _, right_t = np.linalg.svd(
                (source - source_center).T @ (destination - destination_center)
            )
            row_rotation = left @ right_t
            if np.linalg.det(row_rotation) < 0.0:
                left[:, -1] *= -1.0
                row_rotation = left @ right_t
            column_rotation = row_rotation.T
            yaw = float(np.arctan2(column_rotation[1, 0], column_rotation[0, 0]))
            translation = np.median(destination - source @ row_rotation, axis=0)
            residuals = np.linalg.norm(
                raw @ row_rotation + translation - target, axis=1
            )
            updated = residuals <= threshold
            if np.array_equal(updated, inliers):
                break
            inliers = updated

        full_rotation = self._gnss_rotation(yaw)
        map_translation_xy = np.median(
            target_xyz[:, :2] - (raw_xyz @ full_rotation.T)[:, :2], axis=0
        )
        residuals = np.linalg.norm(
            (raw_xyz @ full_rotation.T)[:, :2] + map_translation_xy
            - target_xyz[:, :2], axis=1
        )
        return {
            "yaw_alignment_rad": self._wrapped_angle(yaw),
            "set_lidar_roll_pitch": bool(config.set_lidar_roll_pitch),
            "lidar_roll_deg": float(config.lidar_roll_deg),
            "lidar_pitch_deg": float(config.lidar_pitch_deg),
            "map_translation_m": np.array([
                map_translation_xy[0], map_translation_xy[1],
                self.gnss_calibration_translation_m[2],
            ]),
            "observation_count": len(usable),
            "calibration_inlier_count": int(np.count_nonzero(inliers)),
            "calibration_outlier_count": int(len(inliers) - np.count_nonzero(inliers)),
            "horizontal_rmse_all_m": float(np.sqrt(np.mean(residuals**2))),
            "horizontal_rmse_inliers_m": float(
                np.sqrt(np.mean(residuals[inliers] ** 2))
            ),
            "usable": usable,
        }

    def _select_batch_factor_observations(self, observations, reference_estimates):
        """Select a deterministic, trajectory-wide subset for GNSS factors.

        Calibration continues to use every usable observation.  A positive
        ``batch_max_position_factors`` caps only the factors passed to the
        optimizer. Targets are uniformly spaced along the completed LiDAR
        trajectory's horizontal arc length, so stationary periods do not
        consume most of a sparse factor budget.
        """
        config = self.config.gnss
        by_frame = {}
        for item in sorted(
            observations,
            key=lambda value: (int(value["frame_index"]), float(value["timestamp"])),
        ):
            frame_id = int(item["frame_index"])
            if frame_id in reference_estimates:
                by_frame.setdefault(frame_id, item)
        available = list(by_frame.values())
        maximum = int(config.batch_max_position_factors)
        if maximum == 0 or len(available) <= maximum:
            return available
        if config.batch_factor_selection_policy != "uniform_trajectory_distance":
            raise ValueError("unsupported batch GNSS factor selection policy")

        xy = np.asarray([
            np.asarray(reference_estimates[int(item["frame_index"])], dtype=float)[:2, 3]
            for item in available
        ])
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
        targets = np.linspace(0.0, float(arc[-1]), maximum)
        ideal_indices = np.linspace(0.0, len(available) - 1, maximum)
        unused = set(range(len(available)))
        selected = []
        for target, ideal in zip(targets, ideal_indices):
            index = min(
                unused,
                key=lambda candidate: (
                    abs(float(arc[candidate]) - float(target)),
                    abs(float(candidate) - float(ideal)),
                    candidate,
                ),
            )
            selected.append(index)
            unused.remove(index)
        return [available[index] for index in sorted(selected)]

    def finalize_batch_gnss_optimization(
        self, fine_optimizer, reference_estimates
    ):
        """Constrain timestamp-associated frames after LiDAR graph completion."""
        if not self.config.gnss.batch_final_calibration_enabled:
            return False
        reference = {
            int(id_): np.asarray(pose, dtype=float).copy()
            for id_, pose in reference_estimates.items()
        }
        fit = self._fit_batch_calibration(
            reference, observation_pose_key="frame_index"
        )
        if fit is None:
            self.batch_gnss_optimization.update({
                "executed": True,
                "accepted": False,
                "reason": "insufficient_batch_calibration_observability",
                "observation_count": len(self.batch_gnss_observations),
            })
            return False

        self.gnss_calibration_yaw_rad = fit["yaw_alignment_rad"]
        self.gnss_calibration_translation_m = fit["map_translation_m"].copy()
        rotation = self._gnss_rotation(self.gnss_calibration_yaw_rad)
        lever = np.asarray(self.config.gnss.lever_arm_body_m, dtype=float)
        factor_observations = self._select_batch_factor_observations(
            fit["usable"], reference
        )
        information_scale = float(self.config.gnss.gnss_information_scale)
        events = []
        for item in factor_observations:
            frame_id = int(item["frame_index"])
            frame_pose = np.asarray(reference[frame_id], dtype=float)
            antenna = (
                rotation @ np.asarray(item["raw_enu_position"], dtype=float)
                + self.gnss_calibration_translation_m
            )
            factor_position = antenna - frame_pose[:3, :3] @ lever
            factor_information = (
                np.asarray(item["information"], dtype=float) * information_scale
            )
            if self.config.gnss.horizontal_only:
                factor_position[2] = frame_pose[2, 3]
                fine_optimizer.add_horizontal_position_factor(
                    frame_id, factor_position, factor_information,
                    self.config.gnss.anchor_robust_kernel_delta,
                )
            else:
                fine_optimizer.add_position_factor(
                    frame_id, factor_position, factor_information,
                    self.config.gnss.anchor_robust_kernel_delta,
                )
            event = {
                "timestamp": item["timestamp"],
                "frame_index": frame_id,
                "node_id": frame_id,
                "local_map_node_id": int(item["node_id"]),
                "raw_enu_position": item["raw_enu_position"].tolist(),
                "factor_position": factor_position.tolist(),
                "factor_information": factor_information.tolist(),
                "gnss_information_scale": information_scale,
                "reliable_fix_count": 1,
                "interval_inlier_count": 1,
                "accepted": True,
                "factor_added": True,
                "reason": "accepted_batch_final",
                "aggregation": "one_reliable_fix_per_timestamped_lidar_frame",
                "factor_target_type": "lidar_frame",
            }
            events.append(event)

        if self._uses_constrained_horizontal_optimization():
            stabilizers = fine_optimizer.add_vertical_attitude_stabilizers(
                reference,
                self.config.gnss.vertical_stabilizer_information,
                self.config.gnss.attitude_stabilizer_information,
            )
            try:
                fine_optimizer.optimize()
                estimates = fine_optimizer.estimates()
            finally:
                fine_optimizer.remove_last_vertical_attitude_stabilizers(
                    stabilizers
                )
        else:
            fine_optimizer.optimize()
            estimates = fine_optimizer.estimates()
        maximum_horizontal_shift = max(
            float(np.linalg.norm(
                np.asarray(estimates[id_])[:2, 3]
                - np.asarray(before)[:2, 3]
            ))
            for id_, before in reference.items()
        )
        finite = all(np.all(np.isfinite(pose)) for pose in estimates.values())
        safe = finite and maximum_horizontal_shift <= (
            self.config.gnss.max_anchor_optimization_shift_m
        )
        if not safe:
            fine_optimizer.remove_last_position_factors(len(events))
            fine_optimizer.restore_estimates(reference)
            self.batch_gnss_optimization.update({
                "executed": True,
                "accepted": False,
                "reason": "optimization_guard",
                "finite": finite,
                "max_horizontal_shift_m": maximum_horizontal_shift,
            })
            return False

        self.gnss_anchor_events.extend(events)
        self.gnss_anchor_ids.extend(event["frame_index"] for event in events)
        self.gnss_anchor_decision_reasons["accepted_batch_final"] += len(events)
        self.gnss_anchor_graph_observable = True
        history = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in fit.items() if key != "usable"
        }
        history.update({
            "reason": "batch_final_global_calibration",
            "optimization_index": len(self.gnss_calibration_history),
        })
        self.gnss_calibration_history.append(history)
        self.batch_gnss_optimization.update({
            "executed": True,
            "accepted": True,
            "reason": "accepted",
            "factor_count": len(events),
            "factor_observation_count_before_selection": len(fit["usable"]),
            "factor_selection_policy": (
                self.config.gnss.batch_factor_selection_policy
                if self.config.gnss.batch_max_position_factors > 0
                else "all_reliable_observations"
            ),
            "factor_limit": int(self.config.gnss.batch_max_position_factors),
            "factor_target_type": "lidar_frame",
            "unique_frame_count": len({event["frame_index"] for event in events}),
            "unique_local_map_node_count": len({
                event["local_map_node_id"] for event in events
            }),
            "max_horizontal_shift_m": maximum_horizontal_shift,
            "gnss_information_scale": information_scale,
            **{key: value for key, value in history.items()
               if key not in {"reason", "optimization_index"}},
        })
        return True

    def _rebuild_gnss_position_factors(self, reference_estimates):
        events = [event for event in self.gnss_anchor_events if event.get("factor_added")]
        self.optimizer.remove_last_position_factors(len(events))
        rotation = self._gnss_rotation(self.gnss_calibration_yaw_rad)
        lever = np.asarray(self.config.gnss.lever_arm_body_m, dtype=float)
        add_factor = (
            self.optimizer.add_horizontal_position_factor
            if self.config.gnss.horizontal_only
            else self.optimizer.add_position_factor
        )
        for event in events:
            raw = np.asarray(event["raw_enu_position"], dtype=float)
            antenna = rotation @ raw + self.gnss_calibration_translation_m
            node_pose = np.asarray(reference_estimates[event["node_id"]], dtype=float)
            body = antenna - node_pose[:3, :3] @ lever
            if self.config.gnss.horizontal_only:
                body[2] = node_pose[2, 3]
            information = np.asarray(event["factor_information"], dtype=float)
            add_factor(
                event["node_id"], body, information,
                self.config.gnss.anchor_robust_kernel_delta,
            )
            event["factor_position"] = body.tolist()
            event["calibration_yaw_rad"] = self.gnss_calibration_yaw_rad
            event["calibration_translation_m"] = (
                self.gnss_calibration_translation_m.tolist()
            )

    def _refine_gnss_calibration(self, reference_estimates):
        if not getattr(self.config.gnss, "recursive_calibration_enabled", False):
            return None
        update = self._fit_recursive_calibration(reference_estimates)
        if update is None:
            return None
        old_yaw = self.gnss_calibration_yaw_rad
        old_translation = self.gnss_calibration_translation_m.copy()
        self.gnss_calibration_yaw_rad = update["yaw_alignment_rad"]
        self.gnss_calibration_translation_m = update["map_translation_m"].copy()
        self._rebuild_gnss_position_factors(reference_estimates)
        self.gnss_calibration_history.append({
            **update,
            "map_translation_m": update["map_translation_m"].tolist(),
            "optimization_index": len(self.gnss_calibration_history),
        })
        old_rotation = self._gnss_rotation(old_yaw)
        new_rotation = self._gnss_rotation(self.gnss_calibration_yaw_rad)
        pending = []
        for decision in self.pending_gnss_anchors:
            if decision.raw_position is None:
                continue
            raw = np.asarray(decision.raw_position, dtype=float)
            delta = (
                new_rotation @ raw + self.gnss_calibration_translation_m
                - old_rotation @ raw - old_translation
            )
            pending.append(replace(
                decision,
                body_position=np.asarray(decision.body_position, dtype=float) + delta,
                correction=np.asarray(decision.correction, dtype=float) + delta,
            ))
        self.pending_gnss_anchors = pending
        self.gnss_gate.recalibrate_candidates(
            old_yaw, old_translation,
            self.gnss_calibration_yaw_rad,
            self.gnss_calibration_translation_m,
        )
        return update

    def optimize_pose_graph(self):
        before = self.optimizer.estimates()
        self._refine_gnss_calibration(before)
        if self._uses_constrained_horizontal_optimization() and self.gnss_anchor_ids:
            estimates = self._optimize_preserving_lidar_vertical_attitude(before)
        else:
            self.optimizer.optimize()
            estimates = self.optimizer.estimates()
        for id_, pose in estimates.items():
            if id_ in self.local_map_graph.graph:
                self.local_map_graph[id_].keypose = np.copy(pose)

    def _uses_constrained_horizontal_optimization(self):
        gnss = self.config.gnss
        return bool(
            getattr(gnss, "enabled", False)
            and getattr(gnss, "horizontal_only", False)
            and getattr(gnss, "horizontal_optimization_strategy", "unconstrained_se3")
            == "constrained_xy_preserve_lidar_z_attitude"
        )

    def _optimize_preserving_lidar_vertical_attitude(self, reference_estimates):
        """Solve XY drift while holding the LiDAR z and rotation state fixed."""
        count = self.optimizer.add_vertical_attitude_stabilizers(
            reference_estimates,
            self.config.gnss.vertical_stabilizer_information,
            self.config.gnss.attitude_stabilizer_information,
        )
        try:
            self.optimizer.optimize()
            return self.optimizer.estimates()
        finally:
            self.optimizer.remove_last_vertical_attitude_stabilizers(count)

    @staticmethod
    def _rotation_difference_deg(before, after):
        relative = np.asarray(before, dtype=float)[:3, :3].T @ np.asarray(
            after, dtype=float
        )[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def generate_new_node(self, timestamp=None):
        points = self.odometry.local_map.point_cloud()
        # Reset odometry
        last_local_map = self.local_map_graph.last_local_map
        relative_motion = last_local_map.local_trajectory[-1]
        inverse_relative_motion = np.linalg.inv(relative_motion)
        transformed_local_map = transform_points(points, inverse_relative_motion)

        self.odometry.local_map.clear()
        self.odometry.local_map.add_points(transformed_local_map)
        self.odometry.last_pose = np.eye(4)

        query_id = last_local_map.id
        query_points = self.voxel_grid.point_cloud()
        self.local_map_graph.finalize_local_map(self.voxel_grid)
        self.voxel_grid.clear()
        self.voxel_grid.add_points(transformed_local_map)
        self.optimizer.add_variable(self.local_map_graph.last_id, self.local_map_graph.last_keypose)
        self.optimizer.add_factor(
            self.local_map_graph.last_id, query_id, relative_motion, np.eye(6)
        )
        # Resolve loop closures before starting the GNSS transaction. Otherwise
        # closure optimization could consume newly added GNSS factors before
        # their displacement guard has had a chance to validate them.
        self.compute_closures(query_id, query_points)
        pre_gnss_estimates = self.optimizer.estimates()
        anchor_events = self._add_pending_gnss_anchor(timestamp)
        if anchor_events:
            self._optimize_gnss_transaction(anchor_events, pre_gnss_estimates)

    def _optimize_gnss_transaction(self, anchor_events, pre_optimization_estimates):
        """Commit GNSS factors only when the complete graph update is bounded."""
        recursive = getattr(
            self.config.gnss, "recursive_calibration_enabled", False
        )
        calibration_before = None
        if recursive:
            calibration_before = (
                self.gnss_calibration_yaw_rad,
                self.gnss_calibration_translation_m.copy(),
                len(self.gnss_calibration_history),
                {
                    id(event): list(event["factor_position"])
                    for event in self.gnss_anchor_events
                    if event.get("factor_added")
                },
            )
            self._refine_gnss_calibration(pre_optimization_estimates)
        constrained = self._uses_constrained_horizontal_optimization()
        if constrained:
            estimates = self._optimize_preserving_lidar_vertical_attitude(
                pre_optimization_estimates
            )
        else:
            self.optimizer.optimize()
            estimates = self.optimizer.estimates()
        maximum_horizontal_shift = 0.0
        maximum_vertical_shift = 0.0
        maximum_attitude_shift_deg = 0.0
        finite = True
        for id_, before in pre_optimization_estimates.items():
            after = np.asarray(estimates[id_], dtype=float)
            before = np.asarray(before, dtype=float)
            finite = finite and bool(np.all(np.isfinite(after)))
            maximum_horizontal_shift = max(
                maximum_horizontal_shift,
                float(np.linalg.norm(after[:2, 3] - before[:2, 3])),
            )
            maximum_vertical_shift = max(
                maximum_vertical_shift,
                float(abs(after[2, 3] - before[2, 3])),
            )
            maximum_attitude_shift_deg = max(
                maximum_attitude_shift_deg,
                self._rotation_difference_deg(before, after),
            )

        for event in anchor_events:
            post_position = np.asarray(estimates[event["node_id"]], dtype=float)[:3, 3]
            pre_position = np.asarray(event["pre_optimization_position"], dtype=float)
            event["attempted_post_optimization_position"] = post_position.tolist()
            event["optimization_shift_m"] = float(np.linalg.norm(post_position - pre_position))
            event["horizontal_optimization_shift_m"] = float(
                np.linalg.norm(post_position[:2] - pre_position[:2])
            )
            event["vertical_optimization_shift_m"] = float(
                abs(post_position[2] - pre_position[2])
            )
            event["graph_max_horizontal_shift_m"] = maximum_horizontal_shift
            event["graph_max_vertical_shift_m"] = maximum_vertical_shift
            event["graph_max_attitude_shift_deg"] = maximum_attitude_shift_deg
            event["horizontal_optimization_strategy"] = (
                self.config.gnss.horizontal_optimization_strategy
                if constrained
                else "unconstrained_se3"
            )

        horizontal_limit = self.config.gnss.max_anchor_optimization_shift_m
        vertical_limit = self.config.gnss.max_anchor_vertical_shift_m
        attitude_limit = getattr(
            self.config.gnss, "max_anchor_attitude_shift_deg", float("inf")
        )
        unsafe = (
            not finite
            or maximum_horizontal_shift > horizontal_limit
            or maximum_vertical_shift > vertical_limit
            or maximum_attitude_shift_deg > attitude_limit
        )
        if unsafe:
            if not recursive:
                self.optimizer.remove_last_position_factors(len(anchor_events))
                self.optimizer.restore_estimates(pre_optimization_estimates)
            else:
                assert calibration_before is not None
                active_events = [
                    event for event in self.gnss_anchor_events
                    if event.get("factor_added")
                ]
                self.optimizer.remove_last_position_factors(len(active_events))
                rejected = {id(event) for event in anchor_events}
                self.gnss_calibration_yaw_rad = calibration_before[0]
                self.gnss_calibration_translation_m = calibration_before[1]
                del self.gnss_calibration_history[calibration_before[2]:]
                add_factor = (
                    self.optimizer.add_horizontal_position_factor
                    if self.config.gnss.horizontal_only
                    else self.optimizer.add_position_factor
                )
                for event in active_events:
                    if id(event) in rejected:
                        continue
                    event["factor_position"] = calibration_before[3][id(event)]
                    add_factor(
                        event["node_id"], event["factor_position"],
                        event["factor_information"],
                        self.config.gnss.anchor_robust_kernel_delta,
                    )
                self.optimizer.restore_estimates(pre_optimization_estimates)
            for event in anchor_events:
                event["accepted"] = False
                event["factor_added"] = False
                event["reason"] = "optimization_guard"
                event["optimization_guard"] = {
                    "finite": finite,
                    "max_horizontal_shift_m": maximum_horizontal_shift,
                    "max_vertical_shift_m": maximum_vertical_shift,
                    "max_attitude_shift_deg": maximum_attitude_shift_deg,
                    "horizontal_limit_m": float(horizontal_limit),
                    "vertical_limit_m": float(vertical_limit),
                    "attitude_limit_deg": float(attitude_limit),
                }
            del self.gnss_anchor_ids[-len(anchor_events):]
            self.gnss_anchor_decision_reasons["accepted"] -= len(anchor_events)
            self.gnss_anchor_decision_reasons["optimization_guard"] += len(anchor_events)
            self.gnss_anchor_graph_observable = False
            return False

        for id_, pose in estimates.items():
            if id_ in self.local_map_graph.graph:
                self.local_map_graph[id_].keypose = np.copy(pose)
        for event in anchor_events:
            event["post_optimization_position"] = event[
                "attempted_post_optimization_position"
            ]
        return True

    def _add_pending_gnss_anchor(self, timestamp):
        if not self.config.gnss.enabled or timestamp is None or not self.pending_gnss_anchors:
            return []
        decisions = list(self.pending_gnss_anchors)
        self.pending_gnss_anchors.clear()
        self.last_gnss_anchor_node_timestamp = float(timestamp)
        node_id = self.local_map_graph.last_id
        node_position = np.copy(self.local_map_graph.last_keypose[:3, 3])
        dimensions = slice(0, 2) if self.config.gnss.horizontal_only else slice(0, 3)
        corrections = np.stack([
            np.asarray(decision.correction, dtype=float) for decision in decisions
        ])
        consensus = np.median(corrections, axis=0)
        residuals = np.linalg.norm(
            corrections[:, dimensions] - consensus[dimensions], axis=1
        )
        inliers = residuals <= self.config.gnss.ransac_residual_threshold_m
        inlier_count = int(np.count_nonzero(inliers))
        minimum_inliers = min(self.config.gnss.ransac_min_samples, len(decisions))
        interval_consistent = inlier_count >= minimum_inliers
        if interval_consistent:
            consensus = np.median(corrections[inliers], axis=0)
        innovation = float(np.linalg.norm(consensus[dimensions]))
        anchor_position = np.copy(node_position)
        anchor_position[dimensions] += consensus[dimensions]
        information_samples = np.stack([
            np.asarray(decision.information, dtype=float)
            for decision, keep in zip(decisions, inliers)
            if keep
        ]) if interval_consistent else np.empty((0, 3))
        # GNSS epochs inside an interval are correlated. Use their median
        # information rather than multiplying confidence by the sample count.
        anchor_information = (
            np.median(information_samples, axis=0)
            if len(information_samples)
            else np.full(3, 1e-9)
        )
        innovation_limit = (
            self.config.gnss.max_anchor_innovation_m
            if self.gnss_anchor_graph_observable
            else self.config.gnss.max_recovery_anchor_innovation_m
        )
        event = {
            "timestamp": float(timestamp),
            "node_id": int(node_id),
            "pre_optimization_position": node_position.tolist(),
            "gnss_body_position": anchor_position.tolist(),
            "innovation_m": innovation,
            "accepted": False,
            "proposal_accepted": bool(
                interval_consistent
                and innovation <= innovation_limit
            ),
            "factor_added": False,
            "interval_start_timestamp": float(decisions[0].timestamp),
            "interval_end_timestamp": float(decisions[-1].timestamp),
            "reliable_fix_count": len(decisions),
            "interval_inlier_count": inlier_count,
            "interval_outlier_count": len(decisions) - inlier_count,
            "aggregation": "median_correction_with_ransac_inliers",
            "innovation_limit_m": float(innovation_limit),
            "recovery_mode": not self.gnss_anchor_graph_observable,
            "raw_enu_position": (
                np.median(np.stack([
                    np.asarray(decision.raw_position, dtype=float)
                    for decision, keep in zip(decisions, inliers)
                    if keep and getattr(decision, "raw_position", None) is not None
                ]), axis=0).tolist()
                if interval_consistent
                and any(keep and getattr(decision, "raw_position", None) is not None
                        for decision, keep in zip(decisions, inliers))
                else None
            ),
        }
        self.gnss_anchor_events.append(event)
        if not event["proposal_accepted"]:
            event["reason"] = "interval_consensus" if not interval_consistent else "innovation"
            self.gnss_anchor_decision_reasons[event["reason"]] += 1
            return []
        if self.config.gnss.horizontal_only:
            # Altitude is intentionally absent from the factor residual. Safety
            # is enforced transactionally over the entire graph after solving.
            anchor_position[2] = node_position[2]
            event["vertical_policy"] = (
                "preserve_lidar_z_and_attitude_during_xy_optimization"
                if self._uses_constrained_horizontal_optimization()
                else "unconstrained_xy_factor_with_graph_guard"
            )
        event["factor_position"] = anchor_position.tolist()
        event["factor_information"] = anchor_information.tolist()
        self.staged_gnss_anchors.append(
            (node_id, anchor_position, anchor_information, event)
        )
        minimum = self.config.gnss.min_anchor_nodes_for_optimization
        if (
            not self.gnss_anchor_graph_observable
            and len(self.staged_gnss_anchors) < minimum
        ):
            event["reason"] = "observability_hold"
            self.gnss_anchor_decision_reasons["observability_hold"] += 1
            return []

        if not self.gnss_anchor_graph_observable:
            staged_corrections = np.stack([
                np.asarray(item[1], dtype=float)
                - np.asarray(item[3]["pre_optimization_position"], dtype=float)
                for item in self.staged_gnss_anchors
            ])
            staged_consensus = np.median(staged_corrections, axis=0)
            staged_residuals = np.linalg.norm(
                staged_corrections[:, dimensions]
                - staged_consensus[dimensions],
                axis=1,
            )
            if np.max(staged_residuals) > self.config.gnss.recovery_anchor_consistency_m:
                _, _, _, dropped_event = self.staged_gnss_anchors.pop(0)
                dropped_event["reason"] = "recovery_consistency"
                self.gnss_anchor_decision_reasons["recovery_consistency"] += 1
                return []

        # A single absolute position prior attached to a long chain whose first
        # pose is fixed can be satisfied by rotating/bending the chain.  Position
        # observations do not independently constrain attitude, and two points
        # can impose an unverified planar heading.  Activate the staged factors
        # together only after three distinct graph nodes are available, then
        # optimize once with a minimally checkable position track.
        activated = list(self.staged_gnss_anchors)
        self.staged_gnss_anchors.clear()
        self.gnss_anchor_graph_observable = True
        for staged_id, staged_position, staged_information, staged_event in activated:
            add_factor = (
                self.optimizer.add_horizontal_position_factor
                if self.config.gnss.horizontal_only
                else self.optimizer.add_position_factor
            )
            add_factor(staged_id, staged_position, staged_information,
                       self.config.gnss.anchor_robust_kernel_delta)
            staged_event["accepted"] = True
            staged_event["factor_added"] = True
            staged_event["reason"] = "accepted"
            self.gnss_anchor_ids.append(staged_id)
            self.gnss_anchor_decision_reasons["accepted"] += 1
        return [item[3] for item in activated]

    @property
    def poses(self):
        poses = [np.eye(4)]
        for node in self.local_map_graph.local_maps():
            for rel_pose in node.local_trajectory[1:]:
                poses.append(node.keypose @ rel_pose)
        return poses

    def fine_grained_optimization(self):
        if self.config.gnss.batch_final_calibration_enabled:
            return self._batch_gnss_fine_grained_optimization()

        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        id_ = 0
        pgo.add_variable(id_, self.local_map_graph[id_].keypose)
        pgo.fix_variable(id_)
        for node in self.local_map_graph.local_maps():
            odometry_factors = [
                np.linalg.inv(T0) @ T1
                for T0, T1 in zip(node.local_trajectory[:-1], node.local_trajectory[1:])
            ]
            for i, factor in enumerate(odometry_factors):
                pgo.add_variable(id_ + 1, node.keypose @ node.local_trajectory[i + 1])
                pgo.add_factor(id_ + 1, id_, factor, np.eye(6))
                id_ += 1
            pgo.fix_variable(id_ - 1)

        pgo.optimize()
        poses = [x for x in pgo.estimates().values()]
        return poses, pgo

    def _batch_gnss_fine_grained_optimization(self):
        """Optimize GNSS on frame vertices, not coarse local-map vertices.

        KISS-SLAM and its local-map graph finish first. Their expanded final
        trajectory becomes the zero-residual LiDAR chain used as the prior for
        this last transaction. This preserves the completed LiDAR solution at
        initialization while allowing each timestamp-associated frame to
        receive its own GNSS factor.
        """
        lidar_poses = [np.asarray(pose, dtype=float).copy() for pose in self.poses]
        return self._optimize_completed_lidar_poses(lidar_poses)

    def _optimize_completed_lidar_poses(self, lidar_poses):
        """Run only the final frame-graph solve on completed LiDAR poses."""
        if not lidar_poses:
            raise RuntimeError("fine-grained GNSS optimization has no LiDAR poses")

        pgo = PoseGraphOptimizer(self.config.pose_graph_optimizer)
        for frame_id, pose in enumerate(lidar_poses):
            pgo.add_variable(frame_id, pose)
            if frame_id:
                relative = np.linalg.inv(lidar_poses[frame_id - 1]) @ pose
                pgo.add_factor(frame_id, frame_id - 1, relative, np.eye(6))
        pgo.fix_variable(0)

        # Establish and snapshot the completed LiDAR-only graph before GNSS.
        pgo.optimize()
        lidar_estimates = pgo.estimates()
        self.batch_pre_gnss_poses = [
            np.asarray(lidar_estimates[index], dtype=float).copy()
            for index in range(len(lidar_poses))
        ]
        self.finalize_batch_gnss_optimization(pgo, lidar_estimates)
        estimates = pgo.estimates()
        poses = [np.asarray(estimates[index], dtype=float) for index in range(len(lidar_poses))]
        return poses, pgo

    @classmethod
    def replay_final_gnss_optimization(
        cls, config, lidar_poses, gnss_fixes, scan_timestamps
    ):
        """Rebuild GNSS observations and optimize an already completed LiDAR graph.

        This deliberately performs no point-cloud registration or local-map
        construction. Reliability gating is repeated with the current GNSS
        configuration against the saved LiDAR trajectory, so covariance,
        RANSAC, robust-kernel, calibration, and factor-selection hyperparameters
        can be tuned without replaying KISS-SLAM.
        """
        lidar_poses = [np.asarray(pose, dtype=float).copy() for pose in lidar_poses]
        if len(lidar_poses) != len(gnss_fixes) or len(lidar_poses) != len(scan_timestamps):
            raise ValueError("cached LiDAR poses and GNSS timestamps have different lengths")
        replay = cls.__new__(cls)
        replay.config = config
        replay._initialize_gnss_state(config)
        for frame_index, (pose, fix, scan_timestamp) in enumerate(
            zip(lidar_poses, gnss_fixes, scan_timestamps)
        ):
            replay.gnss_epochs += 1
            if fix is None:
                replay.gnss_missing_epochs += 1
                replay.gnss_decision_reasons["missing_fix"] += 1
                continue
            if isinstance(fix, dict):
                fix = GnssFix(**fix)
            calibrated = replay._apply_current_gnss_calibration(fix)
            decision = replay.gnss_gate.evaluate(
                calibrated, float(scan_timestamp), pose
            )
            replay.gnss_decision_reasons[decision.reason] += 1
            if not decision.accepted:
                continue
            replay.batch_gnss_observations.append({
                "frame_index": frame_index,
                "timestamp": float(decision.timestamp),
                "node_id": frame_index,
                "relative_pose": np.eye(4),
                "raw_enu_position": np.asarray(
                    decision.raw_position, dtype=float
                ).copy(),
                "information": np.asarray(decision.information, dtype=float).copy(),
            })
        poses, graph = replay._optimize_completed_lidar_poses(lidar_poses)
        return poses, graph, replay

    def gnss_diagnostics(self):
        aggregated = sum(
            int(event.get("reliable_fix_count", 0))
            for event in self.gnss_anchor_events
        )
        interval_inliers = sum(
            int(event.get("interval_inlier_count", 0))
            for event in self.gnss_anchor_events
        )
        return {
            "enabled": bool(self.config.gnss.enabled),
            "profile_id": getattr(self.config.gnss, "profile_id", "unspecified"),
            "horizontal_optimization_strategy": getattr(
                self.config.gnss,
                "horizontal_optimization_strategy",
                "unconstrained_se3",
            ),
            "epochs": int(self.gnss_epochs),
            "missing_epochs": int(self.gnss_missing_epochs),
            "decision_reasons": dict(sorted(self.gnss_decision_reasons.items())),
            "accepted_fixes": int(self.gnss_decision_reasons.get("accepted", 0)),
            "anchors_added": len(self.gnss_anchor_ids),
            "anchors_staged": len(self.staged_gnss_anchors),
            "graph_observable": bool(self.gnss_anchor_graph_observable),
            "reliable_fixes_aggregated": aggregated,
            "interval_inliers": interval_inliers,
            "pending_reliable_fixes": len(self.pending_gnss_anchors),
            "anchor_decision_reasons": dict(sorted(self.gnss_anchor_decision_reasons.items())),
            "anchor_events": self.gnss_anchor_events,
            "recursive_calibration": {
                "enabled": bool(getattr(
                    self.config.gnss, "recursive_calibration_enabled", False
                )),
                "initial_yaw_alignment_rad": float(
                    getattr(self.config.gnss, "initial_yaw_alignment_rad", 0.0)
                ),
                "initial_map_translation_m": list(
                    getattr(self.config.gnss, "initial_map_translation_m", (0., 0., 0.))
                ),
                "final_yaw_alignment_rad": float(getattr(
                    self, "gnss_calibration_yaw_rad", 0.0
                )),
                "final_map_translation_m": np.asarray(getattr(
                    self, "gnss_calibration_translation_m", np.zeros(3)
                )).tolist(),
                "update_count": max(0, len(getattr(
                    self, "gnss_calibration_history", []
                )) - 1),
                "history": getattr(self, "gnss_calibration_history", []),
            },
            "batch_final_calibration": getattr(
                self, "batch_gnss_optimization",
                {"enabled": False, "executed": False},
            ),
        }

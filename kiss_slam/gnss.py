# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Processed-GNSS gating and smooth map-frame publication.

The adapter intentionally accepts position-domain fixes, not raw satellite
observations. Positions and poses must use the same right-handed Cartesian map
frame and timestamps must use the same monotonic seconds timebase.
"""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import numpy as np

from kiss_slam.config.config import GnssConfig


@dataclass(frozen=True)
class GnssFix:
    """A processed antenna position sampled at ``timestamp``.

    ``covariance`` is a 3x3 position covariance in the map frame. Literal HDOP
    is optional when valid covariance is supplied. When covariance is absent,
    HDOP and ``hdop_sigma_scale_m`` provide a conservative diagonal estimate.
    """

    timestamp: float
    position: np.ndarray
    hdop: Optional[float] = None
    covariance: Optional[np.ndarray] = None
    # Uncalibrated local ENU antenna position.  Adapters may provide this in
    # addition to ``position`` so the SLAM back end can recursively refine the
    # ENU-to-graph yaw/translation instead of fitting transformed coordinates.
    raw_position: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    timestamp: float
    body_position: Optional[np.ndarray] = None
    correction: Optional[np.ndarray] = None
    information: Optional[np.ndarray] = None
    inlier_count: int = 0
    raw_position: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _Candidate:
    timestamp: float
    body_position: np.ndarray
    correction: np.ndarray
    covariance: Optional[np.ndarray]
    hdop: Optional[float]
    raw_position: Optional[np.ndarray]


class GnssReliabilityGate:
    """HDOP, synchronization, and RANSAC-consensus gate for processed fixes."""

    def __init__(self, config: GnssConfig):
        self.config = config
        self._candidates = deque(maxlen=config.queue_size)

    def evaluate(
        self, fix: GnssFix, lidar_timestamp: float, body_pose_map: np.ndarray
    ) -> GateDecision:
        timestamp = float(fix.timestamp)
        position = np.asarray(fix.position, dtype=float)
        pose = np.asarray(body_pose_map, dtype=float)
        if not self.config.enabled:
            return GateDecision(False, "disabled", timestamp)
        if (
            not np.isfinite(timestamp)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or pose.shape != (4, 4)
            or not np.all(np.isfinite(pose))
        ):
            return GateDecision(False, "invalid_input", timestamp)
        if fix.hdop is None and fix.covariance is None:
            return GateDecision(False, "missing_quality", timestamp)
        if fix.hdop is not None and (
            not np.isfinite(fix.hdop) or fix.hdop <= 0.0 or fix.hdop > self.config.max_hdop
        ):
            return GateDecision(False, "hdop", timestamp)
        if abs(timestamp - float(lidar_timestamp)) > self.config.max_timestamp_offset_s:
            return GateDecision(False, "timestamp", timestamp)

        covariance = None
        if fix.covariance is not None:
            covariance = np.asarray(fix.covariance, dtype=float)
            if (
                covariance.shape != (3, 3)
                or not np.all(np.isfinite(covariance))
                or np.any(np.diag(covariance) <= 0.0)
            ):
                return GateDecision(False, "covariance", timestamp)
            horizontal_sigma = float(np.sqrt(np.max(np.diag(covariance)[:2])))
            if horizontal_sigma > self.config.max_position_sigma_m:
                return GateDecision(False, "covariance_uncertainty", timestamp)

        lever_arm = np.asarray(self.config.lever_arm_body_m, dtype=float)
        body_position = position - pose[:3, :3] @ lever_arm
        candidate = _Candidate(
            timestamp=timestamp,
            body_position=body_position,
            correction=body_position - pose[:3, 3],
            covariance=covariance,
            hdop=float(fix.hdop) if fix.hdop is not None else None,
            raw_position=(
                np.copy(np.asarray(fix.raw_position, dtype=float))
                if fix.raw_position is not None
                else None
            ),
        )
        self._candidates.append(candidate)
        if len(self._candidates) < self.config.ransac_min_samples:
            return GateDecision(False, "warming_up", timestamp)

        corrections = np.stack([item.correction for item in self._candidates])
        dimensions = slice(0, 2) if self.config.horizontal_only else slice(0, 3)
        best_inliers = np.zeros(len(corrections), dtype=bool)
        best_error = np.inf
        # Candidate-derived hypotheses make this deterministic and cover the
        # small bounded queue without adding a random generator to the hot path.
        hypothesis_indices = np.linspace(
            0,
            len(corrections) - 1,
            min(self.config.ransac_iterations, len(corrections)),
            dtype=int,
        )
        for index in hypothesis_indices:
            residuals = np.linalg.norm(
                corrections[:, dimensions] - corrections[index, dimensions], axis=1
            )
            inliers = residuals <= self.config.ransac_residual_threshold_m
            total_error = float(np.sum(residuals[inliers]))
            if np.sum(inliers) > np.sum(best_inliers) or (
                np.sum(inliers) == np.sum(best_inliers) and total_error < best_error
            ):
                best_inliers = inliers
                best_error = total_error

        inlier_count = int(np.sum(best_inliers))
        if inlier_count < self.config.ransac_min_samples:
            return GateDecision(False, "consensus", timestamp, inlier_count=inlier_count)
        consensus = np.median(corrections[best_inliers], axis=0)
        latest_residual = np.linalg.norm(
            candidate.correction[dimensions] - consensus[dimensions]
        )
        if latest_residual > self.config.ransac_residual_threshold_m:
            return GateDecision(False, "ransac_outlier", timestamp, inlier_count=inlier_count)

        information = self._information(candidate)
        return GateDecision(
            True,
            "accepted",
            timestamp,
            body_position=np.copy(body_position),
            correction=np.copy(candidate.correction),
            information=information,
            inlier_count=inlier_count,
            raw_position=(
                np.copy(np.asarray(fix.raw_position, dtype=float))
                if fix.raw_position is not None
                else None
            ),
        )

    def reset_candidates(self):
        """Discard corrections expressed in an obsolete calibration frame."""
        self._candidates.clear()

    def recalibrate_candidates(
        self, old_yaw, old_translation, new_yaw, new_translation
    ):
        """Move the short consensus queue into a refined ENU-to-map frame."""
        old_rotation = np.array([
            [np.cos(old_yaw), -np.sin(old_yaw), 0.0],
            [np.sin(old_yaw), np.cos(old_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        new_rotation = np.array([
            [np.cos(new_yaw), -np.sin(new_yaw), 0.0],
            [np.sin(new_yaw), np.cos(new_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        old_translation = np.asarray(old_translation, dtype=float)
        new_translation = np.asarray(new_translation, dtype=float)
        recalibrated = deque(maxlen=self._candidates.maxlen)
        for candidate in self._candidates:
            if candidate.raw_position is None:
                continue
            raw = np.asarray(candidate.raw_position, dtype=float)
            delta = (
                new_rotation @ raw + new_translation
                - old_rotation @ raw - old_translation
            )
            recalibrated.append(replace(
                candidate,
                body_position=candidate.body_position + delta,
                correction=candidate.correction + delta,
            ))
        self._candidates = recalibrated

    def _information(self, candidate: _Candidate) -> np.ndarray:
        minimum_variance = self.config.min_position_sigma_m**2
        if candidate.covariance is None:
            assert candidate.hdop is not None
            sigma = max(
                self.config.min_position_sigma_m,
                candidate.hdop * self.config.hdop_sigma_scale_m,
            )
            variances = np.full(3, sigma**2)
        else:
            variances = np.maximum(np.diag(candidate.covariance), minimum_variance)
        information = 1.0 / variances
        if self.config.horizontal_only:
            information[2] = 1e-9
        return information


class RecoveryState(str, Enum):
    OUTAGE = "outage"
    RECOVERING = "recovering"
    TRACKING = "tracking"


class SmoothReentryTracker:
    """Rate-limit translation jumps when accepted GNSS resumes after an outage."""

    def __init__(self, config: GnssConfig):
        self.config = config
        self.state = RecoveryState.OUTAGE
        self._last_fix_timestamp: Optional[float] = None
        self._stable_fixes = 0
        self._last_raw_pose: Optional[np.ndarray] = None
        self._last_published_pose: Optional[np.ndarray] = None
        self._reentry_pending = False

    def observe_fix(self, timestamp: float):
        timestamp = float(timestamp)
        was_outage = (
            self._last_fix_timestamp is None
            or timestamp - self._last_fix_timestamp > self.config.outage_timeout_s
            or self.state == RecoveryState.OUTAGE
        )
        self._stable_fixes = 1 if was_outage else self._stable_fixes + 1
        self.state = RecoveryState.RECOVERING if was_outage else self.state
        self._reentry_pending = self._reentry_pending or was_outage
        self._last_fix_timestamp = timestamp

    def publish(
        self,
        raw_pose_map: np.ndarray,
        timestamp: float,
        dead_reckoned_pose_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raw_pose = np.asarray(raw_pose_map, dtype=float)
        if raw_pose.shape != (4, 4) or not np.all(np.isfinite(raw_pose)):
            raise ValueError("raw_pose_map must be a finite 4x4 transform")
        if dead_reckoned_pose_map is None:
            dead_reckoned_input = (
                self._last_raw_pose
                if self._reentry_pending and self._last_raw_pose is not None
                else raw_pose
            )
        else:
            dead_reckoned_input = np.asarray(dead_reckoned_pose_map, dtype=float)
        if dead_reckoned_input.shape != (4, 4) or not np.all(
            np.isfinite(dead_reckoned_input)
        ):
            raise ValueError("dead_reckoned_pose_map must be a finite 4x4 transform")
        timestamp = float(timestamp)
        if (
            self._last_fix_timestamp is None
            or timestamp - self._last_fix_timestamp > self.config.outage_timeout_s
        ):
            self.state = RecoveryState.OUTAGE
            self._stable_fixes = 0

        if self._last_raw_pose is None or self._last_published_pose is None:
            published = np.copy(raw_pose)
        else:
            # Propagate LiDAR motion first. At a graph-update scan the caller
            # supplies the pose immediately before optimization, so only the
            # global correction is rate-limited; real vehicle motion is not.
            raw_delta = np.linalg.inv(self._last_raw_pose) @ dead_reckoned_input
            dead_reckoned = self._last_published_pose @ raw_delta
            error = raw_pose[:3, 3] - dead_reckoned[:3, 3]
            error_norm = float(np.linalg.norm(error))
            if error_norm > self.config.reentry_max_translation_step_m:
                error *= self.config.reentry_max_translation_step_m / error_norm
            published = np.copy(raw_pose)
            published[:3, 3] = dead_reckoned[:3, 3] + error
            if (
                self.state == RecoveryState.RECOVERING
                and error_norm <= self.config.reentry_max_translation_step_m
                and self._stable_fixes >= self.config.reentry_stable_fixes
            ):
                self.state = RecoveryState.TRACKING
            self._reentry_pending = False

        self._last_raw_pose = np.copy(raw_pose)
        self._last_published_pose = np.copy(published)
        return published

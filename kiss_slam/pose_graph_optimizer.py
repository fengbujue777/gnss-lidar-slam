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
import numpy as np

from kiss_slam.config.config import PoseGraphOptimizerConfig
from kiss_slam.kiss_slam_pybind import kiss_slam_pybind


class PoseGraphOptimizer:
    def __init__(self, config: PoseGraphOptimizerConfig):
        self.pgo = kiss_slam_pybind._PoseGraphOptimizer(config.max_iterations)

    def add_variable(self, id_: int, pose: np.ndarray):
        self.pgo._add_variable(id_, pose)

    def fix_variable(self, id_: int):
        self.pgo._fix_variable(id_)

    def add_factor(self, id_source, id_target, relative_pose, information_matrix):
        self.pgo._add_factor(id_source, id_target, relative_pose, information_matrix)

    def add_position_factor(
        self, id_: int, position, information, robust_kernel_delta: float
    ):
        """Add a true unary XYZ prior without introducing attitude residuals."""
        omega = np.diag(np.asarray(information, dtype=float))
        self.pgo._add_position_prior(
            id_, np.asarray(position, dtype=float), omega, float(robust_kernel_delta)
        )

    def add_horizontal_position_factor(
        self, id_: int, position, information, robust_kernel_delta: float
    ):
        """Add a true unary XY prior; altitude is absent from the residual."""
        omega = np.diag(np.asarray(information, dtype=float)[:2])
        self.pgo._add_horizontal_position_prior(
            id_, np.asarray(position, dtype=float)[:2], omega, float(robust_kernel_delta)
        )

    def remove_last_position_factors(self, count: int):
        self.pgo._remove_last_position_priors(int(count))

    def add_vertical_attitude_stabilizers(
        self,
        estimates,
        vertical_information: float,
        attitude_information: float,
    ) -> int:
        """Temporarily preserve LiDAR z/attitude during an XY graph solve."""
        for id_, pose in estimates.items():
            self.pgo._add_vertical_attitude_prior(
                int(id_),
                np.asarray(pose, dtype=float),
                float(vertical_information),
                float(attitude_information),
            )
        return len(estimates)

    def remove_last_vertical_attitude_stabilizers(self, count: int):
        self.pgo._remove_last_vertical_attitude_priors(int(count))

    def restore_estimates(self, estimates):
        for id_, pose in estimates.items():
            self.pgo._set_estimate(int(id_), np.asarray(pose, dtype=float))

    def optimize(self):
        print("KissSLAM| Optimize Pose Graph")
        self.pgo._optimize()

    def estimates(self):
        return self.pgo._estimates()

    def read_graph(self, filename: str):
        self.pgo._read_graph(filename)

    def write_graph(self, filename: str):
        self.pgo._write_graph(filename)

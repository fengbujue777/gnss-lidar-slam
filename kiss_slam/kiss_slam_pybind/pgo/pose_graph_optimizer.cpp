// MIT License

// Copyright (c) 2025 Tiziano Guadagnino

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "pose_graph_optimizer.hpp"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/stuff/macros.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
static constexpr double epsilon = 1e-6;
}
// clang-format off
namespace g2o {
class EdgeSE3PositionPrior final
    : public BaseUnaryEdge<3, Eigen::Vector3d, VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3PositionPrior() {
        information().setIdentity();
        setMeasurement(Eigen::Vector3d::Zero());
    }

    void computeError() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        _error = vertex->estimate().translation() - _measurement;
    }

    void linearizeOplus() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        _jacobianOplusXi.block<3, 3>(0, 0) = vertex->estimate().rotation();
        _jacobianOplusXi.block<3, 3>(0, 3).setZero();
    }

    bool read(std::istream &is) override {
        for (int i = 0; i < 3; ++i) is >> _measurement[i];
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                is >> information()(i, j);
                information()(j, i) = information()(i, j);
            }
        }
        return is.good();
    }

    bool write(std::ostream &os) const override {
        for (int i = 0; i < 3; ++i) os << measurement()[i] << " ";
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) os << information()(i, j) << " ";
        return os.good();
    }
};

class EdgeSE3HorizontalPositionPrior final
    : public BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3HorizontalPositionPrior() {
        information().setIdentity();
        setMeasurement(Eigen::Vector2d::Zero());
    }

    void computeError() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        _error = vertex->estimate().translation().head<2>() - _measurement;
    }

    void linearizeOplus() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        _jacobianOplusXi.block<2, 3>(0, 0) =
            vertex->estimate().rotation().topRows<2>();
        _jacobianOplusXi.block<2, 3>(0, 3).setZero();
    }

    bool read(std::istream &is) override {
        for (int i = 0; i < 2; ++i) is >> _measurement[i];
        for (int i = 0; i < 2; ++i) {
            for (int j = i; j < 2; ++j) {
                is >> information()(i, j);
                information()(j, i) = information()(i, j);
            }
        }
        return is.good();
    }

    bool write(std::ostream &os) const override {
        for (int i = 0; i < 2; ++i) os << measurement()[i] << " ";
        for (int i = 0; i < 2; ++i)
            for (int j = i; j < 2; ++j) os << information()(i, j) << " ";
        return os.good();
    }
};

// A horizontal GNSS observation must not manufacture altitude or attitude
// information.  At the same time, leaving those four directions free in an
// SE(3) graph lets the solver satisfy an XY prior by tilting/bending the LiDAR
// trajectory.  This temporary prior freezes the LiDAR-derived z and rotation
// during a horizontal correction while leaving x and y completely absent from
// its residual.
class EdgeSE3VerticalAttitudePrior final
    : public BaseUnaryEdge<4, Eigen::Isometry3d, VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3VerticalAttitudePrior() {
        information().setIdentity();
        setMeasurement(Eigen::Isometry3d::Identity());
    }

    void computeError() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        const auto &estimate = vertex->estimate();
        _error[0] = estimate.translation().z() - _measurement.translation().z();

        Eigen::Quaterniond delta(
            _measurement.rotation().transpose() * estimate.rotation());
        if (delta.w() < 0.0) delta.coeffs() *= -1.0;
        const double vector_norm = delta.vec().norm();
        if (vector_norm < 1e-12) {
            _error.tail<3>() = 2.0 * delta.vec();
        } else {
            const double angle = 2.0 * std::atan2(vector_norm, delta.w());
            _error.tail<3>() = angle * delta.vec() / vector_norm;
        }
    }

    void linearizeOplus() override {
        const auto *vertex = static_cast<const VertexSE3 *>(_vertices[0]);
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<1, 3>(0, 0) =
            vertex->estimate().rotation().row(2);
        // The stabilizer remains in the small-error region by construction;
        // the SO(3) right-Jacobian inverse is therefore identity to first order.
        _jacobianOplusXi.block<3, 3>(1, 3).setIdentity();
    }

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }
};

G2O_REGISTER_TYPE(VERTEX_SE3:QUAT, VertexSE3)
G2O_REGISTER_TYPE(EDGE_SE3:QUAT, EdgeSE3)
G2O_REGISTER_TYPE(EDGE_SE3_POSITION_PRIOR, EdgeSE3PositionPrior)
G2O_REGISTER_TYPE(EDGE_SE3_HORIZONTAL_POSITION_PRIOR, EdgeSE3HorizontalPositionPrior)
G2O_REGISTER_TYPE(EDGE_SE3_VERTICAL_ATTITUDE_PRIOR, EdgeSE3VerticalAttitudePrior)
}  // namespace g2o
// clang-format on

namespace pgo {
using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
using AlgorithmType = g2o::OptimizationAlgorithmDogleg;

PoseGraphOptimizer::PoseGraphOptimizer(const int max_iterations) : max_iterations_(max_iterations) {
    graph = std::make_unique<g2o::SparseOptimizer>();
    graph->setVerbose(true);

    auto solver =
        new AlgorithmType(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    auto terminateAction = new g2o::SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(epsilon);
    graph->addPostIterationAction(terminateAction);
    graph->setAlgorithm(solver);

}

void PoseGraphOptimizer::addPositionPrior(const int id,
                                          const Eigen::Vector3d &position,
                                          const Eigen::Matrix3d &information_matrix,
                                          const double robust_kernel_delta) {
    auto *factor = new g2o::EdgeSE3PositionPrior();
    factor->setVertex(0, graph->vertex(id));
    factor->setMeasurement(position);
    factor->setInformation(information_matrix);
    if (robust_kernel_delta > 0.0) {
        auto *kernel = new g2o::RobustKernelHuber();
        kernel->setDelta(robust_kernel_delta);
        factor->setRobustKernel(kernel);
    }
    graph->addEdge(factor);
    position_priors_.push_back(factor);
}

void PoseGraphOptimizer::addHorizontalPositionPrior(
    const int id,
    const Eigen::Vector2d &position,
    const Eigen::Matrix2d &information_matrix,
    const double robust_kernel_delta) {
    auto *factor = new g2o::EdgeSE3HorizontalPositionPrior();
    factor->setVertex(0, graph->vertex(id));
    factor->setMeasurement(position);
    factor->setInformation(information_matrix);
    if (robust_kernel_delta > 0.0) {
        auto *kernel = new g2o::RobustKernelHuber();
        kernel->setDelta(robust_kernel_delta);
        factor->setRobustKernel(kernel);
    }
    graph->addEdge(factor);
    position_priors_.push_back(factor);
}

void PoseGraphOptimizer::addVerticalAttitudePrior(
    const int id,
    const Eigen::Matrix4d &pose,
    const double vertical_information,
    const double attitude_information) {
    if (vertical_information <= 0.0 || attitude_information <= 0.0) {
        throw std::invalid_argument("stabilizer information must be positive");
    }
    auto *factor = new g2o::EdgeSE3VerticalAttitudePrior();
    factor->setVertex(0, graph->vertex(id));
    factor->setMeasurement(Eigen::Isometry3d(pose));
    Eigen::Matrix4d information = Eigen::Matrix4d::Zero();
    information(0, 0) = vertical_information;
    information.block<3, 3>(1, 1) =
        attitude_information * Eigen::Matrix3d::Identity();
    factor->setInformation(information);
    graph->addEdge(factor);
    vertical_attitude_priors_.push_back(factor);
}

void PoseGraphOptimizer::removeLastPositionPriors(const int count) {
    if (count < 0 || static_cast<std::size_t>(count) > position_priors_.size()) {
        throw std::invalid_argument("invalid position-prior rollback count");
    }
    for (int index = 0; index < count; ++index) {
        auto *factor = position_priors_.back();
        position_priors_.pop_back();
        // HyperGraph::removeEdge owns and releases the edge.
        if (!graph->removeEdge(factor)) {
            throw std::runtime_error("failed to remove position prior");
        }
    }
}

void PoseGraphOptimizer::removeLastVerticalAttitudePriors(const int count) {
    if (count < 0 || static_cast<std::size_t>(count) > vertical_attitude_priors_.size()) {
        throw std::invalid_argument("invalid vertical-attitude-prior rollback count");
    }
    for (int index = 0; index < count; ++index) {
        auto *factor = vertical_attitude_priors_.back();
        vertical_attitude_priors_.pop_back();
        if (!graph->removeEdge(factor)) {
            throw std::runtime_error("failed to remove vertical-attitude prior");
        }
    }
}

void PoseGraphOptimizer::setEstimate(const int id, const Eigen::Matrix4d &T) {
    auto *variable = dynamic_cast<g2o::VertexSE3 *>(graph->vertex(id));
    if (variable == nullptr) throw std::invalid_argument("unknown SE3 variable id");
    variable->setEstimate(Eigen::Isometry3d(T));
}

void PoseGraphOptimizer::fixVariable(const int id) { graph->vertex(id)->setFixed(true); }

void PoseGraphOptimizer::addVariable(const int id, const Eigen::Matrix4d &T) {
    Eigen::Isometry3d pose(T);
    g2o::VertexSE3 *variable = new g2o::VertexSE3();
    variable->setId(id);
    variable->setEstimate(pose);
    graph->addVertex(variable);
}

void PoseGraphOptimizer::addFactor(const int id_source,
                                   const int id_target,
                                   const Eigen::Matrix4d &T,
                                   const Eigen::Matrix6d &information_matrix) {
    Eigen::Isometry3d relative_pose(T);
    g2o::EdgeSE3 *factor = new g2o::EdgeSE3();
    factor->setVertex(0, graph->vertex(id_target));
    factor->setVertex(1, graph->vertex(id_source));
    factor->setInformation(information_matrix);
    factor->setMeasurement(relative_pose);
    graph->addEdge(factor);
}

PoseGraphOptimizer::PoseIDMap PoseGraphOptimizer::estimates() const {
    const g2o::HyperGraph::VertexIDMap &variables = graph->vertices();
    PoseIDMap poses;
    std::transform(variables.cbegin(), variables.cend(), std::inserter(poses, poses.end()),
                   [](const auto &id_var) {
                       const auto &[id, v] = id_var;
                       Eigen::Isometry3d pose = static_cast<g2o::VertexSE3 *>(v)->estimate();
                       return std::make_pair(id, pose.matrix());
                   });
    return poses;
}

void PoseGraphOptimizer::optimize() {
    graph->initializeOptimization();
    graph->optimize(max_iterations_);
}
}  // namespace pgo

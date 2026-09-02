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
#pragma once
#include <g2o/core/sparse_optimizer.h>

#include <Eigen/Geometry>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace Eigen {
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Matrix2d = Eigen::Matrix<double, 2, 2>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector2d = Eigen::Matrix<double, 2, 1>;
}  // namespace Eigen
namespace pgo {
class PoseGraphOptimizer {
public:
    using PoseIDMap = std::map<int, Eigen::Matrix4d>;
    explicit PoseGraphOptimizer(const int max_iterations);

    void fixVariable(const int id);
    void addVariable(const int id, const Eigen::Matrix4d &T);

    void addFactor(const int id_source,
                   const int id_target,
                   const Eigen::Matrix4d &T,
                   const Eigen::Matrix6d &information_matrix);

    void addPositionPrior(const int id,
                          const Eigen::Vector3d &position,
                          const Eigen::Matrix3d &information_matrix,
                          const double robust_kernel_delta);

    void addHorizontalPositionPrior(const int id,
                                    const Eigen::Vector2d &position,
                                    const Eigen::Matrix2d &information_matrix,
                                    const double robust_kernel_delta);

    void addVerticalAttitudePrior(const int id,
                                  const Eigen::Matrix4d &pose,
                                  const double vertical_information,
                                  const double attitude_information);

    void removeLastPositionPriors(const int count);
    void removeLastVerticalAttitudePriors(const int count);
    void setEstimate(const int id, const Eigen::Matrix4d &T);

    [[nodiscard]] PoseIDMap estimates() const;

    inline void readGraph(const std::string &filename) {
        std::ifstream file(filename.c_str());
        graph->clear();
        graph->load(file);
    }
    inline void writeGraph(const std::string &filename) const { graph->save(filename.c_str()); }

    void optimize();

private:
    std::unique_ptr<g2o::SparseOptimizer> graph;
    std::vector<g2o::OptimizableGraph::Edge *> position_priors_;
    std::vector<g2o::OptimizableGraph::Edge *> vertical_attitude_priors_;
    int max_iterations_;
};
}  // namespace pgo

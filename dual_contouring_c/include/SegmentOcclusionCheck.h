#ifndef __SEGMENT_OCCLUSION_CHECK_H__
#define __SEGMENT_OCCLUSION_CHECK_H__

#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include <Eigen/Dense>
#include "NumericUtils.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, 1> VecI64;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;

struct OcclusionHitResult
{
    bool is_occluded = false;
    int64_t hit_node_id = -1;
};

class SegmentOcclusionChecker
{
public:
    SegmentOcclusionChecker(
        const MatF3& centers,
        const MatF3& nearestPoints,
        const VecI64& selfNodeIds,
        const VecI64& occupiedCellOccluder,
        const Eigen::Vector3f& minCorner,
        float cellSize,
        int resolution)
        : centers_(centers),
          nearestPoints_(nearestPoints),
          selfNodeIds_(selfNodeIds),
          occupiedCellOccluder_(occupiedCellOccluder),
          minCorner_(minCorner),
          cellSize_(cellSize),
          resolution_(resolution)
    {
        if (centers_.rows() != nearestPoints_.rows() ||
            centers_.rows() != selfNodeIds_.rows()) {
            throw std::invalid_argument(
                "SegmentOcclusionChecker: centers, nearestPoints, and selfNodeIds must have the same number of rows.");
        }

        if (cellSize_ <= 0.0f) {
            throw std::invalid_argument("SegmentOcclusionChecker: cellSize must be positive.");
        }

        if (resolution_ <= 0) {
            throw std::invalid_argument("SegmentOcclusionChecker: resolution must be positive.");
        }

        const int64_t expectedSize =
            static_cast<int64_t>(resolution_) * resolution_ * resolution_;

        if (occupiedCellOccluder_.size() != expectedSize) {
            throw std::invalid_argument(
                "SegmentOcclusionChecker: occupiedCellOccluder size must equal resolution^3.");
        }
    }

public:
    inline int64_t encodeGrid(int x, int y, int z) const
    {
        return (static_cast<int64_t>(x) * resolution_ + y) * resolution_ + z;
    }

    inline bool isInsideGrid(int x, int y, int z) const
    {
        return (x >= 0 && x < resolution_ &&
                y >= 0 && y < resolution_ &&
                z >= 0 && z < resolution_);
    }

    inline int clampGridCoord(int v) const
    {
        return std::max(0, std::min(resolution_ - 1, v));
    }

    inline Eigen::Vector3i worldToGrid(const Eigen::Vector3f& p) const
    {
        Eigen::Array3f g = ((p - minCorner_).array() / cellSize_).floor();
        Eigen::Vector3i gi;
        gi[0] = clampGridCoord(static_cast<int>(g[0]));
        gi[1] = clampGridCoord(static_cast<int>(g[1]));
        gi[2] = clampGridCoord(static_cast<int>(g[2]));
        return gi;
    }

    inline int64_t getOccupiedNodeId(int x, int y, int z) const
    {
        return occupiedCellOccluder_[encodeGrid(x, y, z)];
    }

public:
    OcclusionHitResult checkSingleSegment(int64_t segIdx) const
    {
        if (segIdx < 0 || segIdx >= centers_.rows()) {
            throw std::out_of_range("checkSingleSegment: segIdx out of range.");
        }

        const Eigen::Vector3f p0 = centers_.row(segIdx);
        const Eigen::Vector3f p1 = nearestPoints_.row(segIdx);
        const int64_t selfNodeId = selfNodeIds_[segIdx];

        const Eigen::Vector3f direction = p1 - p0;
        const float length = direction.norm();

        OcclusionHitResult result;
        if (compareDist(length, 0.0f) == 0) {
            return result;
        }

        const Eigen::Vector3f rayDir = direction / length;

        constexpr float shrinkEps = 1e-8f;
        const Eigen::Vector3f p1Shrink = p1 - rayDir * shrinkEps;

        const Eigen::Vector3i startCell = worldToGrid(p0);
        const Eigen::Vector3i endCell   = worldToGrid(p1Shrink);

        int x = startCell[0];
        int y = startCell[1];
        int z = startCell[2];

        const int endX = endCell[0];
        const int endY = endCell[1];
        const int endZ = endCell[2];

        const int stepX =
            (compareDist(std::abs(rayDir[0]), 0.0f) == 0) ? 0 : (rayDir[0] > 0.0f ? 1 : -1);
        const int stepY =
            (compareDist(std::abs(rayDir[1]), 0.0f) == 0) ? 0 : (rayDir[1] > 0.0f ? 1 : -1);
        const int stepZ =
            (compareDist(std::abs(rayDir[2]), 0.0f) == 0) ? 0 : (rayDir[2] > 0.0f ? 1 : -1);

        const float inf = std::numeric_limits<float>::infinity();

        const float tDeltaX = (stepX == 0) ? inf : std::abs(cellSize_ / rayDir[0]);
        const float tDeltaY = (stepY == 0) ? inf : std::abs(cellSize_ / rayDir[1]);
        const float tDeltaZ = (stepZ == 0) ? inf : std::abs(cellSize_ / rayDir[2]);

        const Eigen::Vector3f cellMin =
            minCorner_ + Eigen::Vector3f(
                static_cast<float>(x) * cellSize_,
                static_cast<float>(y) * cellSize_,
                static_cast<float>(z) * cellSize_);

        float tMaxX = inf;
        float tMaxY = inf;
        float tMaxZ = inf;

        if (stepX > 0) {
            const float nextXPlane = cellMin[0] + cellSize_;
            tMaxX = (nextXPlane - p0[0]) / rayDir[0];
        } else if (stepX < 0) {
            const float nextXPlane = cellMin[0];
            tMaxX = (nextXPlane - p0[0]) / rayDir[0];
        }

        if (stepY > 0) {
            const float nextYPlane = cellMin[1] + cellSize_;
            tMaxY = (nextYPlane - p0[1]) / rayDir[1];
        } else if (stepY < 0) {
            const float nextYPlane = cellMin[1];
            tMaxY = (nextYPlane - p0[1]) / rayDir[1];
        }

        if (stepZ > 0) {
            const float nextZPlane = cellMin[2] + cellSize_;
            tMaxZ = (nextZPlane - p0[2]) / rayDir[2];
        } else if (stepZ < 0) {
            const float nextZPlane = cellMin[2];
            tMaxZ = (nextZPlane - p0[2]) / rayDir[2];
        }

        const float tEnd = length;
        const int maxIter = 3 * resolution_ + 10;

        for (int iter = 0; iter < maxIter; ++iter) {
            const float tNext = std::min(tMaxX, std::min(tMaxY, tMaxZ));

            if (compareDist(tNext, tEnd) > 0) {
                break;
            }

            if (compareDist(tMaxX, tNext) <= 0) {
                x += stepX;
                tMaxX += tDeltaX;
            }
            if (compareDist(tMaxY, tNext) <= 0) {
                y += stepY;
                tMaxY += tDeltaY;
            }
            if (compareDist(tMaxZ, tNext) <= 0) {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }

            if (!isInsideGrid(x, y, z)) {
                break;
            }

            const int64_t hitNodeId = getOccupiedNodeId(x, y, z);
            if (hitNodeId >= 0 && hitNodeId != selfNodeId) {
                result.is_occluded = true;
                result.hit_node_id = hitNodeId;
                return result;
            }

            if (x == endX && y == endY && z == endZ) {
                break;
            }
        }

        return result;
    }

    void checkAllSegments(
        ArrayB1& occludedFlags,
        VecI64* hitNodeIds = nullptr) const
    {
        const int64_t n = centers_.rows();

        occludedFlags.resize(n);
        occludedFlags.setConstant(false);

        if (hitNodeIds) {
            hitNodeIds->resize(n);
            hitNodeIds->setConstant(-1);
        }

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const OcclusionHitResult r = checkSingleSegment(i);
            occludedFlags[i] = r.is_occluded;

            if (hitNodeIds) {
                (*hitNodeIds)[i] = r.hit_node_id;
            }
        }
    }

private:
    MatF3 centers_;
    MatF3 nearestPoints_;
    VecI64 selfNodeIds_;

    // flatten 3D array: size = resolution^3, value = node.id, empty = -1
    VecI64 occupiedCellOccluder_;

    Eigen::Vector3f minCorner_;
    float cellSize_ = 0.0f;
    int resolution_ = 0;
};
ArrayB1 batch_segment_occlusion_check(
    const MatF3& centers,
    const MatF3& nearestPoints,
    const VecI64& selfNodeIds,
    const VecI64& occupiedCellOccluder,
    const Eigen::Vector3f& minCorner,
    float cellSize,
    int resolution);

std::pair<ArrayB1, VecI64> batch_segment_occlusion_check_with_hit_ids(
    const MatF3& centers,
    const MatF3& nearestPoints,
    const VecI64& selfNodeIds,
    const VecI64& occupiedCellOccluder,
    const Eigen::Vector3f& minCorner,
    float cellSize,
    int resolution);
void test_flattened_occupied_cell_order(const VecI64& flatData, int resolution);
#endif // __SEGMENT_OCCLUSION_CHECK_H__
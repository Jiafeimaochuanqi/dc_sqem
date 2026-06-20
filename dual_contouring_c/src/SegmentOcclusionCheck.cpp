#include "SegmentOcclusionCheck.h"

#include <iostream>
#include <stdexcept>
#include <utility>

ArrayB1 batch_segment_occlusion_check(
    const MatF3& centers,
    const MatF3& nearestPoints,
    const VecI64& selfNodeIds,
    const VecI64& occupiedCellOccluder,
    const Eigen::Vector3f& minCorner,
    float cellSize,
    int resolution)
{
    SegmentOcclusionChecker checker(
        centers,
        nearestPoints,
        selfNodeIds,
        occupiedCellOccluder,
        minCorner,
        cellSize,
        resolution
    );

    ArrayB1 occludedFlags;
    checker.checkAllSegments(occludedFlags, nullptr);
    return occludedFlags;
}

std::pair<ArrayB1, VecI64> batch_segment_occlusion_check_with_hit_ids(
    const MatF3& centers,
    const MatF3& nearestPoints,
    const VecI64& selfNodeIds,
    const VecI64& occupiedCellOccluder,
    const Eigen::Vector3f& minCorner,
    float cellSize,
    int resolution)
{
    SegmentOcclusionChecker checker(
        centers,
        nearestPoints,
        selfNodeIds,
        occupiedCellOccluder,
        minCorner,
        cellSize,
        resolution
    );

    ArrayB1 occludedFlags;
    VecI64 hitNodeIds;
    checker.checkAllSegments(occludedFlags, &hitNodeIds);

    return std::make_pair(occludedFlags, hitNodeIds);
}

void test_flattened_occupied_cell_order(const VecI64& flatData, int resolution)
{
    if (resolution <= 0) {
        throw std::invalid_argument("test_flattened_occupied_cell_order: resolution must be positive.");
    }

    const int64_t expectedSize =
        static_cast<int64_t>(resolution) * resolution * resolution;

    if (flatData.size() != expectedSize) {
        throw std::invalid_argument(
            "test_flattened_occupied_cell_order: flatData.size() must equal resolution^3.");
    }

    std::cout << "==============================" << std::endl;
    std::cout << "C++ received flattened data" << std::endl;
    std::cout << "resolution = " << resolution << std::endl;
    std::cout << "flat size   = " << flatData.size() << std::endl;
    std::cout << "------------------------------" << std::endl;

    std::cout << "[1D order]" << std::endl;
    for (int64_t i = 0; i < flatData.size(); ++i) {
        std::cout << "flat[" << i << "] = " << flatData[i] << std::endl;
    }

    std::cout << "------------------------------" << std::endl;

    std::cout << "[(x,y,z) -> encoded idx -> value]" << std::endl;
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                const int64_t idx =
                    (static_cast<int64_t>(x) * resolution + y) * resolution + z;

                std::cout
                    << "(" << x << "," << y << "," << z << ")"
                    << " -> idx " << idx
                    << " -> value " << flatData[idx]
                    << std::endl;
            }
        }
    }

    std::cout << "==============================" << std::endl;
}
#include "cloud_rvd.h"
#include "NanoKNNSearch.h"

std::vector<std::vector<std::vector<int>>> cloud_grouping(
    MatF3& cloud,
    MatF3& cloudNormals,
    const ArrayB1& cloudFeatureFlags,
    const Eigen::VectorXi& cellSeedNums,
    const Eigen::VectorXi& nearestCloudIdsFlat,
    const VecI64& cellNodeIds)
{
    PointCloudRVD<float> rvd(cloud, cloudNormals, cloudFeatureFlags);
    rvd.label(cellSeedNums, nearestCloudIdsFlat, cellNodeIds);
    return rvd.computeProjectionClusters(cellSeedNums, nearestCloudIdsFlat, cellNodeIds);
}
#ifndef __CLOUD_RVD_H__
#define __CLOUD_RVD_H__
#define _USE_MATH_DEFINES
#include <cmath>
#include <queue>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <Eigen/Dense>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <set>
#include "NanoKNNSearch.h"
#include "NumericUtils.h"
#include "UnionFind.h"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<float, 1, 3, Eigen::RowMajor> RowVector3f;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayB1;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, 1> VecI64;
template <typename T>
class PointCloudRVD {
public:

    inline static const T cosFeatureAndFeature = std::cos(55.0 * static_cast<T>(M_PI / 180.0));
    using Edge = std::pair<int, T>;   // {neighbor, weight}
    using Node = std::pair<T, int>;   // {distance, vertex}
    using MatT3 = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using RowVector3T = Eigen::Matrix<T, 1, 3, Eigen::RowMajor>;

    struct LabelState {
        T dist;
        int v;
        int64_t label;

        bool operator>(const LabelState& other) const {
            return dist > other.dist;
        }
    };

    struct Result {
        std::vector<int64_t> labels;
        T dist;

        Result() : dist(std::numeric_limits<T>::infinity()) {}
        Result(std::vector<int64_t> l, T d) : labels(std::move(l)), dist(d) {}
    };

public:

    PointCloudRVD(const MatT3& points, const MatT3& normals, const ArrayB1& featureFlags)
        : points_(points), normals_(normals), featureFlags_(featureFlags) {
        rows_ = points_.rows();
        cols_ = 3;
        results.assign(points.rows(), Result());
    }

    PointCloudRVD(MatT3&& points, MatT3&& normals, ArrayB1&& featureFlags)
        : points_(std::move(points)), normals_(std::move(normals)), featureFlags_(std::move(featureFlags)) {
        rows_ = points_.rows();
        cols_ = 3;
        results.assign(points_.rows(), Result());
    }

    bool normalPass(
        int64_t i,
        int64_t j,
        const MatT3& normals,
        T cosThreshold,
        bool unorientedNormals) const
    {
        RowVector3T ni = normals.row(i);
        RowVector3T nj = normals.row(j);
        T dot = ni.dot(nj);
        T denom = ni.norm() * nj.norm();
        T cosv = dot / denom;

        if (cosv > static_cast<T>(1)) cosv = static_cast<T>(1);
        if (cosv < static_cast<T>(-1)) cosv = static_cast<T>(-1);

        if (unorientedNormals) {
            cosv = std::abs(cosv);
        }

        return cosv >= cosThreshold;
    }

    void label(const Eigen::VectorXi& cellSeedNums,
               const Eigen::VectorXi& nearestCloudIdsFlat,
               const VecI64& cellNodeIds)
    {
        const T INF = std::numeric_limits<T>::infinity();

        if (cellSeedNums.size() <= 0) {
            throw std::invalid_argument("label: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "label: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        if (cellNodeIds.size() != cellSeedNums.size()) {
            throw std::invalid_argument(
                "label: cellNodeIds.size() must equal cellSeedNums.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "label: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        results.assign(static_cast<size_t>(rows_), Result());
        for (auto& r : results) {
            r.dist = INF;
            r.labels.clear();
        }

        if (!graphBuilt_) {
            NanoKNNSearch<T> knn(points_.data(), points_.rows(), 3, 8);
            buildKnnGraph(knn, 26, true, static_cast<T>(60.0),static_cast<T>(90.0), 1);
        }

        std::priority_queue<LabelState, std::vector<LabelState>, std::greater<LabelState>> pq;

        int offset = 0;
        for (int cellIdx = 0; cellIdx < cellSeedNums.size(); ++cellIdx) {
            const int cnt = cellSeedNums[cellIdx];
            const int64_t nodeId = cellNodeIds[cellIdx];

            if (cnt < 0) {
                throw std::invalid_argument("label: cellSeedNums contains negative value.");
            }

            for (int k = 0; k < cnt; ++k) {
                const int seed = nearestCloudIdsFlat[offset + k];
                int cmp = compareDist<T>(results[static_cast<size_t>(seed)].dist, static_cast<T>(0));

                if (cmp > 0) {
                    results[static_cast<size_t>(seed)].dist = static_cast<T>(0);
                    results[static_cast<size_t>(seed)].labels.clear();
                    results[static_cast<size_t>(seed)].labels.push_back(nodeId);
                    pq.push(LabelState{static_cast<T>(0), seed, nodeId});
                }
                else if (cmp == 0) {
                    if (appendLabelUnique(results[static_cast<size_t>(seed)].labels, nodeId)) {
                        pq.push(LabelState{static_cast<T>(0), seed, nodeId});
                    }
                }
            }
            offset += cnt;
        }

        while (!pq.empty()) {
            auto [curDist, u, label] = pq.top();
            pq.pop();

            if (compareDist(curDist, results[static_cast<size_t>(u)].dist) > 0) {
                continue;
            }

            if (!hasLabel(u, label)) {
                continue;
            }

            for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                const T nd = curDist + w;
                int cmp = compareDist(nd, results[static_cast<size_t>(v)].dist);

                if (cmp < 0) {
                    results[static_cast<size_t>(v)].dist = nd;
                    results[static_cast<size_t>(v)].labels.clear();
                    results[static_cast<size_t>(v)].labels.push_back(label);
                    pq.push(LabelState{nd, v, label});
                }
                else if (cmp == 0) {
                    if (appendLabelUnique(results[static_cast<size_t>(v)].labels, label)) {
                        pq.push(LabelState{results[static_cast<size_t>(v)].dist, v, label});
                    }
                }
            }
        }

        for (auto& r : results) {
            std::sort(r.labels.begin(), r.labels.end());
        }
    }

    void label(const std::vector<std::vector<int>>& cellNearestCloudIds,
               const VecI64& cellNodeIds)
    {
        const int numCells = static_cast<int>(cellNearestCloudIds.size());
        if (numCells <= 0) {
            throw std::invalid_argument("label(nested): cellNearestCloudIds is empty.");
        }

        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument("label(nested): cellNodeIds size mismatch.");
        }

        Eigen::VectorXi cellSeedNums(numCells);

        int total = 0;
        for (int c = 0; c < numCells; ++c) {
            const int cnt = static_cast<int>(cellNearestCloudIds[c].size());
            if (cnt < 0) {
                throw std::invalid_argument("label(nested): invalid negative seed count.");
            }
            cellSeedNums[c] = cnt;
            total += cnt;
        }

        Eigen::VectorXi nearestCloudIdsFlat(total);
        int offset = 0;
        for (int c = 0; c < numCells; ++c) {
            for (int seed : cellNearestCloudIds[c]) {
                nearestCloudIdsFlat[offset++] = seed;
            }
        }

        label(cellSeedNums, nearestCloudIdsFlat, cellNodeIds);
    }

    bool hasLabelSorted(int u, int64_t nodeId) const
    {
        const auto& labels = results[static_cast<size_t>(u)].labels;
        return std::binary_search(labels.begin(), labels.end(), nodeId);
    }

    bool hasLabel(int u, int64_t nodeId) const
    {
        const auto& labels = results[static_cast<size_t>(u)].labels;
        return std::find(labels.begin(), labels.end(), nodeId) != labels.end();
    }

    std::vector<std::vector<int>>
    computeCellRelatedPointIds(const Eigen::VectorXi& cellSeedNums,
                               const VecI64& cellNodeIds)
    {
        const int numCells = static_cast<int>(cellSeedNums.size());
        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument("computeCellRelatedPointIds: cellNodeIds size mismatch.");
        }

        std::unordered_map<int64_t, int> nodeIdToCellIdx;
        nodeIdToCellIdx.reserve(static_cast<size_t>(numCells) * 2);
        for (int i = 0; i < numCells; ++i) {
            nodeIdToCellIdx[cellNodeIds[i]] = i;
        }

        std::vector<std::vector<int>> result(numCells, std::vector<int>());

        for (int64_t i = 0; i < rows_; ++i) {
            const auto& labels = results[static_cast<size_t>(i)].labels;
            for (const auto& nodeId : labels) {
                auto it = nodeIdToCellIdx.find(nodeId);
                if (it != nodeIdToCellIdx.end()) {
                    result[it->second].push_back(static_cast<int>(i));
                }
            }
        }
        return result;
    }

    std::vector<std::vector<std::vector<int>>>
    computeCellRelatedPointClusters(const Eigen::VectorXi& cellSeedNums,
                                    const Eigen::VectorXi& nearestCloudIdsFlat,
                                    const VecI64& cellNodeIds) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeCellRelatedPointClusters: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeCellRelatedPointClusters: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument(
                "computeCellRelatedPointClusters: cellNodeIds.size() must equal cellSeedNums.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeCellRelatedPointClusters: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int64_t nodeId = cellNodeIds[cellIdx];
            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);
            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> visited(static_cast<size_t>(rows_), 0);

            for (int u0 = 0; u0 < rows_; ++u0) {
                if (visited[static_cast<size_t>(u0)]) {
                    continue;
                }

                const bool belongs0 =
                    isSeedOfCell[static_cast<size_t>(u0)] || hasLabelSorted(u0, nodeId);

                if (!belongs0) {
                    continue;
                }

                std::vector<int> cluster;
                std::queue<int> q;

                visited[static_cast<size_t>(u0)] = 1;
                q.push(u0);

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();

                    cluster.push_back(u);

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (visited[static_cast<size_t>(v)]) {
                            continue;
                        }

                        const bool belongs =
                            isSeedOfCell[static_cast<size_t>(v)] || hasLabelSorted(v, nodeId);

                        if (!belongs) {
                            continue;
                        }

                        visited[static_cast<size_t>(v)] = 1;
                        q.push(v);
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }

    
    
    std::vector<std::pair<int,int>>
    extractUniqueUndirectedEdges()
    {
        std::set<std::pair<int,int>> edge_set;

        for (int u = 0; u < static_cast<int>(adj_.size()); ++u) {
            for (const auto& e : adj_[u]) {
                int v = e.first;
                if (u == v) continue;

                int a = std::min(u, v);
                int b = std::max(u, v);
                edge_set.insert({a, b});
            }
        }

        return std::vector<std::pair<int,int>>(edge_set.begin(), edge_set.end());
    }

    void buildGraphFromEdges(
        const std::vector<std::tuple<int,int,T>>& edges,
        bool symmetric = true,
        bool deduplicate = true)
    {
        if (rows_ == 0) {
            throw std::runtime_error("buildGraphFromEdges: point cloud empty.");
        }

        adj_.assign(rows_, {});

        for (const auto& e : edges) {
            int u = std::get<0>(e);
            int v = std::get<1>(e);
            T w   = std::get<2>(e);

            if (u < 0 || u >= rows_ || v < 0 || v >= rows_) {
                throw std::invalid_argument("buildGraphFromEdges: invalid vertex index in edge.");
            }

            adj_[static_cast<size_t>(u)].push_back({v, w});
            if (symmetric) {
                adj_[static_cast<size_t>(v)].push_back({u, w});
            }
        }

        if (deduplicate) {
            deduplicateAdjacency();
        }

        graphBuilt_ = true;
    }
    std::vector<std::vector<std::vector<int>>>
    computeProjectionClustersNoLabel(const Eigen::VectorXi& cellSeedNums,
                                    const Eigen::VectorXi& nearestCloudIdsFlat) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeProjectionClustersNoLabel: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeProjectionClustersNoLabel: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeProjectionClustersNoLabel: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];

            std::unordered_map<int, std::vector<int>> seedLocalIndices;
            seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                seedLocalIndices[seed].push_back(local);
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> visited(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];

                if (visited[static_cast<size_t>(seed)]) {
                    continue;
                }

                std::vector<int> cluster;
                std::queue<int> q;

                visited[static_cast<size_t>(seed)] = 1;
                q.push(seed);

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();

                    auto it = seedLocalIndices.find(u);
                    if (it != seedLocalIndices.end()) {
                        for (int idx : it->second) {
                            cluster.push_back(idx);
                        }
                    }

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (visited[static_cast<size_t>(v)]) {
                            continue;
                        }
                        if (!isSeedOfCell[static_cast<size_t>(v)]) {
                            continue;
                        }

                        visited[static_cast<size_t>(v)] = 1;
                        q.push(v);
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }
    std::vector<std::vector<std::vector<int>>>
    computeProjectionClustersNoLabelOneRingBridge(const Eigen::VectorXi& cellSeedNums,
                                                const Eigen::VectorXi& nearestCloudIdsFlat) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeProjectionClustersNoLabelOneRingBridge: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeProjectionClustersNoLabelOneRingBridge: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeProjectionClustersNoLabelOneRingBridge: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];

            std::unordered_map<int, std::vector<int>> seedLocalIndices;
            seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                seedLocalIndices[seed].push_back(local);
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> visitedSeed(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];

                if (visitedSeed[static_cast<size_t>(seed)]) {
                    continue;
                }

                std::vector<int> cluster;
                std::queue<int> q;

                visitedSeed[static_cast<size_t>(seed)] = 1;
                q.push(seed);

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();
                    auto it = seedLocalIndices.find(u);
                    if (it != seedLocalIndices.end()) {
                        for (int idx : it->second) {
                            cluster.push_back(idx);
                        }
                    }

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (!isSeedOfCell[static_cast<size_t>(v)]) {
                            continue;
                        }

                        if (visitedSeed[static_cast<size_t>(v)]) {
                            continue;
                        }

                        visitedSeed[static_cast<size_t>(v)] = 1;
                        q.push(v);
                    }

                    for (const auto& [mid, w1] : adj_[static_cast<size_t>(u)]) {
                        (void)w1;

                        if (isSeedOfCell[static_cast<size_t>(mid)]) {
                            continue;
                        }

                        if (featureFlags_[static_cast<size_t>(mid)]) {
                            continue;
                        }

                        for (const auto& [v, w2] : adj_[static_cast<size_t>(mid)]) {
                            (void)w2;

                            if (!isSeedOfCell[static_cast<size_t>(v)]) {
                                continue;
                            }

                            if (visitedSeed[static_cast<size_t>(v)]) {
                                continue;
                            }

                            visitedSeed[static_cast<size_t>(v)] = 1;
                            q.push(v);
                        }
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }
    std::vector<std::vector<std::vector<int>>>
    computeProjectionClustersNoLabelOneRingExpanded(const Eigen::VectorXi& cellSeedNums,
                                                    const Eigen::VectorXi& nearestCloudIdsFlat) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeProjectionClustersNoLabelOneRingExpanded: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeProjectionClustersNoLabelOneRingExpanded: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeProjectionClustersNoLabelOneRingExpanded: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];

            std::unordered_map<int, std::vector<int>> seedLocalIndices;
            seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                seedLocalIndices[seed].push_back(local);
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> inExpandedSubgraph(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                inExpandedSubgraph[static_cast<size_t>(seed)] = 1;

                for (const auto& [nbr, w] : adj_[static_cast<size_t>(seed)]) {
                    (void)w;

                    if (isSeedOfCell[static_cast<size_t>(nbr)]) {
                        inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                        continue;
                    }

                    if (!featureFlags_(nbr)) {
                        inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                    }
                }
            }

            std::vector<char> visitedSeed(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];

                if (visitedSeed[static_cast<size_t>(seed)]) {
                    continue;
                }

                std::vector<int> cluster;

                std::queue<int> q;
                std::vector<char> visitedExpanded(static_cast<size_t>(rows_), 0);

                visitedExpanded[static_cast<size_t>(seed)] = 1;
                q.push(seed);

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();

                    auto it = seedLocalIndices.find(u);
                    if (it != seedLocalIndices.end()) {
                        if (!visitedSeed[static_cast<size_t>(u)]) {
                            visitedSeed[static_cast<size_t>(u)] = 1;
                        }
                        for (int idx : it->second) {
                            cluster.push_back(idx);
                        }
                    }

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (visitedExpanded[static_cast<size_t>(v)]) {
                            continue;
                        }

                        if (!inExpandedSubgraph[static_cast<size_t>(v)]) {
                            continue;
                        }

                        visitedExpanded[static_cast<size_t>(v)] = 1;
                        q.push(v);
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                cluster.erase(std::unique(cluster.begin(), cluster.end()), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }
    std::vector<std::vector<int>> computeCellProjectionClustersNoLabelOneRingExpandedInSamePatch(
        const Eigen::VectorXi& nearestCloudIds,
        const Eigen::VectorXi&patchIds
    ) const
    {
        const int n= nearestCloudIds.size();
        for (int i = 0; i < nearestCloudIds.size(); ++i) {
            int seed = nearestCloudIds[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeCellProjectionClustersNoLabelOneRingExpandedInSamePatch: nearestCloudIds contains invalid cloud point id.");
            }
        }
        std::unordered_map<int, std::vector<int>> seedLocalIndices;
        seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

        std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);
        int start=0;
        for (int local = 0; local < n; ++local) {
            const int seed = nearestCloudIds[start + local];
            seedLocalIndices[seed].push_back(local);
            isSeedOfCell[static_cast<size_t>(seed)] = 1;
        }

        std::vector<char> inExpandedSubgraph(static_cast<size_t>(rows_), 0);

        std::vector<int> cloudIdsV(nearestCloudIds.data(), nearestCloudIds.data() + nearestCloudIds.size());

        std::sort(cloudIdsV.begin(), cloudIdsV.end());
        auto last = std::unique(cloudIdsV.begin(), cloudIdsV.end());
        cloudIdsV.erase(last, cloudIdsV.end());
        for (int local = 0; local < cloudIdsV.size(); ++local) {
            const int seed = cloudIdsV[local];
            inExpandedSubgraph[static_cast<size_t>(seed)] = 1;
            for (const auto& [nbr, w] : adj_[static_cast<size_t>(seed)]) {
                (void)w;

                if (isSeedOfCell[static_cast<size_t>(nbr)]) {
                    inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                    continue;
                }

                if (!featureFlags_(nbr)) {
                    inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                }
            }
        }

        std::vector<char> visitedSeed(static_cast<size_t>(rows_), 0);
        std::vector<std::vector<int>> clusters;
        for (int local = 0; local < cloudIdsV.size(); ++local) {
            const int seed = cloudIdsV[local];

            if (visitedSeed[static_cast<size_t>(seed)]) {
                continue;
            }

            std::vector<int> cluster;

            std::queue<int> q;
            std::vector<char> visitedExpanded(static_cast<size_t>(rows_), 0);

            visitedExpanded[static_cast<size_t>(seed)] = 1;
            q.push(seed);
            int cluster_patch_id=-1;
            while (!q.empty()) {
                const int u = q.front();
                q.pop();
                if(patchIds(u)!=-1&&cluster_patch_id==-1){
                    cluster_patch_id=patchIds(u);
                }
                auto it = seedLocalIndices.find(u);
                if (it != seedLocalIndices.end()) {
                    if (!visitedSeed[static_cast<size_t>(u)]) {
                        visitedSeed[static_cast<size_t>(u)] = 1;
                        for (int idx : it->second) {
                            cluster.push_back(idx);
                        }
                    }
                    
                }

                for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                    (void)w;

                    if (visitedExpanded[static_cast<size_t>(v)]) {
                        continue;
                    }

                    if (!inExpandedSubgraph[static_cast<size_t>(v)]) {
                        continue;
                    }
                    if(cluster_patch_id!=-1&&patchIds(v)!=-1&&patchIds(v)!=cluster_patch_id){
                        std::cout<<patchIds(v)<<std::endl;
                        continue;
                    }
                    visitedExpanded[static_cast<size_t>(v)] = 1;
                    q.push(v);
                    if(patchIds(v)!=-1&&cluster_patch_id==-1){
                        cluster_patch_id=patchIds(v);
                    }
                }
            }

            std::sort(cluster.begin(), cluster.end());
            cluster.erase(std::unique(cluster.begin(), cluster.end()), cluster.end());
            clusters.push_back(std::move(cluster));
        }
        return clusters;
    }
    std::vector<std::vector<std::vector<int>>>
    computeProjectionClustersNoLabelOneRingExpandedInSamePatch(const Eigen::VectorXi& cellSeedNums,
                                                    const Eigen::VectorXi& nearestCloudIdsFlat,
                                                    const Eigen::VectorXi&patchIds) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeProjectionClustersNoLabelOneRingExpanded: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeProjectionClustersNoLabelOneRingExpanded: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeProjectionClustersNoLabelOneRingExpanded: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }
        if (patchIds.size() != rows_) {
            throw std::invalid_argument(
                "computeProjectionClustersNoLabelOneRingExpandedInSamePatch: "
                "patchIds.size() must equal point cloud size.");
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];

            std::unordered_map<int, std::vector<int>> seedLocalIndices;
            seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                seedLocalIndices[seed].push_back(local);
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> inExpandedSubgraph(static_cast<size_t>(rows_), 0);
            std::vector<int> cloudIdsV(nearestCloudIdsFlat.data()+offsets[cellIdx], nearestCloudIdsFlat.data()+offsets[cellIdx]+n);
            std::sort(cloudIdsV.begin(), cloudIdsV.end());
            auto last = std::unique(cloudIdsV.begin(), cloudIdsV.end());
            cloudIdsV.erase(last, cloudIdsV.end());
            for (int local = 0; local < cloudIdsV.size(); ++local) {
                const int seed = cloudIdsV[local];
                inExpandedSubgraph[static_cast<size_t>(seed)] = 1;

                for (const auto& [nbr, w] : adj_[static_cast<size_t>(seed)]) {
                    (void)w;

                    if (isSeedOfCell[static_cast<size_t>(nbr)]) {
                        inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                        continue;
                    }

                    if (!featureFlags_(nbr)) {
                        inExpandedSubgraph[static_cast<size_t>(nbr)] = 1;
                    }
                }
            }

            std::vector<char> visitedSeed(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < cloudIdsV.size(); ++local) {
                const int seed = cloudIdsV[local];

                if (visitedSeed[static_cast<size_t>(seed)]) {
                    continue;
                }

                std::vector<int> cluster;

                std::queue<int> q;
                std::vector<char> visitedExpanded(static_cast<size_t>(rows_), 0);

                visitedExpanded[static_cast<size_t>(seed)] = 1;
                q.push(seed);
                int cluster_patch_id=-1;
                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();
                    if(patchIds(u)!=-1&&cluster_patch_id==-1){
                        cluster_patch_id=patchIds(u);
                    }
                    auto it = seedLocalIndices.find(u);
                    if (it != seedLocalIndices.end()) {
                        if (!visitedSeed[static_cast<size_t>(u)]) {
                            visitedSeed[static_cast<size_t>(u)] = 1;
                            for (int idx : it->second) {
                                cluster.push_back(idx);
                            }
                        }
                        
                    }

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (visitedExpanded[static_cast<size_t>(v)]) {
                            continue;
                        }

                        if (!inExpandedSubgraph[static_cast<size_t>(v)]) {
                            continue;
                        }
                        if(cluster_patch_id!=-1&&patchIds(v)!=-1&&patchIds(v)!=cluster_patch_id){
                            continue;
                        }
                        visitedExpanded[static_cast<size_t>(v)] = 1;
                        q.push(v);
                        if(patchIds(v)!=-1&&cluster_patch_id==-1){
                            cluster_patch_id=patchIds(v);
                        }
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                cluster.erase(std::unique(cluster.begin(), cluster.end()), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }

    std::vector<std::vector<std::vector<int>>>
    computeProjectionClusters(const Eigen::VectorXi& cellSeedNums,
                              const Eigen::VectorXi& nearestCloudIdsFlat,
                              const VecI64& cellNodeIds) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeProjectionClusters: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeProjectionClusters: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument(
                "computeProjectionClusters: cellNodeIds.size() must equal cellSeedNums.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeProjectionClusters: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int>>> allClusters(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int64_t nodeId = cellNodeIds[cellIdx];
            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& clusters = allClusters[cellIdx];
            
            std::unordered_map<int, std::vector<int>> seedLocalIndices;
            seedLocalIndices.reserve(static_cast<size_t>(n) * 2);

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                seedLocalIndices[seed].push_back(local);
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> visited(static_cast<size_t>(rows_), 0);

            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];

                if (visited[static_cast<size_t>(seed)]) {
                    continue;
                }

                std::vector<int> cluster;
                std::queue<int> q;

                visited[static_cast<size_t>(seed)] = 1;
                q.push(seed);

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();

                    auto it = seedLocalIndices.find(u);
                    if (it != seedLocalIndices.end()) {
                        for (int idx : it->second) {
                            cluster.push_back(idx);
                        }
                    }

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        if (visited[static_cast<size_t>(v)]) {
                            continue;
                        }

                        const bool belongs =
                            isSeedOfCell[static_cast<size_t>(v)] || hasLabelSorted(v, nodeId);

                        if (!belongs) {
                            continue;
                        }

                        visited[static_cast<size_t>(v)] = 1;
                        q.push(v);
                    }
                }

                std::sort(cluster.begin(), cluster.end());
                clusters.push_back(std::move(cluster));
            }
        }

        return allClusters;
    }
    
    std::vector<std::vector<std::vector<int64_t>>>
    computeCellRelatedPointClusterBlockers(const Eigen::VectorXi& cellSeedNums,
                                        const Eigen::VectorXi& nearestCloudIdsFlat,
                                        const VecI64& cellNodeIds) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeCellRelatedPointClusterBlockers: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeCellRelatedPointClusterBlockers: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument(
                "computeCellRelatedPointClusterBlockers: cellNodeIds.size() must equal cellSeedNums.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeCellRelatedPointClusterBlockers: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        std::vector<int> offsets(numCells, 0);
        for (int c = 1; c < numCells; ++c) {
            offsets[c] = offsets[c - 1] + cellSeedNums[c - 1];
        }

        std::vector<std::vector<std::vector<int64_t>>> allBlockers(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {

            const int64_t nodeId = cellNodeIds[cellIdx];
            const int start = offsets[cellIdx];
            const int n     = cellSeedNums[cellIdx];

            auto& cellBlockers = allBlockers[cellIdx];

            std::vector<char> isSeedOfCell(static_cast<size_t>(rows_), 0);
            for (int local = 0; local < n; ++local) {
                const int seed = nearestCloudIdsFlat[start + local];
                isSeedOfCell[static_cast<size_t>(seed)] = 1;
            }

            std::vector<char> visited(static_cast<size_t>(rows_), 0);

            for (int u0 = 0; u0 < rows_; ++u0) {
                if (visited[static_cast<size_t>(u0)]) {
                    continue;
                }

                const bool belongs0 =
                    isSeedOfCell[static_cast<size_t>(u0)] || hasLabelSorted(u0, nodeId);

                if (!belongs0) {
                    continue;
                }

                std::queue<int> q;
                visited[static_cast<size_t>(u0)] = 1;
                q.push(u0);

                std::set<int64_t> blockerSet;

                while (!q.empty()) {
                    const int u = q.front();
                    q.pop();

                    for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                        (void)w;

                        const bool belongs =
                            isSeedOfCell[static_cast<size_t>(v)] || hasLabelSorted(v, nodeId);

                        if (!belongs) {
                            const auto& labelsV = results[static_cast<size_t>(v)].labels;
                            for (int64_t otherNodeId : labelsV) {
                                if (otherNodeId != nodeId) {
                                    blockerSet.insert(otherNodeId);
                                }
                            }
                            continue;
                        }

                        if (!visited[static_cast<size_t>(v)]) {
                            visited[static_cast<size_t>(v)] = 1;
                            q.push(v);
                        }
                    }
                }

                cellBlockers.emplace_back(blockerSet.begin(), blockerSet.end());
            }
        }

        return allBlockers;
    }
    
    std::vector<std::vector<int64_t>>
    computeCellBlockerNodeIds(const Eigen::VectorXi& cellSeedNums,
                            const Eigen::VectorXi& nearestCloudIdsFlat,
                            const VecI64& cellNodeIds) const
    {
        const auto allClusterBlockers =
            computeCellRelatedPointClusterBlockers(cellSeedNums, nearestCloudIdsFlat, cellNodeIds);

        const int numCells = static_cast<int>(allClusterBlockers.size());
        std::vector<std::vector<int64_t>> result(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            std::set<int64_t> blockerSet;

            for (const auto& clusterBlockers : allClusterBlockers[cellIdx]) {
                for (int64_t nodeId : clusterBlockers) {
                    blockerSet.insert(nodeId);
                }
            }

            result[cellIdx].assign(blockerSet.begin(), blockerSet.end());
        }

        return result;
    }
    
    std::pair<std::vector<std::vector<int64_t>>,
          std::vector<std::pair<int64_t, int64_t>>>
    computeMutualBlockerPairs(const Eigen::VectorXi& cellSeedNums,
                            const Eigen::VectorXi& nearestCloudIdsFlat,
                            const VecI64& cellNodeIds) const
    {
        const int numCells = static_cast<int>(cellSeedNums.size());

        if (numCells <= 0) {
            throw std::invalid_argument("computeMutualBlockerPairs: cellSeedNums is empty.");
        }

        if (cellSeedNums.sum() != nearestCloudIdsFlat.size()) {
            throw std::invalid_argument(
                "computeMutualBlockerPairs: sum(cellSeedNums) must equal nearestCloudIdsFlat.size().");
        }

        if (cellNodeIds.size() != numCells) {
            throw std::invalid_argument(
                "computeMutualBlockerPairs: cellNodeIds.size() must equal cellSeedNums.size().");
        }

        for (int i = 0; i < nearestCloudIdsFlat.size(); ++i) {
            const int seed = nearestCloudIdsFlat[i];
            if (seed < 0 || seed >= rows_) {
                throw std::invalid_argument(
                    "computeMutualBlockerPairs: nearestCloudIdsFlat contains invalid cloud point id.");
            }
        }

        const auto allClusterBlockers =
            computeCellRelatedPointClusterBlockers(cellSeedNums, nearestCloudIdsFlat, cellNodeIds);

        std::vector<std::vector<int64_t>> cellBlockerNodeIds(numCells);

        #pragma omp parallel for schedule(static)
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            std::set<int64_t> blockerSet;

            for (const auto& clusterBlockers : allClusterBlockers[cellIdx]) {
                for (int64_t otherNodeId : clusterBlockers) {
                    blockerSet.insert(otherNodeId);
                }
            }

            cellBlockerNodeIds[cellIdx].assign(blockerSet.begin(), blockerSet.end());
        }

        std::unordered_map<int64_t, int> nodeIdToCellIdx;
        nodeIdToCellIdx.reserve(static_cast<size_t>(numCells) * 2);

        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            nodeIdToCellIdx[cellNodeIds[cellIdx]] = cellIdx;
        }

        std::set<std::pair<int64_t, int64_t>> mutualPairSet;

        for (int cellIdxA = 0; cellIdxA < numCells; ++cellIdxA) {
            const int64_t nodeIdA = cellNodeIds[cellIdxA];
            if (allClusterBlockers[cellIdxA].size() < 2) {
                continue;
            }

            const auto& blockersA = cellBlockerNodeIds[cellIdxA];

            for (int64_t nodeIdB : blockersA) {
                auto it = nodeIdToCellIdx.find(nodeIdB);
                if (it == nodeIdToCellIdx.end()) {
                    continue;
                }

                const int cellIdxB = it->second;

                if (allClusterBlockers[cellIdxB].size() < 2) {
                    continue;
                }

                const auto& blockersB = cellBlockerNodeIds[cellIdxB];

                if (std::binary_search(blockersB.begin(), blockersB.end(), nodeIdA)) {
                    const int64_t a = std::min(nodeIdA, nodeIdB);
                    const int64_t b = std::max(nodeIdA, nodeIdB);
                    if (a != b) {
                        mutualPairSet.insert({a, b});
                    }
                }
            }
        }

        std::vector<std::pair<int64_t, int64_t>> mutualBlockerPairs(
            mutualPairSet.begin(), mutualPairSet.end()
        );

        return std::make_pair(cellBlockerNodeIds, mutualBlockerPairs);
    }
    

    void buildKnnGraph(
        NanoKNNSearch<T>& knn,
        int K,
        bool symmetric = true,
        T maxSeparateNormalAngleDeg = static_cast<T>(90),
        T maxConnectNormalAngleDeg = static_cast<T>(90),
        int candidateMultiplier = 1)
    {
        if (K <= 0) {
            throw std::invalid_argument("buildKnnGraph: K must be positive.");
        }
        if (points_.rows() != normals_.rows()) {
            throw std::invalid_argument("buildKnnGraph: points number must be equal to normals number.");
        }
        if (K >= points_.rows()) {
            throw std::invalid_argument("buildKnnGraph: K must be smaller than rows.");
        }
        if (maxSeparateNormalAngleDeg < static_cast<T>(0) || maxSeparateNormalAngleDeg > static_cast<T>(90)) {
            throw std::invalid_argument("buildKnnGraph: maxSeparateNormalAngleDeg must be in [0, 90].");
        }
        if (maxConnectNormalAngleDeg < static_cast<T>(0) || maxConnectNormalAngleDeg > static_cast<T>(90)) {
            throw std::invalid_argument("buildKnnGraph: maxConnectNormalAngleDeg must be in [0, 90].");
        }
        if (candidateMultiplier <= 0) {
            throw std::invalid_argument("buildKnnGraph: candidateMultiplier must be positive.");
        }

        int candidateK = K + 1;

        candidateK = std::min<int64_t>(
            rows_,
            static_cast<int64_t>(candidateMultiplier * K + 1)
        );
        if (candidateK <= K) {
            candidateK = std::min<int64_t>(rows_, static_cast<int64_t>(K + 1));
        }

        std::vector<int> indices;
        std::vector<T> dists_dummy;
        knn.searchKNN(indices, dists_dummy, points_.data(), points_.rows(), 3, candidateK);

        std::vector<std::vector<int>> knn_list(rows_);
        knn_list.assign(rows_, {});

        const T cosThreshold = std::cos(maxSeparateNormalAngleDeg * static_cast<T>(M_PI / 180.0));
        const T cosFeatureAndNonFeature=std::cos(maxConnectNormalAngleDeg * static_cast<T>(M_PI / 180.0));
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < rows_; ++i) {
            auto& nbrs = knn_list[i];
            nbrs.reserve(K);

            for (int t = 0; t < candidateK; ++t) {
                int j = indices[static_cast<size_t>(i) * candidateK + t];
                if (j < 0 || j >= rows_) continue;
                if (j == i) continue;
                if (!featureFlags_[i]&&!featureFlags_[j]) {
                    //both not feature
                    if(!normalPass(i, j, normals_, cosThreshold, false))
                        continue;
                }else if(featureFlags_[i]&&featureFlags_[j]){
                     //both feature
                    if(t>1 && !normalPass(i, j, normals_, cosFeatureAndFeature, false))
                        continue;
                }else{
                    //feature and not feature
                    if(t>1 &&!normalPass(i, j, normals_, cosFeatureAndNonFeature, false)){
                        continue;
                    }
                }

                nbrs.push_back(j);
                // if (featureFlags_[i] && nbrs.size()>=8){
                //     break;
                // }
                if ((int)nbrs.size() == K) break;
            }
        }

        adj_.assign(rows_, {});
        for (int64_t i = 0; i < rows_; ++i) {
            adj_[i].reserve(knn_list[i].size());
        }

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < rows_; ++i) {
            auto& nbrs = adj_[i];
            nbrs.reserve(knn_list[i].size());

            for (int j : knn_list[i]) {
                T w = (points_.row(i) - points_.row(j)).norm();
                nbrs.push_back({j, w});
            }
        }

        if (symmetric) {
            std::vector<std::vector<Edge>> adjSym(rows_);

            for (int64_t i = 0; i < rows_; ++i) {
                for (const auto& e : adj_[i]) {
                    int j = e.first;
                    T w = e.second;
                    adjSym[i].push_back({j, w});
                    adjSym[j].push_back({(int)i, w});
                }
            }

            adj_.swap(adjSym);
            deduplicateAdjacency();
        }
        graphBuilt_ = true;
    }

    void deduplicateAdjacency()
    {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < rows_; ++i) {
            auto& nbrs = adj_[i];

            std::sort(
                nbrs.begin(),
                nbrs.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                }
            );

            std::vector<Edge> uniqueNbrs;
            uniqueNbrs.reserve(nbrs.size());

            for (const auto& e : nbrs) {
                if (uniqueNbrs.empty() || uniqueNbrs.back().first != e.first) {
                    uniqueNbrs.push_back(e);
                } else if (e.second < uniqueNbrs.back().second) {
                    uniqueNbrs.back().second = e.second;
                }
            }

            nbrs.swap(uniqueNbrs);
        }
    }
    std::vector<int> computeSmoothPatchIds()
    {
        if (rows_ <= 0) {
            return {};
        }

        if (!graphBuilt_) {
            throw std::runtime_error(
                "computeSmoothPatchIds: graph has not been built. "
                "Please call buildKnnGraph/buildRadiusGraph/buildHybridGraph first."
            );
        }

        const int n = static_cast<int>(rows_);
        UnionFindWithSize uf(n);

        // Compute connected components of the non-feature induced subgraph G[V \ F].
        for (int u = 0; u < n; ++u) {
            if (featureFlags_(u)) {
                continue;
            }

            for (const auto& [v, w] : adj_[static_cast<size_t>(u)]) {
                (void)w;

                if (v < 0 || v >= n) {
                    continue;
                }

                if (featureFlags_(v)) {
                    continue;
                }

                uf.unite(u, v);
            }
        }

        std::vector<int> patchIds(n, -1);
        std::unordered_map<int, int> rootToPatchId;
        rootToPatchId.reserve(static_cast<size_t>(n));

        int nextPatchId = 0;

        for (int i = 0; i < n; ++i) {
            if (featureFlags_(i)) {
                patchIds[i] = -1;
                continue;
            }

            int root = uf.find(i);

            auto it = rootToPatchId.find(root);
            if (it == rootToPatchId.end()) {
                rootToPatchId.emplace(root, nextPatchId);
                patchIds[i] = nextPatchId;
                ++nextPatchId;
            } else {
                patchIds[i] = it->second;
            }
        }

        return patchIds;
    }
public:
    void assign_result(int start, int n, const Result& val)
    {
        std::fill(results.begin() + start, results.begin() + start + n, val);
    }

    const std::vector<Result>& get_results() const { return results; }

private:
    static bool appendLabelUnique(std::vector<int64_t>& labels, int64_t label)
    {
        if (std::find(labels.begin(), labels.end(), label) != labels.end()) {
            return false;
        }
        labels.push_back(label);
        return true;
    }

    static bool mergeLabels(std::vector<int64_t>& dst, const std::vector<int64_t>& src)
    {
        bool changed = false;
        for (int64_t x : src) {
            if (std::find(dst.begin(), dst.end(), x) == dst.end()) {
                dst.push_back(x);
                changed = true;
            }
        }
        return changed;
    }

private:
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    bool graphBuilt_ = false;

    std::vector<std::vector<Edge>> adj_;
    MatT3 points_;
    MatT3 normals_;
    ArrayB1 featureFlags_;
    std::vector<Result> results;
};
std::vector<std::vector<std::vector<int>>> cloud_grouping(
    MatF3& cloud,
    MatF3& cloudNormals,
    const ArrayB1& cloudFeatureFlags,
    const Eigen::VectorXi& cellSeedNums,
    const Eigen::VectorXi& nearestCloudIdsFlat,
    const VecI64& cellNodeIds);
#endif // __CLOUD_RVD_H__
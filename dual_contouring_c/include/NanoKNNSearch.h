#ifndef NANO_KNNSEARCH_H
#define NANO_KNNSEARCH_H

#include <nanoflann.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class NanoKNNSearch
{
public:
    NanoKNNSearch(const T* data, int64_t rows, int64_t cols = 3, int searchThreads = 4)
        : rows_(rows), cols_(cols), searchThreads_(searchThreads)
    {
        if (data == nullptr) {
            throw std::invalid_argument("NanoKNNSearch: data is null.");
        }
        if (rows_ <= 0) {
            throw std::invalid_argument("NanoKNNSearch: rows must be positive.");
        }
        if (cols_ <= 0) {
            throw std::invalid_argument("NanoKNNSearch: cols must be positive.");
        }
        if (searchThreads_ <= 0) {
            throw std::invalid_argument("NanoKNNSearch: searchThreads must be positive.");
        }

        dataset_.resize(static_cast<size_t>(rows_) * static_cast<size_t>(cols_));
        std::memcpy(dataset_.data(), data, sizeof(T) * dataset_.size());

        adaptor_ = std::make_unique<PointCloudAdaptor>(dataset_, rows_, cols_);
        index_ = std::make_unique<KDTree>(
            static_cast<int>(cols_), *adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10)
        );
        index_->buildIndex();
    }

    ~NanoKNNSearch() = default;

    NanoKNNSearch(const NanoKNNSearch&) = delete;
    NanoKNNSearch& operator=(const NanoKNNSearch&) = delete;

    NanoKNNSearch(NanoKNNSearch&& other) noexcept
        : rows_(other.rows_),
          cols_(other.cols_),
          dataset_(std::move(other.dataset_)),
          searchThreads_(other.searchThreads_),
          adaptor_(std::move(other.adaptor_)),
          index_(std::move(other.index_))
    {
        other.rows_ = 0;
        other.cols_ = 0;
        other.searchThreads_ = 4;
    }

    NanoKNNSearch& operator=(NanoKNNSearch&& other) noexcept
    {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            dataset_ = std::move(other.dataset_);
            searchThreads_ = other.searchThreads_;
            adaptor_ = std::move(other.adaptor_);
            index_ = std::move(other.index_);

            other.rows_ = 0;
            other.cols_ = 0;
            other.searchThreads_ = 4;
        }
        return *this;
    }

    void searchKNN(std::vector<int>& indices,
                   std::vector<T>& distances,
                   const T* pts,
                   int64_t queryRows,
                   int64_t queryCols,
                   int K = 1) const
    {
        if (pts == nullptr) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: pts is null.");
        }
        if (queryRows <= 0) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: queryRows must be positive.");
        }
        if (queryCols != cols_) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: queryCols must equal dataset cols.");
        }
        if (K <= 0) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: K must be positive.");
        }
        if (K > rows_) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: K must be <= dataset rows.");
        }

        indices.resize(static_cast<size_t>(queryRows) * static_cast<size_t>(K));
        distances.resize(static_cast<size_t>(queryRows) * static_cast<size_t>(K));

        #pragma omp parallel for schedule(static) num_threads(searchThreads_)
        for (int64_t qi = 0; qi < queryRows; ++qi) {
            const T* query = pts + qi * queryCols;

            std::vector<uint32_t> ret_index(static_cast<size_t>(K));
            std::vector<T> out_dist_sqr(static_cast<size_t>(K));

            nanoflann::KNNResultSet<T, uint32_t> resultSet(static_cast<size_t>(K));
            resultSet.init(ret_index.data(), out_dist_sqr.data());
            index_->findNeighbors(resultSet, query, nanoflann::SearchParameters());

            for (int k = 0; k < K; ++k) {
                indices[static_cast<size_t>(qi) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                    static_cast<int>(ret_index[static_cast<size_t>(k)]);
                distances[static_cast<size_t>(qi) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                    out_dist_sqr[static_cast<size_t>(k)];
            }
        }
    }

    void searchKNN(int* indices,
                   T* distances,
                   const T* pts,
                   int64_t queryRows,
                   int64_t queryCols,
                   int K = 1) const
    {
        if (indices == nullptr || distances == nullptr || pts == nullptr) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: null output/input pointer.");
        }
        if (queryRows <= 0) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: queryRows must be positive.");
        }
        if (queryCols != cols_) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: queryCols must equal dataset cols.");
        }
        if (K <= 0) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: K must be positive.");
        }
        if (K > rows_) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN: K must be <= dataset rows.");
        }

        #pragma omp parallel for schedule(static) num_threads(searchThreads_)
        for (int64_t qi = 0; qi < queryRows; ++qi) {
            const T* query = pts + qi * queryCols;
            int* out_idx = indices + qi * K;
            T* out_dist = distances + qi * K;

            std::vector<uint32_t> ret_index(static_cast<size_t>(K));
            nanoflann::KNNResultSet<T, uint32_t> resultSet(static_cast<size_t>(K));
            resultSet.init(ret_index.data(), out_dist);
            index_->findNeighbors(resultSet, query, nanoflann::SearchParameters());

            for (int k = 0; k < K; ++k) {
                out_idx[k] = static_cast<int>(ret_index[static_cast<size_t>(k)]);
            }
        }
    }

    void radiusSearch(std::vector<int>& indices,
                      std::vector<T>& distances,
                      const T* pt,
                      T radius) const
    {
        if (pt == nullptr) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: pt is null.");
        }
        if (radius < static_cast<T>(0)) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: radius must be non-negative.");
        }

        std::vector<std::vector<int>> allIndices;
        std::vector<std::vector<T>> allDistances;
        radiusSearch(allIndices, allDistances, pt, 1, cols_, radius);

        indices.clear();
        distances.clear();

        if (!allIndices.empty()) {
            indices = std::move(allIndices[0]);
            distances = std::move(allDistances[0]);
        }
    }

    void radiusSearch(std::vector<std::vector<int>>& indices,
                      std::vector<std::vector<T>>& distances,
                      const T* pts,
                      int64_t queryRows,
                      int64_t queryCols,
                      T radius) const
    {
        if (pts == nullptr) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: pts is null.");
        }
        if (queryRows <= 0) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: queryRows must be positive.");
        }
        if (queryCols != cols_) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: queryCols must equal dataset cols.");
        }
        if (radius < static_cast<T>(0)) {
            throw std::invalid_argument("NanoKNNSearch::radiusSearch: radius must be non-negative.");
        }

        const T radiusSquared = radius * radius;
        indices.assign(static_cast<size_t>(queryRows), {});
        distances.assign(static_cast<size_t>(queryRows), {});

        #pragma omp parallel for schedule(static) num_threads(searchThreads_)
        for (int64_t qi = 0; qi < queryRows; ++qi) {
            const T* query = pts + qi * queryCols;

            std::vector<nanoflann::ResultItem<uint32_t, T>> matches;
            nanoflann::RadiusResultSet<T, uint32_t> resultSet(radiusSquared, matches);
            index_->findNeighbors(resultSet, query, nanoflann::SearchParameters());

            auto& outIdx = indices[static_cast<size_t>(qi)];
            auto& outDist = distances[static_cast<size_t>(qi)];
            outIdx.resize(matches.size());
            outDist.resize(matches.size());

            for (size_t i = 0; i < matches.size(); ++i) {
                outIdx[i] = static_cast<int>(matches[i].first);
                outDist[i] = matches[i].second;
            }
        }
    }
    void searchKNN1(int& index, T& dist2, const T* pt) const
    {
        if (pt == nullptr) {
            throw std::invalid_argument("NanoKNNSearch::searchKNN1: pt is null.");
        }

        uint32_t outIndex = 0;
        nanoflann::KNNResultSet<T, uint32_t> resultSet(1);
        resultSet.init(&outIndex, &dist2);
        index_->findNeighbors(resultSet, pt, nanoflann::SearchParameters());

        index = static_cast<int>(outIndex);
    }
    int64_t rows() const noexcept { return rows_; }
    int64_t cols() const noexcept { return cols_; }
    int searchThreads() const noexcept { return searchThreads_; }

private:
    struct PointCloudAdaptor
    {
        const std::vector<T>& dataset;
        int64_t rows;
        int64_t cols;

        PointCloudAdaptor(const std::vector<T>& dataset_, int64_t rows_, int64_t cols_)
            : dataset(dataset_), rows(rows_), cols(cols_) {}

        inline size_t kdtree_get_point_count() const
        {
            return static_cast<size_t>(rows);
        }

        inline T kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            return dataset[idx * static_cast<size_t>(cols) + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const
        {
            return false;
        }
    };

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, PointCloudAdaptor>,
        PointCloudAdaptor,
        -1,
        uint32_t
    >;

private:
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    std::vector<T> dataset_;
    int searchThreads_ = 4;

    std::unique_ptr<PointCloudAdaptor> adaptor_;
    std::unique_ptr<KDTree> index_;
};

#endif // NANO_KNNSEARCH_H
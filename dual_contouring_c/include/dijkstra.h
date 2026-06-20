#ifndef DIJKSTRA_H
#define DIJKSTRA_H
#define _USE_MATH_DEFINES
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <type_traits>

template<typename T>
struct Path
{
    T dis = std::numeric_limits<T>::infinity();
    bool reachable = false;
    std::vector<int> path;
};

template<typename T>
class Dijkstra
{
    static_assert(std::is_floating_point_v<T>,
                  "Dijkstra<T>: T should be a floating-point type, e.g. float or double.");

public:
    using route = Path<T>;

    Dijkstra(const std::vector<T>& points, int dimensions, int srcId = 0)
        : points_(points), dimensions_(dimensions), srcId_(srcId)
    {
        if (dimensions_ <= 0) {
            throw std::invalid_argument("dimensions must be positive.");
        }
        if (points_.size() % static_cast<size_t>(dimensions_) != 0) {
            throw std::invalid_argument("points.size() must be divisible by dimensions.");
        }

        vexnum_ = static_cast<int>(points_.size() / dimensions_);
        if (vexnum_ == 0) {
            throw std::invalid_argument("points is empty.");
        }
        checkVertexId(srcId_);
    }

    void InitGraph(const std::vector<int>& edges)
    {
        if (edges.size() % 2 != 0) {
            throw std::invalid_argument("edges.size() must be even.");
        }

        adj_.assign(vexnum_, {});

        for (size_t k = 0; k < edges.size(); k += 2) {
            int u = edges[k];
            int v = edges[k + 1];

            checkVertexId(u);
            checkVertexId(v);

            if (u == v) {
                continue;
            }

            T w = distance(u, v);
            adj_[u].push_back({v, w});
            adj_[v].push_back({u, w});
        }

        initialized_ = true;
    }

    std::vector<route> searchPath(const std::vector<int>& edges)
    {
        return searchPath(edges, srcId_);
    }

    std::vector<route> searchPath(const std::vector<int>& edges, int srcId)
    {
        InitGraph(edges);
        return searchPath(srcId);
    }

    std::vector<route> searchPath()
    {
        return searchPath(srcId_);
    }

    std::vector<route> searchPath(int srcId)
    {
        if (!initialized_) {
            throw std::runtime_error("Graph is not initialized. Call InitGraph(edges) first.");
        }
        checkVertexId(srcId);

        const T INF = std::numeric_limits<T>::infinity();

        std::vector<T> dist(vexnum_, INF);
        std::vector<int> prev(vexnum_, -1);
        std::vector<bool> visited(vexnum_, false);

        using Node = std::pair<T, int>; // {distance, vertex}

        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

        dist[srcId] = static_cast<T>(0);
        pq.push({static_cast<T>(0), srcId});

        while (!pq.empty()) {
            auto [curDist, u] = pq.top();
            pq.pop();

            if (visited[u]) {
                continue;
            }
            visited[u] = true;

            if (curDist > dist[u]) {

                continue;
            }

            for (const auto& e : adj_[u]) {
                int v = e.to;
                T w = e.weight;

                if (visited[v]) {
                    continue;
                }

                T nd = dist[u] + w;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.push({nd, v});
                }
            }
        }

        std::vector<route> result(vexnum_);
        for (int v = 0; v < vexnum_; ++v) {
            result[v].dis = dist[v];
            result[v].reachable = std::isfinite(dist[v]);

            if (result[v].reachable) {
                result[v].path = reconstructPath(srcId, v, prev);
            }
        }

        return result;
    }

    void printPath(int srcId, const std::vector<route>& result) const
    {
        checkVertexId(srcId);

        for (int v = 0; v < vexnum_; ++v) {
            if (v == srcId) {
                continue;
            }

            std::cout << "from " << srcId << " to " << v << ":\n";
            if (!result[v].reachable) {
                std::cout << "  unreachable\n";
                continue;
            }

            std::cout << "  distance = " << result[v].dis << "\n";
            std::cout << "  path     = ";

            for (size_t i = 0; i < result[v].path.size(); ++i) {
                if (i > 0) {
                    std::cout << " -> ";
                }
                std::cout << result[v].path[i];
            }
            std::cout << "\n";
        }
    }

    int vertexCount() const { return vexnum_; }
    int dimensions() const { return dimensions_; }
    int sourceId() const { return srcId_; }

private:
    struct Edge
    {
        int to;
        T weight;
    };

private:
    void checkVertexId(int id) const
    {
        if (id < 0 || id >= vexnum_) {
            throw std::out_of_range("vertex id out of range.");
        }
    }

    T distance(int i, int j) const
    {
        const T* p1 = points_.data() + static_cast<size_t>(i) * dimensions_;
        const T* p2 = points_.data() + static_cast<size_t>(j) * dimensions_;

        T sum2 = static_cast<T>(0);
        for (int k = 0; k < dimensions_; ++k) {
            T diff = p1[k] - p2[k];
            sum2 += diff * diff;
        }
        return std::sqrt(sum2);
    }

    std::vector<int> reconstructPath(int srcId, int dstId, const std::vector<int>& prev) const
    {
        std::vector<int> path;
        for (int cur = dstId; cur != -1; cur = prev[cur]) {
            path.push_back(cur);
        }
        std::reverse(path.begin(), path.end());

        if (path.empty() || path.front() != srcId) {
            return {};
        }
        return path;
    }

private:
    const std::vector<T>& points_;
    int dimensions_ = 0;
    int vexnum_ = 0;
    int srcId_ = 0;
    bool initialized_ = false;

    std::vector<std::vector<Edge>> adj_;
};

#endif // DIJKSTRA_H

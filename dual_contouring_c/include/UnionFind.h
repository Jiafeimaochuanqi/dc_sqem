#ifndef __UNIONFIND_H__
#define __UNIONFIND_H__
class UnionFindWithSize {
public:
    explicit UnionFindWithSize(int n)
        : parent_(n), size_(n, 1), count_(n)
    {
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    int find(int x)
    {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    int findIterative(int x)
    {
        int root = x;

        while (parent_[root] != root) {
            root = parent_[root];
        }

        while (parent_[x] != root) {
            int p = parent_[x];
            parent_[x] = root;
            x = p;
        }

        return root;
    }

    bool unite(int x, int y)
    {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) {
            return false;
        }

        if (size_[rootX] < size_[rootY]) {
            parent_[rootX] = rootY;
            size_[rootY] += size_[rootX];
        } else {
            parent_[rootY] = rootX;
            size_[rootX] += size_[rootY];
        }

        --count_;
        return true;
    }

    bool isConnected(int x, int y)
    {
        return find(x) == find(y);
    }

    int getSize(int x)
    {
        int root = find(x);
        return size_[root];
    }

    int getCount() const
    {
        return count_;
    }

    std::unordered_map<int, std::vector<int>> getComponents()
    {
        std::unordered_map<int, std::vector<int>> components;

        for (int i = 0; i < static_cast<int>(parent_.size()); ++i) {
            int root = find(i);
            components[root].push_back(i);
        }

        return components;
    }

    int size() const
    {
        return static_cast<int>(parent_.size());
    }

private:
    std::vector<int> parent_;
    std::vector<int> size_;
    int count_;
};
#endif // __UNIONFIND_H__
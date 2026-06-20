#ifndef __MULPRIORITYQUEUE_H__
#define __MULPRIORITYQUEUE_H__
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// C++ version of Python MulPriorityQueue:
// - priority is tuple-like vector<double>, compared lexicographically.
// - smaller priority is popped first.
// - if priorities are equal, earlier insertion counter is popped first.
// - update_priority uses lazy deletion for heap entries.
// - traverse_via_map preserves Python-dict-like insertion order using std::list.
//
// Item must be hashable and equality-comparable, e.g. std::string, int.
template <class Item, class Hash = std::hash<Item>, class Equal = std::equal_to<Item>>
class MulPriorityQueue {
public:
    using Priority = std::vector<float>;

private:
    struct Entry {
        Priority priority;
        std::size_t counter = 0;
        Item item;
    };

    // std::priority_queue is a max-heap by default. This comparator reverses
    // the order so that the lexicographically smallest priority is on top.
    struct HeapCompare {
        bool operator()(const Entry& a, const Entry& b) const {
            if (a.priority != b.priority) {
                return a.priority > b.priority;
            }
            return a.counter > b.counter;
        }
    };

    using OrderList = std::list<Item>;
    using OrderIt = typename OrderList::iterator;
    using ConstOrderIt = typename OrderList::const_iterator;

    struct MapValue {
        Priority priority;
        std::size_t counter = 0;
        OrderIt order_it;
    };

    std::priority_queue<Entry, std::vector<Entry>, HeapCompare> heap_;//less in front
    std::unordered_map<Item, MapValue, Hash, Equal> entry_map_;
    OrderList order_;                  // emulate Python dict insertion order
    std::size_t counter_ = 0;
    mutable std::mutex mutex_;

    bool is_current_unlocked(const Entry& e) const {
        auto it = entry_map_.find(e.item);
        return it != entry_map_.end()
            && it->second.counter == e.counter
            && it->second.priority == e.priority;
    }

    void remove_item_unlocked(const Item& item) {
        auto it = entry_map_.find(item);
        if (it == entry_map_.end()) {
            return;
        }
        order_.erase(it->second.order_it);
        entry_map_.erase(it);
    }

    void push_unlocked(const Item& item, const Priority& priority) {
        auto it = entry_map_.find(item);
        if (it != entry_map_.end()) {
            if (it->second.priority == priority) {
                return;
            }

            // Match Python implementation:
            // remove old key, then insert a new key.
            // So updated items move to the end of insertion order.
            remove_item_unlocked(item);
        }

        Entry entry{priority, counter_, item};
        heap_.push(entry);

        order_.push_back(item);
        auto order_it = std::prev(order_.end());
        entry_map_.emplace(item, MapValue{priority, counter_, order_it});

        ++counter_;
    }

public:
    MulPriorityQueue() = default;

    void push(const Item& item, const Priority& priority) {
        std::lock_guard<std::mutex> lock(mutex_);
        push_unlocked(item, priority);
    }

    void update_priority(const Item& item, const Priority& new_priority) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entry_map_.find(item) != entry_map_.end()) {
            push_unlocked(item, new_priority);
        }
    }

    std::pair<Item, Priority> pop() {
        std::lock_guard<std::mutex> lock(mutex_);

        while (!heap_.empty()) {
            Entry entry = heap_.top();
            heap_.pop();

            if (is_current_unlocked(entry)) {
                auto it = entry_map_.find(entry.item);
                if (it != entry_map_.end()) {
                    order_.erase(it->second.order_it);
                    entry_map_.erase(it);
                }
                return {entry.item, entry.priority};
            }
        }

        throw std::out_of_range("queue is empty");
    }

    bool contains(const Item& item) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entry_map_.find(item) != entry_map_.end();
    }

    bool is_empty() const {
        return size() == 0;
    }

    explicit operator bool() const {
        return !is_empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entry_map_.size();
    }

    std::vector<Item> traverse_via_map() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Item> result;
        result.reserve(entry_map_.size());

        for (ConstOrderIt it = order_.cbegin(); it != order_.cend(); ++it) {
            result.push_back(*it);
        }

        return result;
    }

    std::vector<std::pair<Item, Priority>> items_sorted() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Entry> valid_entries;
        valid_entries.reserve(entry_map_.size());

        auto heap_copy = heap_;
        while (!heap_copy.empty()) {
            Entry entry = heap_copy.top();
            heap_copy.pop();

            if (is_current_unlocked(entry)) {
                valid_entries.push_back(std::move(entry));
            }
        }

        std::sort(
            valid_entries.begin(),
            valid_entries.end(),
            [](const Entry& a, const Entry& b) {
                if (a.priority != b.priority) {
                    return a.priority < b.priority;
                }
                return a.counter < b.counter;
            }
        );

        std::vector<std::pair<Item, Priority>> result;
        result.reserve(valid_entries.size());

        for (const auto& entry : valid_entries) {
            result.push_back({entry.item, entry.priority});
        }

        return result;
    }
};

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << '(';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << values[i];
    }
    os << ')';
    return os;
}
#endif // __MULPRIORITYQUEUE_H__
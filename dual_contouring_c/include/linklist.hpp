#pragma once
#include <memory>
#include <string>
#include <sstream>
#include <iostream>

class Node {
public:
    int data;
    std::shared_ptr<Node> prev;
    std::shared_ptr<Node> next;

    Node(int d) : data(d), prev(nullptr), next(nullptr) {}
};

class EdgeNode : public Node {
public:
    int cavity_id;
    int next_vertex_id;

    EdgeNode(int d, int cavity = -1)
        : Node(d), cavity_id(cavity), next_vertex_id(-1) {}

    std::string get_info() const {
        std::ostringstream oss;
        oss << "data=" << data << " cavity_id=" << cavity_id
            << " next_vertex_id=" << next_vertex_id;
        return oss.str();
    }
};

class DoublyLinkedList {
private:
    std::shared_ptr<Node> head;
    std::shared_ptr<Node> tail;
    size_t size;

public:
    DoublyLinkedList() : head(nullptr), tail(nullptr), size(0) {}

    bool is_empty() const { return size == 0; }

    void append(std::shared_ptr<Node> node) {
        if (!node) return;

        if (is_empty()) {
            head = tail = node;
        } else {
            node->prev = tail;
            tail->next = node;
            tail = node;
        }
        ++size;
    }

    void append(int data) {
        append(std::make_shared<Node>(data));
    }

    size_t get_size() const { return size; }

    std::string traverse_forward() const {
        std::ostringstream oss;
        auto current = head;
        while (current) {
            oss << current->data;
            if (current->next) oss << " -> ";
            current = current->next;
        }
        return oss.str();
    }

    void print() const {
        std::cout << traverse_forward() << std::endl;
    }
};
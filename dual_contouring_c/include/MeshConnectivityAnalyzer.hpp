#ifndef __MESHCONNECTIVITYANALYZER_H__
#define __MESHCONNECTIVITYANALYZER_H__
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <memory>
#include "UnionFind.h"
class MeshConnectivityAnalyzer {
public:
    std::vector<std::vector<int>> faces;
    std::vector<std::pair<int,int>> alone_edges_list;

    std::unordered_set<int> all_vertices;

    std::unordered_map<int,std::vector<int>> vertex_to_faces;
    std::unordered_map<int,std::vector<int>> vertex_to_alone_edges;

    std::unique_ptr<UnionFindWithSize> uf;
    std::vector<bool> face_marks;
    std::vector<bool> edge_marks;
    int root;

    MeshConnectivityAnalyzer(const std::vector<std::vector<int>>& f,
                             const std::vector<std::pair<int,int>>& alone_edges = {})
        : faces(f), alone_edges_list(alone_edges)
    {
        for (auto& face : faces) {
            all_vertices.insert(face.begin(), face.end());
        }
        for (auto& e : alone_edges_list) {
            all_vertices.insert(e.first);
            all_vertices.insert(e.second);
        }

        build_vertex_to_faces();
        build_vertex_to_alone_edges();
    }

    void build_vertex_to_faces() {
        for (int i = 0; i < faces.size(); ++i) {
            auto& f = faces[i];
            for(int vid:f){
                vertex_to_faces[vid].push_back(i);
            }
        }
    }

    void build_vertex_to_alone_edges() {
        for (int i = 0; i < alone_edges_list.size(); ++i) {
            auto& e = alone_edges_list[i];
            vertex_to_alone_edges[e.first].push_back(i);
            vertex_to_alone_edges[e.second].push_back(i);
        }
    }

    void analyze_connectivity() {
        if (uf) return;
        int n_faces = faces.size();
        int n_edges = alone_edges_list.size();
        uf = std::make_unique<UnionFindWithSize>(n_faces + n_edges);

        // connect faces by vertices
        for (auto& [v, f_ids] : vertex_to_faces) {
            if (f_ids.size() < 2) continue;
            int first = f_ids[0];
            for (int j = 1; j < f_ids.size(); ++j){
                uf->unite(first, f_ids[j]);
            } 
        }

        // connect alone edges
        for (auto& [v, e_ids] : vertex_to_alone_edges) {
            if (e_ids.size() < 2) {
                continue;
            }
            int first = n_faces + e_ids[0];
            for (int j = 1; j < e_ids.size(); ++j) {
                uf->unite(first, n_faces + e_ids[j]);
            }
        }

        // connect faces and edges sharing vertices
        for (int v : all_vertices) {
            if (vertex_to_faces.count(v) && vertex_to_alone_edges.count(v)) {
                int f_idx = vertex_to_faces[v][0];
                int e_idx = n_faces + vertex_to_alone_edges[v][0];
                uf->unite(f_idx, e_idx);
            }
        }

        auto comps = uf->getComponents();
        if (!comps.empty()) {
            auto max_elem = std::max_element(comps.begin(), comps.end(),
                                [](auto& a, auto& b){ return a.second.size() < b.second.size(); });
            root = max_elem->first;
        }

        generate_marks();
    }

    void generate_marks() {
        int n_faces = faces.size();
        int n_edges = alone_edges_list.size();
        face_marks.resize(n_faces, false);
        edge_marks.resize(n_edges, false);

        for (int i = 0; i < n_faces; ++i) {
            face_marks[i] = (uf->find(i) == root);
        }
        for (int i = 0; i < n_edges; ++i) {
            edge_marks[i] = (uf->find(n_faces + i) == root);
        }
    }

    std::tuple<std::vector<std::vector<int>>, std::vector<std::pair<int,int>>,std::vector<bool>,std::vector<bool>> filter_mesh(bool keep=true) {
        if (keep) {
            if (!uf) analyze_connectivity();
            std::vector<std::vector<int>> f;
            std::vector<std::pair<int,int>> e;
            for (int i = 0; i < faces.size(); ++i){
                if (face_marks[i]) {
                    f.push_back(faces[i]);
                }
            } 
            for (int i = 0; i < alone_edges_list.size(); ++i) {
                if (edge_marks[i]) {
                    e.push_back(alone_edges_list[i]);
                }
            }
            return {f,e,face_marks,edge_marks};
        } else {
            int n_faces = faces.size();
            int n_edges = alone_edges_list.size();
            face_marks.resize(n_faces, true);
            edge_marks.resize(n_edges, true);
            return {faces, alone_edges_list,face_marks,edge_marks};
        }
    }
};


inline std::tuple<std::vector<std::vector<int>>, std::vector<std::pair<int,int>>,std::vector<bool>,std::vector<bool>> remove_redundant_patch(
    const std::vector<std::vector<int>>& faces,
    const std::vector<std::pair<int,int>>& alone_edges)
{
    MeshConnectivityAnalyzer analyzer(faces, alone_edges);
    return analyzer.filter_mesh(true);
}

#endif // __MESHCONNECTIVITYANALYZER_H__
import numpy as np
from typing import List, Tuple, Set, Dict, Union, Optional


class MeshConnectivityAnalyzer:
    def __init__(self, faces, alone_edges_list=None):
        self.faces = [tuple(face) for face in faces] if isinstance(faces[0], list) else faces
        self.alone_edges_list = alone_edges_list if alone_edges_list is not None else []

        self.all_vertices = set()
        for face in self.faces:
            self.all_vertices.update(face)
        for edge in self.alone_edges_list:
            self.all_vertices.update(edge)

        self.vertex_to_faces = self._build_vertex_to_faces()
        self.vertex_to_alone_edges = self._build_vertex_to_alone_edges()
        
        self.uf = None
        self.face_marks = None
        self.edge_marks = None
        self.largest_root = None
    def _build_vertex_to_faces(self) -> Dict[int, List[int]]:
        vertex_to_faces = {}
        for i, face in enumerate(self.faces):
            for vertex in face:
                if vertex not in vertex_to_faces:
                    vertex_to_faces[vertex] = []
                vertex_to_faces[vertex].append(i)
        return vertex_to_faces
    
    def _build_vertex_to_alone_edges(self) -> Dict[int, List[int]]:
        vertex_to_edges = {}
        for i, edge in enumerate(self.alone_edges_list):
            for vertex in edge:
                if vertex not in vertex_to_edges:
                    vertex_to_edges[vertex] = []
                vertex_to_edges[vertex].append(i)
        return vertex_to_edges
    def _connect_faces(self):

        for vertex, face_indices in self.vertex_to_faces.items():
            if len(face_indices) > 1:
                first_face = face_indices[0]
                for other_face in face_indices[1:]:
                    self.uf.union(first_face, other_face)
    
    def _connect_alone_edges(self):
        n_faces = len(self.faces)

        for vertex, edge_indices in self.vertex_to_alone_edges.items():
            if len(edge_indices) > 1:

                first_edge = n_faces + edge_indices[0]
                for other_edge in edge_indices[1:]:
                    self.uf.union(first_edge, n_faces + other_edge)
    
    def _connect_faces_and_edges(self):
        n_faces = len(self.faces)

        for vertex in self.all_vertices:
            if vertex in self.vertex_to_faces and vertex in self.vertex_to_alone_edges:
                face_idx = self.vertex_to_faces[vertex][0]
                edge_idx = n_faces + self.vertex_to_alone_edges[vertex][0]
                self.uf.union(face_idx, edge_idx)
    def _find_largest_component(self):
        components = self.uf.get_components()
        
        if not components:
            return None, 0
        largest_root = max(components.items(), key=lambda x: len(x[1]))[0]
        max_size = len(components[largest_root])
        
        return largest_root, max_size
    
    def _generate_marks(self, target_root):
        n_faces = len(self.faces)
        n_edges = len(self.alone_edges_list)
        face_marks = []
        for i in range(n_faces):
            root = self.uf.find(i)
            face_marks.append(root == target_root)
        edge_marks = []
        for i in range(n_edges):
            root = self.uf.find(n_faces + i)
            edge_marks.append(root == target_root)
        
        return face_marks, edge_marks
    def analyze_connectivity(self):
        if self.uf is not None:
            return
        
        n_faces = len(self.faces)
        n_edges = len(self.alone_edges_list)
        total_elements = n_faces + n_edges
        
        self.uf = UnionFindWithSize(total_elements)

        self._connect_faces()
        
        if n_edges > 0:
            self._connect_alone_edges()
            self._connect_faces_and_edges()

        self.largest_root, max_size = self._find_largest_component()

        self.face_marks, self.edge_marks = self._generate_marks(self.largest_root)

    
    def get_max_component_marks(self):
        if self.uf is None:
            self.analyze_connectivity()
        
        return self.face_marks, self.edge_marks
    def get_component_info(self):
        if self.uf is None:
            raise ValueError("please run analyze_connectivity()")
        
        n_faces = len(self.faces)
        n_edges = len(self.alone_edges_list)
        
        components = {}

        for i in range(n_faces):
            root = self.uf.find(i)
            if root not in components:
                components[root] = {"faces": [], "edges": []}
            components[root]["faces"].append(i)

        for i in range(n_edges):
            root = self.uf.find(n_faces + i)
            if root not in components:
                components[root] = {"faces": [], "edges": []}
            components[root]["edges"].append(i)

        sorted_components = sorted(
            components.items(), 
            key=lambda x: len(x[1]["faces"]) + len(x[1]["edges"]), 
            reverse=True
        )
        
        return sorted_components
    def filter_mesh(self, keep=True):
        if keep:
            if self.uf is None:
                self.analyze_connectivity()

            filtered_faces = []
            for i, mark in enumerate(self.face_marks):
                if mark:
                    filtered_faces.append(self.faces[i])

            edge_list = list(self.alone_edges_list)
            filtered_edges = set()
            for i, mark in enumerate(self.edge_marks):
                if mark:
                    filtered_edges.add(edge_list[i])
            
            face_marks = self.face_marks
            edge_marks = self.edge_marks
        else:
            filtered_faces = self.faces
            filtered_edges = self.alone_edges_list
            face_marks = [True] * len(self.faces)
            edge_marks = [True] * len(self.alone_edges_list)
        
        return filtered_faces, filtered_edges, np.array(face_marks,dtype=bool), np.array(edge_marks,dtype=bool)

class UnionFindWithSize:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def find_iterative(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != root:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent
        
        return root
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        
        self.count -= 1
        return True
    
    def is_connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_size(self, x):
        root = self.find(x)
        return self.size[root]
    
    def get_components(self):
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components
    


def remove_redundant_patch(faces, alone_edges=None):
    analyzer = MeshConnectivityAnalyzer(faces, alone_edges)
    return analyzer.filter_mesh(keep=True)


def get_max_connected_component(faces, alone_edges=None):

    analyzer = MeshConnectivityAnalyzer(faces, alone_edges)
    analyzer.analyze_connectivity()
    return analyzer.face_marks, analyzer.edge_marks

import numpy as np
from collections import defaultdict, deque

def get_largest_connected_component_simple(faces, isolated_edges=None):
    if isolated_edges is None:
        isolated_edges = set()
    
    adjacency = defaultdict(set)

    for face in faces:
        for i in range(4):
            v1 = face[i]
            v2 = face[(i + 1) % 4]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    for edge in isolated_edges:
        v1, v2 = edge
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)
    
    visited = set()
    components = []
    
    for vertex in adjacency:
        if vertex not in visited:
            component_vertices = set()
            queue = deque([vertex])
            visited.add(vertex)
            
            while queue:
                current = queue.popleft()
                component_vertices.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component_vertices)
    
    if not components:
        return np.array([], dtype=faces.dtype), set()

    largest_component = max(components, key=len)

    faces_largest = []
    keep_face_ids=[]
    for i,face in enumerate(faces):
        if all(v in largest_component for v in face):
            faces_largest.append(face)
            keep_face_ids.append(i)

    edges_largest = set()
    for edge in isolated_edges:
        v1, v2 = edge
        if v1 in largest_component and v2 in largest_component:
            edges_largest.add(edge)
    
    return np.array(faces_largest), edges_largest,keep_face_ids


def get_all_connected_components(faces, isolated_edges=None):

    if isolated_edges is None:
        isolated_edges = set()

    adjacency = defaultdict(set)

    for face in faces:
        for i in range(4):
            v1 = face[i]
            v2 = face[(i + 1) % 4]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)

    for edge in isolated_edges:
        v1, v2 = edge
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)

    visited = set()
    components = []
    
    for vertex in adjacency:
        if vertex not in visited:
            component_vertices = set()
            queue = deque([vertex])
            visited.add(vertex)
            
            while queue:
                current = queue.popleft()
                component_vertices.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component_vertices)
    
    component_info = []
    for i, component_vertices in enumerate(components):
        component_faces = []
        for face in faces:
            if all(v in component_vertices for v in face):
                component_faces.append(face)
        component_edges = set()
        for edge in isolated_edges:
            v1, v2 = edge
            if v1 in component_vertices and v2 in component_vertices:
                component_edges.add(edge)
        
        component_info.append({
            'id': i,
            'vertices': component_vertices,
            'vertex_count': len(component_vertices),
            'faces': np.array(component_faces) if component_faces else np.array([], dtype=faces.dtype),
            'face_count': len(component_faces),
            'edges': component_edges,
            'edge_count': len(component_edges),
            'total_size': len(component_faces) + len(component_edges)
        })
    
    component_info.sort(key=lambda x: x['vertex_count'], reverse=True)
    
    return component_info


def filter_mesh_components(faces, isolated_edges=None, min_vertices=1, keep_largest_only=False):
    components = get_all_connected_components(faces, isolated_edges)
    
    if keep_largest_only:
        if components:
            largest = components[0]
            return largest['faces'], largest['edges']
        else:
            return np.array([], dtype=faces.dtype), set()
    else:
        filtered_faces_list = []
        filtered_edges = set()
        
        for comp in components:
            if comp['vertex_count'] >= min_vertices:
                if len(comp['faces']) > 0:
                    filtered_faces_list.extend(comp['faces'])

                filtered_edges.update(comp['edges'])
        
        if filtered_faces_list:
            filtered_faces = np.array(filtered_faces_list)
        else:
            filtered_faces = np.array([], dtype=faces.dtype)
        
        return filtered_faces, filtered_edges


def reindex_mesh(faces, edges=None):
    used_vertices = set()
    
    for face in faces:
        used_vertices.update(face)
    
    if edges is not None:
        for edge in edges:
            used_vertices.add(edge[0])
            used_vertices.add(edge[1])
    
    vertex_list = sorted(used_vertices)
    vertex_map = {old: new for new, old in enumerate(vertex_list)}
    
    faces_new = np.vectorize(vertex_map.get)(faces)
    
    edges_new = None
    if edges is not None:
        edges_new = set()
        for edge in edges:
            new_edge = (vertex_map[edge[0]], vertex_map[edge[1]])
            edges_new.add(new_edge)
    
    return vertex_map, faces_new, edges_new


class SimpleMeshProcessor:
   
    def __init__(self, faces, isolated_edges=None):
        self.faces = np.array(faces)
        self.edges = set(isolated_edges) if isolated_edges else set()
        self.components = None
    
    def analyze(self):
        self.components = get_all_connected_components(self.faces, self.edges)
        return self.components
    
    def keep_largest(self):
        if self.components is None:
            self.analyze()
        
        if self.components:
            largest = self.components[0]
            self.faces = largest['faces']
            self.edges = largest['edges']
        
        return self.faces, self.edges
    
    def filter(self, min_vertices=1, keep_largest=False):
        filtered_faces, filtered_edges = filter_mesh_components(
            self.faces, self.edges, min_vertices, keep_largest
        )
        self.faces = filtered_faces
        self.edges = filtered_edges
        return self.faces, self.edges
    
    def reindex(self):
        vertex_map, faces_new, edges_new = reindex_mesh(self.faces, self.edges)
        self.faces = faces_new
        self.edges = edges_new
        return vertex_map, self.faces, self.edges
    
    def print_info(self):
        if self.components is None:
            self.analyze()
        
        print(f"=== Mesh Information ===")
        print(f"Total faces: {len(self.faces)}")
        print(f"Total isolated edges: {len(self.edges)}")
        print(f"Number of components: {len(self.components)}")
        print()
        
        print("Components (sorted by size):")
        for i, comp in enumerate(self.components):
            print(f"  Component {i}:")
            print(f"    Vertices: {comp['vertex_count']}")
            print(f"    Faces: {comp['face_count']}")
            print(f"    Isolated edges: {comp['edge_count']}")
            print(f"    Total size: {comp['total_size']}")

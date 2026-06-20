import numpy as np
from pyigl.unique_edge_map import *
from pyigl.dihedral_angle import *
from typing import List, Tuple, Dict
from enum import Enum

class EdgeType(Enum):
    CONVEX = -1
    SMOOTH = 0
    CONCAVE = 1
    FLAT = 2
class ConerType(Enum):
    OUTER=0
    INNER=1
class CornerPoint:
    def __init__(self, vertex_id: int, sharp_edges: List[int]):
        self.vertex_id = vertex_id
        self.sharp_edges = sharp_edges
        self.type=ConerType.INNER
    
    def __repr__(self):
        return f"CornerPoint(vertex={self.vertex_id}, edges={self.sharp_edges})"


class SharpFeatureDetector:
    def __init__(self, V: np.ndarray, F: np.ndarray, angle_threshold_degrees: float = 95.0):
        self.V = V
        self.F = F
        self.angle_threshold_degrees=angle_threshold_degrees
        self.angle_threshold = np.deg2rad(angle_threshold_degrees)

        self.N = None
        self.E = None
        self.uE = None
        self.EMAP = None
        self.uE2E = None
        
    def precompute(self) -> bool:
        try: 
            #len(self.F)=m len(self.E)=3 * m
            self.E, self.uE, self.EMAP, self.uE2E = unique_edge_map(self.F,return_uE2E=True)
            
            return True
        except Exception as e:
            print(f"precompute error: {e}")
            return False
    
    def detect_sharp_edges(self) -> Tuple[List[EdgeType], Dict[int, List[int]]]:

        if not self.precompute():
            raise RuntimeError("precompute fail!")
        
        edge_types = [EdgeType.SMOOTH] * self.uE.shape[0]
        vertex_to_edges = {i: [] for i in range(self.V.shape[0])}
        min_dihedral_angle_degree=180.0
        max_shape_edge=None
        for ue_idx in range(self.uE.shape[0]):
            edges = self.uE2E[ue_idx]
            ue=self.uE[ue_idx]
            v0, v1 = ue
            vertex_to_edges[v0].append(ue_idx)
            vertex_to_edges[v1].append(ue_idx)

            if len(edges) > 2 or len(edges)==0:
                continue
            if len(edges)==1:
                # edge_type = EdgeType.CONVEX
                # edge_types[ue_idx] = edge_type
                pass
            else:
                #triangle 1:i0 i1 i2
                #triangle 2:i0 i1 i3
                #F.shape[0] = m

                e1, e2 = edges
                f1 = e1 % self.F.shape[0]
                f2 = e2 % self.F.shape[0]                
                i0=-1
                i1=-1
                i2=-1
                i3=-1
                idx0=-1
                idx1=-1
                for i in range(3):
                    if i2==-1:
                        if self.F[f1,i]!=ue[0] and self.F[f1,i]!=ue[1]:
                            i2=self.F[f1,i]
                    if i3==-1:
                        if self.F[f2,i]!=ue[0] and self.F[f2,i]!=ue[1]:
                            i3=self.F[f2,i]
                    if i0==-1:
                        if self.F[f1,i]==ue[0] or self.F[f1,i]==ue[1]:
                            idx0=i
                            i0=self.F[f1,i]
                    elif i1==-1:
                        if self.F[f1,i]==ue[0] or self.F[f1,i]==ue[1]:
                            idx1=i
                            i1=self.F[f1,i]
                #ensure that the order of the two vertices of the shared edge (i0, i1) is consistent with the orientation of the first triangle (F[f1]).
                if (idx1 - idx0) % 3 == 1:
                    pass
                else:
                    i0,i1=i1,i0
                
                dihedral_angle_degree=compute_approximate_dihedral_angle(self.V[i0],self.V[i1],self.V[i2],self.V[i3])    
                
                type=orientation_3(self.V[i0],self.V[i1],self.V[i2],self.V[i3])
                
                if abs(dihedral_angle_degree) < self.angle_threshold_degrees:
                    edge_type = EdgeType(type)
                    edge_types[ue_idx] = edge_type
                    if min_dihedral_angle_degree>abs(dihedral_angle_degree):
                        min_dihedral_angle_degree=abs(dihedral_angle_degree)
                        max_shape_edge=ue
                    
                else:
                    if dihedral_angle_degree==180:
                        edge_type=EdgeType.FLAT
        
        return edge_types, vertex_to_edges,min_dihedral_angle_degree,max_shape_edge
    
    def detect_corner_points(self, edge_types: List[EdgeType], 
                           vertex_to_edges: Dict[int, List[int]]) -> List[CornerPoint]:
        corner_points = []
        for vertex_id, edge_indices in vertex_to_edges.items():
            # if vertex_id==116:
            #     print(vertex_id)
            if len(edge_indices) < 2:  
                continue
            sharp_edges = []
            
            for edge_idx in edge_indices:
                edge_type = edge_types[edge_idx]
                if edge_type in [EdgeType.CONVEX, EdgeType.CONCAVE]:
                    sharp_edges.append(edge_idx)
  
            
            if len(sharp_edges)>=3:
                corner=CornerPoint(vertex_id, sharp_edges)
                
                skip=False
                for edge_idx in sharp_edges:
                    edge_type = edge_types[edge_idx]
                    if edge_type==EdgeType.CONCAVE:
                        skip=True
                        break
                if skip:
                    corner.type=ConerType.INNER
                else:
                    corner.type=ConerType.OUTER
                corner_points.append(corner)        
        return corner_points
   
    
    def get_statistics(self, edge_types: List[EdgeType], corner_points: List[CornerPoint]) -> Dict:
        stats = {
            'total_edges': len(edge_types),
            'smooth_edges': sum(1 for et in edge_types if et == EdgeType.SMOOTH),
            'convex_edges': sum(1 for et in edge_types if et == EdgeType.CONVEX),
            'concave_edges': sum(1 for et in edge_types if et == EdgeType.CONCAVE),
            'flat_edges': sum(1 for et in edge_types if et == EdgeType.FLAT),
            'corner_points': len(corner_points),
            'sharp_edges_per_corner': [len(cp.sharp_edges) for cp in corner_points]
        }
        return stats
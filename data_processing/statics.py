from pathlib import Path
import os
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_vertex_index(tok: str, nverts: int) -> int:
    s = tok.split('/')[0]
    idx = int(s)
    if idx > 0:
        return idx - 1
    else:
        return nverts + idx


def count_components(face_vertex_lists, line_edge_list):
    n_faces = len(face_vertex_lists)
    n_lines = len(line_edge_list)
    total = n_faces + n_lines

    if total == 0:
        return 0, 0, 0

    uf = UnionFind(total)

    vertex_to_face_ids = {}
    for fid, verts in enumerate(face_vertex_lists):
        for v in verts:
            vertex_to_face_ids.setdefault(v, []).append(fid)

    vertex_to_line_ids = {}
    for eid, edge in enumerate(line_edge_list):
        for v in edge:
            vertex_to_line_ids.setdefault(v, []).append(eid)

    for _, ids in vertex_to_face_ids.items():
        if len(ids) > 1:
            base = ids[0]
            for other in ids[1:]:
                uf.union(base, other)

    for _, ids in vertex_to_line_ids.items():
        if len(ids) > 1:
            base = n_faces + ids[0]
            for other in ids[1:]:
                uf.union(base, n_faces + other)

    all_vertices = set(vertex_to_face_ids.keys()) | set(vertex_to_line_ids.keys())
    for v in all_vertices:
        fids = vertex_to_face_ids.get(v, [])
        eids = vertex_to_line_ids.get(v, [])
        if fids and eids:
            uf.union(fids[0], n_faces + eids[0])

    roots = {}
    face_component_ids = set()
    edge_component_ids = set()

    for i in range(total):
        r = uf.find(i)
        roots.setdefault(r, []).append(i)
        if i < n_faces:
            face_component_ids.add(r)
        else:
            edge_component_ids.add(r)

    return len(roots), len(face_component_ids), len(edge_component_ids)


def parse_obj_topology(obj_path: Path, use_all_vertices: bool = False):
    num_obj_vertices = 0
    used_vertices = set()
    face_edges = set()
    line_edges = set()
    num_faces = 0
    num_face_records = 0
    num_line_records = 0

    face_vertex_lists = []
    line_edge_list = []

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('v '):
            num_obj_vertices += 1

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('f '):
            num_face_records += 1
            parts = line.split()[1:]
            verts = [parse_vertex_index(p, num_obj_vertices) for p in parts]
            if len(verts) < 3:
                continue

            num_faces += 1
            face_vertex_lists.append(tuple(verts))
            used_vertices.update(verts)

            m = len(verts)
            for i in range(m):
                a = verts[i]
                b = verts[(i + 1) % m]
                if a != b:
                    face_edges.add(tuple(sorted((a, b))))

        elif line.startswith('l '):
            num_line_records += 1
            parts = line.split()[1:]
            verts = [parse_vertex_index(p, num_obj_vertices) for p in parts]
            if len(verts) < 2:
                continue

            used_vertices.update(verts)

            for i in range(len(verts) - 1):
                a = verts[i]
                b = verts[i + 1]
                if a != b:
                    e = tuple(sorted((a, b)))
                    line_edges.add(e)
                    line_edge_list.append(e)

    line_edge_list = sorted(set(line_edge_list))

    all_edges = face_edges | line_edges

    V = num_obj_vertices if use_all_vertices else len(used_vertices)
    E = len(all_edges)
    F = num_faces
    chi = V - E + F

    num_components, num_face_components, num_edge_components = count_components(
        face_vertex_lists, line_edge_list
    )

    return {
        "file": obj_path.name,
        "path": str(obj_path),
        "V": V,
        "E": E,
        "F": F,
        "chi": chi,
        "num_components": num_components,
        "num_face_components": num_face_components,
        "num_edge_components": num_edge_components,
        "num_obj_vertices": num_obj_vertices,
        "num_used_vertices": len(used_vertices),
        "num_unused_vertices": num_obj_vertices - len(used_vertices),
        # "num_face_edges": len(face_edges),
        # "num_line_edges": len(line_edges),
        # "num_total_unique_edges": len(all_edges),
        # "num_face_records": num_face_records,
        # "num_line_records": num_line_records,
    }


if __name__ == "__main__":
    #INPUTDIR="/mnt/WindowsH/octree_group/abc_test_2048_100"
    INPUTDIR="I:/octree_patch_cpp/noncad"
    dataset="bug"
    # dataset="0012280"
    #dataset="01_00040144_cdff4f5642a84a6798b49ad6_trimesh_004"
    
    obj_file =os.path.join(INPUTDIR,dataset+"/"+dataset+".depth7.tri.sqef.obj")
    result = parse_obj_topology(Path(obj_file), use_all_vertices=False)
    for k, v in result.items():
        print(f"{k}: {v}")
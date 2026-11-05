# DC-SQEM source code

> DC-SQEM: Medial Mesh Extraction by Dual Contouring with Spherical
> Quadric Error Metric
**Project page:** [https://dc-sqem.github.io/](https://dc-sqem.github.io/)

## System Requirements

- Supported operating systems:
  - **Ubuntu (Linux)** – Tested on Ubuntu 24.04
  - **Windows** – Tested on Windows 11 with Visual Studio 2019

- ### Hardware requirements
  - **Memory (RAM)**: **64 GB or more recommended**. Actual runtime memory usage may exceed 40+ GB.

- Python 3.10+
- CMake 3.10+
- C++ compiler:
  - **Ubuntu**: GCC 13.0 or higher (must support C++14/17)
  - **Windows**: Visual Studio 2019 (MSVC v142)
  - Other operating systems (e.g., macOS) are currently **not supported**

## Installation and Compilation

### 1. Create a Python virtual environment (recommended)

```bash
# Using conda
conda create -n dc_sqem python=3.10
conda activate dc_sqem
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install pybind11   # required for compiling C++ bindings
```

> `requirements.txt` is located in the project root directory and contains all runtime Python packages.

### 3. Compile the C++ library using CMake

CMake will automatically include the corresponding configuration file (`CMakeLists_ubuntu.txt` or `CMakeLists_windows.txt`) based on the operating system.

Navigate to the `dual_contouring_c` folder under the project directory. (where `CMakeLists.txt` is located) and run:

```bash
mkdir build && cd build
cmake .. --fresh -DCMAKE_BUILD_TYPE=Release
```

Then choose the compilation command according to your OS:

- **Ubuntu (using GCC 14.0)**:
  ```bash
  make -j4
  ```
- **Windows (using Visual Studio 2019)**:
  ```bash
  cmake --build . --config Release
  ```
  You can also open the generated `graph_grouping_c.sln` solution file in Visual Studio 2019.

After successful compilation, the generated library file will be located at:

- **Ubuntu**: `build/GRAPH_GROUPING.so`
- **Windows**: `build/Release/GRAPH_GROUPING.pyd`

> Under Windows, pybind11 outputs a `.pyd` file, which is essentially a directly importable Python module.

### 4. Move the library file to a directory importable by Python

Copy the compiled library file to `cpp_module` directory so it can be imported.

- **Ubuntu**:
  ```bash
  cp build/GRAPH_GROUPING.so ./cpp_module/
  ```
- **Windows** (PowerShell or cmd):
  ```cmd
  copy build/Release/GRAPH_GROUPING.pyd ./cpp_module/
  ```

## Usage Steps

### Data Preparation

#### Data Normalization

Modify `INPUT_DIR` (input directory) and `OUTPUT_DIR` (output directory) in `scale_mesh.py`, then execute:

```bash
python scale_mesh.py
```

The data in `datasets/scaled` is already normalized; no need to normalize again.

#### Data Sampling

Copy the `datasets` folder to the project root directory.

- **Sampling CAD data**

Example:

```bash
python run_mesh_sampling.py --dataset 01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000 \
    --datasets_dir ./datasets/scaled/ABC \
    --output_dir ./cloud/ABC/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000 \
    --skip_feature 0
```

`01_00040051_b95bd5d02f4d435098659bb3_trimesh_000` is the filename without extension.  
`./datasets/scaled/ABC` is the directory containing mesh files.  
`./cloud/ABC` is the output directory for sampling files.  

After execution, a file named `01_00040051_b95bd5d02f4d435098659bb3_trimesh_000.sampling.npz` will be generated in the output directory.

- **Sampling organic shape data**

Example:

```bash
python run_mesh_sampling.py --dataset bear \
    --datasets_dir ./datasets/scaled/noncad \
    --output_dir ./cloud/noncad/bear \
    --skip_feature 1
```

The flag `--skip_feature 1` means that feature points are not detected.

### Running Examples

```bash
python run_mat_pcl_sqef.py \
    --dataset 01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000 \
    --input_dir ./cloud/ABC/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000 \
    --max_depth 7 \
    --output_dir ./octree/ABC/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000
```

The result will be saved as `./octree/ABC/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000.depth7.tri.sqef.obj` and `./octree/ABC/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000/01_00040053_fdbd1ff80ef645bca6474bad_trimesh_000.depth7.sqef.ma`.



```bash
python run_mat_pcl_sqef.py \
    --dataset bear \
    --input_dir ./cloud/noncad/bear \
    --max_depth 7 \
    --output_dir ./octree/noncad/bear
```

The result will be saved as `./octree/noncad/bear/bear.depth7.tri.sqef.obj` and `./octree/noncad/bear/bear.depth7.sqef.ma`.



### Output Description

`bear.depth7.tri.sqef.obj` – The generated medial axis OBJ triangular mesh may contain a mix of `f` and `l` faces; use Blender for visualization.


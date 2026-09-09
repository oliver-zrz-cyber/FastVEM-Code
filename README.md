# Fast VEM Fluid Simulation

This is the official code repository for [**Fast VEM Fluid Simulation**](https://dl.acm.org/doi/10.1145/3811315) by Runze Zhang and Bo Ren, published in ACM Transactions on Graphics (2026). The paper is also available on [arXiv](https://arxiv.org/abs/2607.17725).

**This is a very preliminary release.** Much of the code, dependency setup, examples, and documentation still needs to be cleaned up. I am working on this and will do my best to provide a more organized version before **October 7, 2026**. Thank you for your patience!

## Basic usage

The current setup targets Linux with a CUDA-capable NVIDIA GPU. Some dependency and data paths are still specific to the development machine and need to be edited before use.

### 1. Prepare dependencies

The project uses CMake 3.18+, C++17, CUDA, OpenCL, Eigen3, CGAL, TBB, VTK, OpenVDB, GLFW/OpenGL, matio, and Python 3 with development headers and NumPy. OpenMP is used when available.

The build also expects AMGCL in `external/amgcl` and GLAD in `external/glad` (including `src/glad.c` and its headers). The `external/` directory is currently ignored by Git, so these dependencies must be supplied separately; dependency packaging is still in progress.

Edit [CMakeLists.txt](CMakeLists.txt) to match your local CUDA/OpenCL, CGAL, and OpenVDB installation paths. The current configuration references CUDA 12.1 and CGAL 5.4.5 and sets `CMAKE_CUDA_ARCHITECTURES` to `86`; adjust these settings for your environment and GPU.

### 2. Build

From the repository root:

```bash
cmake -S . -B cmake-build-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cmake-build-relwithdebinfo -j2
```

The executable is `cmake-build-relwithdebinfo/VEMFLUID`. Double precision is the default; add `-DVEMFLUID_USE_FLOAT=ON` to the configuration command to select single precision.

### 3. Configure and run

The default [run_vemfluid.sh](run_vemfluid.sh) uses `models/ball.obj` with GMG at resolution 64 and writes results to `outputs/ball-gmg-64/`. All paths are resolved from the script location, so you can launch it from any directory.

The script configures CMake and builds `VEMFLUID` before starting the simulation.

```bash
bash run_vemfluid.sh --solver gmg --resolution 64
bash run_vemfluid.sh --solver amg --resolution 64
```

`--solver` accepts `gmg` or `amg`. `--resolution` sets the finest grid resolution. GMG requires `8, 16, 32, 64, ...`, because its hierarchy begins at 8 and doubles at each level; AMG accepts any positive integer. The script creates a separate `outputs/ball-<solver>-<resolution>/` directory. Keep all ten positional arguments; the current entry point still uses the original argument layout:

Both paths use AMGCL's restarted right-preconditioned GMRES outer solver with a relative tolerance of `1e-3`, a maximum of `10000` iterations, and restart length `30`. The `amg` path uses AMGCL's algebraic multigrid preconditioner; the `gmg` path uses this project's geometric multigrid V-cycle. AMG uses the same Chebyshev smoother parameters as GMG (`degree=3`, `power_iters=200`, `higher=2`, `lower=1/120`, and diagonal scaling); its other AMG settings use AMGCL defaults.

| Position | Meaning in the current demo |
| --- | --- |
| 1 | Legacy argument; currently unused. |
| 2 | Filename whose stem is used as the output name, e.g. `demo.vtu` gives `demo`. This file is not loaded by the current entry point. |
| 3–5 | Smoke source center: x, y, z. |
| 6 | Smoke source radius. |
| 7 | Output directory. Create it first and include a trailing `/`. |
| 8 | CFL mode; keep this at `1` for the current demo. |
| 9 | Legacy AABB path; passed through but currently unused in the active simulation routine. |
| 10 | Input boundary mesh, e.g. an OBJ file. Use a triangulated mesh within the unit simulation domain `[0, 1]^3`. |

Define custom paths relative to `ROOT_DIR`, as in the script. It creates the output directory and runs there so generated meshes and profiling data stay with the tracer output. The profiling CSV is saved as `function_memory_profile.csv`; the script does not require Python plotting tools.

The current [src/main.cpp](src/main.cpp) dispatches `gmg` to the geometric-multigrid path and `amg` to the single-grid AMGCL path. Simulation logic is in [src/sim3d.cpp](src/sim3d.cpp), and serial BSP mesh construction is in [src/mini_lcc.cpp](src/mini_lcc.cpp).

Smoke tracer frames are written as `<output_name>_<frame>.bin` in the configured output directory. Example assets and a simpler configuration interface are still being organized.

cmake_minimum_required(VERSION 3.18)

project(VEMFLUID LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

option(VEMFLUID_ENABLE_TIME_REPORT "Enable compiler time reports (-ftime-report)." OFF)
option(VEMFLUID_ENABLE_PCH "Enable precompiled headers for C++ sources." ON)
option(VEMFLUID_ENABLE_CCACHE "Use ccache when available." ON)
option(VEMFLUID_ENABLE_UNITY_BUILD "Enable unity build for small C++ translation units." OFF)
option(VEMFLUID_ENABLE_CUDA_SEPARABLE_COMPILATION "Enable CUDA separable compilation / device link." OFF)
option(VEMFLUID_USE_FLOAT "Use float as the main simulation scalar type instead of double." OFF)

if(VEMFLUID_ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "✅ Using ccache: ${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

set(OpenCL_LIBRARY
    /usr/local/cuda-12.1/lib64/libOpenCL.so
    CACHE FILEPATH "Force CUDA OpenCL" FORCE
)
set(OpenCL_INCLUDE_DIR
    /usr/local/cuda-12.1/include
    CACHE PATH "Force CUDA OpenCL headers" FORCE
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP found, enabling AMGCL multithreading")
    add_definitions(-DAMGCL_HAVE_OPENMP)
    set(AMGCL_HAVE_OPENMP ON CACHE BOOL "Enable OpenMP backend in AMGCL")
else()
    message(WARNING "⚠️ OpenMP not found, fallback to builtin backend")
endif()

add_subdirectory(external/amgcl)

if(VEMFLUID_ENABLE_TIME_REPORT)
    add_compile_options(-ftime-report)
endif()

set(VEMFLUID_CPP_SOURCES
    src/AbSolver.cpp
    src/mini_lcc.cpp
    src/Render3d.cpp
    src/Visualization.cpp
    src/amgcl_smoother.cpp
    src/bsp_mesh.cpp
    src/bsp_naive.cpp
    src/convergence.cpp
    src/dfn.cpp
    src/func.cpp
    src/liquidSim3d.cpp
    src/mac.cpp
    src/mac3d.cpp
    src/main.cpp
    src/mgsolver_mac.cpp
    src/mgsolver_vemg.cpp
    src/mgsolver_vemp.cpp
    src/sim.cpp
    src/sim3d.cpp
    src/stb_imp.cpp
    src/tinyxml2.cpp
    src/vem2d.cpp
    src/vem3d.cpp
    src/vem3d_particles.cpp
    src/vex_gmres.cpp
    src/vex_pcg.cpp
)

set(VEMFLUID_CUDA_SOURCES
    src/PCG.cu
    src/mgsolver_gpu.cu
        src/vemliquid.cpp
)

add_executable(VEMFLUID
    ${VEMFLUID_CPP_SOURCES}
    ${VEMFLUID_CUDA_SOURCES}
)

set_target_properties(VEMFLUID PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${VEMFLUID_ENABLE_CUDA_SEPARABLE_COMPILATION}
)

if(VEMFLUID_ENABLE_UNITY_BUILD)
    set_target_properties(VEMFLUID PROPERTIES
        UNITY_BUILD ON
        UNITY_BUILD_BATCH_SIZE 8
    )
    set_source_files_properties(
        src/mini_lcc.cpp
        src/liquidSim3d.cpp
        src/sim3d.cpp
        src/vem3d.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
    )
endif()

target_include_directories(VEMFLUID PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/amgcl
)

if(VEMFLUID_USE_FLOAT)
    target_compile_definitions(VEMFLUID PRIVATE VEMFLUID_USE_FLOAT)
endif()

if(VEMFLUID_ENABLE_PCH)
    target_precompile_headers(VEMFLUID PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/src/dfn.h>"
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(VEMFLUID PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
target_link_libraries(VEMFLUID PRIVATE glfw OpenGL::GL)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(VEMFLUID PRIVATE Eigen3::Eigen)

target_link_libraries(VEMFLUID PRIVATE amgcl)

find_path(MATIO_INCLUDE_DIR matio.h PATH_SUFFIXES include)
find_library(MATIO_LIBRARY matio PATH_SUFFIXES lib)
if(MATIO_INCLUDE_DIR AND MATIO_LIBRARY)
    message(STATUS "✅ Found matio: ${MATIO_LIBRARY}")
    target_include_directories(VEMFLUID PRIVATE ${MATIO_INCLUDE_DIR})
    target_link_libraries(VEMFLUID PRIVATE ${MATIO_LIBRARY} z)
else()
    message(FATAL_ERROR "❌ matio not found, please install libmatio-dev")
endif()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
target_include_directories(VEMFLUID PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(VEMFLUID PRIVATE ${Python3_LIBRARIES})

add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)
target_link_libraries(VEMFLUID PRIVATE glad)

find_package(VTK REQUIRED)
if(VTK_FOUND)
    message(STATUS "✅ Found VTK version: ${VTK_VERSION}")
    include(${VTK_USE_FILE})
    target_link_libraries(VEMFLUID PRIVATE ${VTK_LIBRARIES})
else()
    message(FATAL_ERROR "❌ VTK not found! Please install libvtk7-dev or libvtk9-dev")
endif()

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "✅ Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
    target_link_libraries(VEMFLUID PRIVATE CUDA::cudart)
else()
    message(FATAL_ERROR "❌ CUDA Toolkit not found!")
endif()

find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    set(OpenCL_INCLUDE_DIR /usr/local/cuda-12.1/include)
    set(OpenCL_LIBRARY /usr/local/cuda-12.1/targets/x86_64-linux/lib/libOpenCL.so)

    message(STATUS "✅ OpenCL found:")
    message(STATUS "   OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "   OpenCL_LIBRARIES    = ${OpenCL_LIBRARIES}")

    target_include_directories(VEMFLUID PRIVATE ${OpenCL_INCLUDE_DIR})
    target_link_libraries(VEMFLUID PRIVATE ${OpenCL_LIBRARY})
    target_compile_definitions(VEMFLUID PRIVATE AMGCL_HAVE_VEXCL CL_HPP_TARGET_OPENCL_VERSION=300)
else()
    message(FATAL_ERROR "❌ OpenCL not found! Please install OpenCL ICD loader.")
endif()

find_package(TBB REQUIRED)
if(TBB_FOUND)
    target_link_libraries(VEMFLUID PRIVATE TBB::tbb)
endif()

set(CGAL_DIR "/home/lhw/桌面/MyLibs/CGAL-5.4.5/")
find_package(CGAL CONFIG REQUIRED PATHS ${CGAL_DIR})
target_link_libraries(VEMFLUID PRIVATE CGAL::CGAL)

set(OpenVDB_INCLUDE_DIR "/usr/local/include/openvdb")
set(OpenVDB_LIBRARIES "/usr/local/lib/libopenvdb.so")
target_include_directories(VEMFLUID PRIVATE ${OpenVDB_INCLUDE_DIR})
target_link_libraries(VEMFLUID PRIVATE ${OpenVDB_LIBRARIES})

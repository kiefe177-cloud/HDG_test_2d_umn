cmake_minimum_required(VERSION 3.10)
project(MatlabToCppPort)

# --- 1. System Configuration & Policies ---
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options(-Wno-stringop-overflow)   # suppress false positive from Eigen sparse

enable_testing()

# --- 2. Output Directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output")
file(MAKE_DIRECTORY ${OUTPUT_DIR})
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Per-target output directories
set(OUTPUT_DIR_DRIVER     "${OUTPUT_DIR}/driver")
set(OUTPUT_DIR_CYL_DRIVER "${OUTPUT_DIR}/cyl_driver")
set(OUTPUT_DIR_TEST       "${OUTPUT_DIR}/tests")
file(MAKE_DIRECTORY ${OUTPUT_DIR_DRIVER})
file(MAKE_DIRECTORY ${OUTPUT_DIR_CYL_DRIVER})
file(MAKE_DIRECTORY ${OUTPUT_DIR_TEST})

# --- 3. HDF5 Discovery ---
set(HDF5_DIR "C:/Program Files/HDF_Group/HDF5/1.14.6/cmake")
find_package(HDF5 REQUIRED COMPONENTS CXX CONFIG)

# --- 4. HighFive & Eigen Integration ---
add_definitions(-DHIGHFIVE_USE_EIGEN)

# --- 5. The Library (cfd_core) ---
set(SOURCE_FILES 
    src/amr_connect.cpp
    src/amr_dump.cpp
    src/amr_flatten.cpp
    src/amr_number.cpp
    src/amr_refine.cpp
    src/base_grid_metrics.cpp
    src/baseflow_flatplate_1d.cpp
    src/bl_similarity.cpp
    src/calc_geom_iso_tfi.cpp
    src/calc_grads_strong.cpp
    src/comp_space_metrics.cpp
    src/debug_print.cpp
    src/dg_mat2_Roe.cpp
    src/dg_solve.cpp
    src/generate_tfi.cpp
    src/fd_coeffs_uniform.cpp
    src/flux_lin_euler.cpp
    src/flux_lin_viscous.cpp
    src/flux_lin_viscous_face.cpp
    src/flux_split.cpp
    src/GLDeriv.cpp
    src/GLQuad.cpp
    src/get_tau_visc.cpp
    src/interp1.cpp
    src/jacobiP.cpp
    src/jacobiZerosGW.cpp
    src/legendreP.cpp
    src/meshgrid.cpp
    src/msh_amr_curv.cpp
    src/msh_connect.cpp
    src/msh_test_grad.cpp
    src/multidof_ops.cpp
    src/ndgrid.cpp
    src/plot_comp_tfi_grid.cpp
    src/read_cone_dat_file.cpp
    src/save_grid_diagnostics.cpp
    src/sd.cpp
    src/sem1.cpp
    src/sem2.cpp
    src/source_terms_Q.cpp
    src/Spline.cpp
    src/sparse_utils.cpp
    src/tfi_elliptic_full_orthogonal.cpp
    src/tfi_elliptic_orthogonal.cpp
    src/tfi_elliptic_solve.cpp
    src/tfi_elliptic_steger_sorenson_v2.cpp
    src/vec.cpp
)

add_library(cfd_core ${SOURCE_FILES})
target_link_libraries(cfd_core PUBLIC hdf5::hdf5-shared)
target_include_directories(cfd_core PUBLIC
    "${CMAKE_SOURCE_DIR}/external/eigen-5.0.0"
    "${CMAKE_SOURCE_DIR}/external/HighFive/include"
    "${CMAKE_SOURCE_DIR}/include"
    ${HDF5_INCLUDE_DIRS}
)

# --- 6. The Executables (Drivers) ---
add_executable(driver src/driver.cpp)
target_link_libraries(driver PRIVATE cfd_core)
target_compile_definitions(driver PRIVATE
    OUTPUT_DIR="${OUTPUT_DIR_DRIVER}")

add_executable(cyl_driver src/cylinder_driver.cpp)
target_link_libraries(cyl_driver PRIVATE cfd_core)
target_compile_definitions(cyl_driver PRIVATE
    OUTPUT_DIR="${OUTPUT_DIR_CYL_DRIVER}")

# --- 7. The Tests ---
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE cfd_core)
    target_compile_definitions(${test_name} PRIVATE
        OUTPUT_DIR_TEST="${OUTPUT_DIR_TEST}")
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
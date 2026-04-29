cmake_minimum_required(VERSION 3.18)
project(rref_accelerator)

set(CMAKE_CXX_STANDARD 20)

set(TT_METAL_ROOT "/mnt/nvme10/joseph_ufl/tt-metal")
set(TT_METAL_BUILD "${TT_METAL_ROOT}/build_Release")

include_directories(
    ${TT_METAL_BUILD}/include
    ${TT_METAL_BUILD}/include/tt-metalium
    ${TT_METAL_ROOT}/tt_metal/api
    ${TT_METAL_ROOT}/tt_metal/api/tt-metalium
    ${TT_METAL_ROOT}/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
)

link_directories(${TT_METAL_BUILD}/lib)

add_executable(rref_host rref_host.cpp)

target_link_libraries(rref_host PRIVATE
    tt_metal
    fmt
)

set_target_properties(rref_host PROPERTIES
    BUILD_RPATH "${TT_METAL_BUILD}/lib"
    INSTALL_RPATH "${TT_METAL_BUILD}/lib"
)

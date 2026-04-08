cmake_minimum_required(VERSION 3.18)
project(rref_accelerator)

set(CMAKE_CXX_STANDARD 20)

# Find the tt-metal installation
list(APPEND CMAKE_PREFIX_PATH $ENV{TT_METAL_HOME})
find_package(TTMetal REQUIRED)

add_executable(rref_host rref_host.cpp)

target_link_libraries(rref_host PRIVATE
    TTMetal::tt_metal
    TTMetal::tt_metal_distributed
)

# Kernel files are loaded at runtime by path — just ensure they're present
# relative to the working directory when you run the binary.
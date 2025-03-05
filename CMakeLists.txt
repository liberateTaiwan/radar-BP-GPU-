cmake_minimum_required(VERSION 3.10)
project(radar_cuda CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# 设置 CUDA 路径
set(CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/v11.8")
set(CUDA_INCLUDE_DIRS "${CUDA_ROOT}/include")
set(CUDA_LIB_DIR "${CUDA_ROOT}/lib/x64")

# 设置 Eigen 路径
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
)

# 链接目录
link_directories(
    ${CUDA_LIB_DIR}
)

# 将源文件重命名为 .cu
set_source_files_properties(CUDA.cpp PROPERTIES LANGUAGE CUDA)

# 添加 CUDA 编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    --expt-relaxed-constexpr
    -gencode arch=compute_75,code=sm_75)

# 添加可执行文件
add_executable(radar_cuda CUDA.cpp)

# 设置 CUDA 架构
set_target_properties(radar_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75)

# 链接 CUDA 库
target_link_libraries(radar_cuda 
    cudart
    cufft
)
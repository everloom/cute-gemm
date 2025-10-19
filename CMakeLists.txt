# 执行命令 cmake -DCMAKE_BUILD_TYPE=Debug ..


cmake_minimum_required(VERSION 3.18)
project(cute_gemm LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 86)

# 设置构建类型，默认为 Debug 以支持调试
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 添加 CUTLASS include 目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/cutlass/include)

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart shared --cudadevrt none")

# Debug 模式下添加调试符号（支持 cuda-gdb）
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Release 模式优化
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# 查找 CUDA 库
find_package(CUDAToolkit REQUIRED)

# 获取所有 .cu 文件
file(GLOB CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# 排除没有 main 函数的文件（只包含 kernel 定义的文件）
list(FILTER CU_FILES EXCLUDE REGEX ".*vector_add\\.cu$")

# 为每个 .cu 文件创建可执行文件
foreach(cu_file ${CU_FILES})
    # 获取文件名（不带扩展名）
    get_filename_component(exe_name ${cu_file} NAME_WE)
    
    # 创建可执行文件
    add_executable(${exe_name} ${cu_file})
    
    # 链接必要的库
    target_link_libraries(${exe_name}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    
    # 设置目标属性
    set_target_properties(${exe_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()

# 打印配置信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA debug flags: ${CMAKE_CUDA_FLAGS_DEBUG}")


cmake_minimum_required(VERSION 3.29)
project(add CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(add main.cu)

set_target_properties(add PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

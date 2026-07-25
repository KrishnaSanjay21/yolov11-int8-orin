# Build the DFL TensorRT plugin.  # RUN ON DEVICE (Jetson Orin NX, JetPack 6.x)
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Produces libdfl_plugin.so, loaded by trtexec via --plugins or by
# ctypes.CDLL(...) before deserializing an engine that uses the DFL node.
cmake_minimum_required(VERSION 3.18)
project(dfl_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Orin NX is compute capability 8.7 (Ampere). Override with -DCMAKE_CUDA_ARCHITECTURES if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)

# TensorRT lives under /usr/lib/aarch64-linux-gnu on JetPack; override TENSORRT_DIR if custom.
set(TENSORRT_DIR "/usr" CACHE PATH "TensorRT install prefix")
find_path(TRT_INCLUDE NvInfer.h HINTS ${TENSORRT_DIR}/include /usr/include/aarch64-linux-gnu)
find_library(TRT_NVINFER nvinfer HINTS ${TENSORRT_DIR}/lib /usr/lib/aarch64-linux-gnu)

add_library(dfl_plugin SHARED
  DFLPlugin.cpp
  dfl_kernel.cu
)
target_include_directories(dfl_plugin PRIVATE ${TRT_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(dfl_plugin PRIVATE ${TRT_NVINFER} CUDA::cudart)
set_target_properties(dfl_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

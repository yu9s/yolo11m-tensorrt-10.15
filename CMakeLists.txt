cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(yolo11_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -msse4.2 -mavx2 -mfma")
add_compile_options(-fPIC)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
if(NOT DEFINED TensorRT_DIR)
    set(TensorRT_DIR "/usr/local/TensorRT-10.0.1.5")
endif()
message(STATUS "TensorRT_DIR = ${TensorRT_DIR}")
include_directories("${TensorRT_DIR}/include")
link_directories("${TensorRT_DIR}/lib")
link_directories("/usr/local/cuda/lib64")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/lib/cmake/opencv5")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_VERSION = ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
file(GLOB YOLO11_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
add_library(yolo11_sdk SHARED ${YOLO11_SOURCES}) # 生成共享算法库
target_include_directories(yolo11_sdk
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/internal
)
target_link_libraries(yolo11_sdk
    PRIVATE ${CUDA_LIBRARIES}
            nvinfer
            nvinfer_plugin
            cuda
            ${OpenCV_LIBS}
)
file(GLOB YOLO11_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
add_executable(client
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${YOLO11_TEST_SOURCES}
)
target_link_libraries(client
    PRIVATE yolo11_sdk
            ${CUDA_LIBRARIES}
            ${OpenCV_LIBS}
            pthread
)

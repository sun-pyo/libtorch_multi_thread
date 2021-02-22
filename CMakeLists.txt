cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(thpool-test)

set(CMAKE_PREFIX_PATH /home/kmsjames/pytorch/libtorch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/kmsjames/pytorch/pytorch-test/include)
file(GLOB SOURCES "src/*")
#set(SOURCES src/test.cpp src/alex.cpp src/vgg.cpp src/thpool.c) //manual add

include_directories(${COMMON_DIR})

add_executable(thpool-test ${SOURCES})
target_link_libraries(thpool-test "${TORCH_LIBRARIES}")
set_property(TARGET thpool-test PROPERTY CXX_STANDARD 14)

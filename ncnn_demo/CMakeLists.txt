
cmake_minimum_required(VERSION 3.4)
project(demo)
add_definitions(-std=c++17)
set(DCMAKE_BUILD_TYPE Release)
#eigen
find_package(Eigen3  REQUIRED)
#opencv
find_package(OpenCV 4 REQUIRED)
#openmp
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(
    include
    /usr/local/include
    /usr/local/include/ncnn
    ${EIGEN3_INCLUDE_DIR}
    )

aux_source_directory(src SRC)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} libncnn.a ${OpenCV_LIBS})
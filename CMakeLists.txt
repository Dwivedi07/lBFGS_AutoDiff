cmake_minimum_required(VERSION 3.10)
project(LBFGSB_Ceres_Autodiff)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/../ceres-bin")

find_package(Ceres REQUIRED)
set(CMAKE_CXX_STANDARD 17)

# Add paths to header-only dependencies
include_directories(
    ${CMAKE_SOURCE_DIR}/eigen
    ${CMAKE_SOURCE_DIR}/include
)

# Tell CMake where to find Ceres (built in ceres-bin)
set(CERES_DIR "${CMAKE_SOURCE_DIR}/ceres-bin")

# Add each source file as a separate executable
add_executable(quadratic_exec quadratic.cpp)
add_executable(rosenbrock_exec rosenbrock.cpp)
add_executable(ellipse_exec ellipse.cpp)
add_executable(exponential_exec exponential.cpp)

target_link_libraries(quadratic_exec ${CERES_LIBRARIES})
target_link_libraries(rosenbrock_exec ${CERES_LIBRARIES})
target_link_libraries(ellipse_exec ${CERES_LIBRARIES})
target_link_libraries(exponential_exec ${CERES_LIBRARIES})

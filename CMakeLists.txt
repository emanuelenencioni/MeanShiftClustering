project(mean_shift_seq)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17) # Adjust the C++ standard version as needed
set(CMAKE_BUILD_TYPE Debug)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Compiling without OpenMP support.")
endif()
add_executable(${PROJECT_NAME}   
src/main.cpp 
src/STBImage.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE include )

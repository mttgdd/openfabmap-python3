cmake_minimum_required(VERSION 2.6)
project(openfabmap_python3)

# Macro for opencv version
find_package(OpenCV REQUIRED)
if (NOT (OpenCV_VERSION VERSION_LESS "2.4.0"))
    add_definitions(-DOPENCV2P4)
endif ()

# openmp link
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("Found OpenMP")
endif(OPENMP_FOUND)

# python libraries
find_package(PythonInterp REQUIRED)

# numpy
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "import numpy as np\nprint(np.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

if (NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()
if (NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif ()
if (NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif ()


add_subdirectory(pybind11)

pybind11_add_module(
        openfabmap_python3
        openfabmap/src/bowmsctrainer.cpp
        openfabmap/src/chowliutree.cpp
        openfabmap/src/fabmap.cpp
        openfabmap/src/inference.cpp
        openfabmap/src/msckd.cpp
        opencv-ndarray-conversion/conversion.cpp
        src/detectorsAndExtractors.cpp
        src/FabMapVocabulary.cpp
        src/ChowLiuTree.cpp
        src/openFABMAPPython.cpp
        src/PythonBindings.cpp)

include_directories(
        src
        openfabmap/include
        ${NUMPY_INCLUDE_DIR}
        opencv-ndarray-conversion)

target_link_libraries(
        openfabmap_python3
        PRIVATE
        ${OpenCV_LIBRARIES})

file(COPY example.py
        DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
cmake_minimum_required(VERSION 2.6)
project(py-openFABMAP)

#Set the openFABMAP directory path
#TODO: Convert this to a proper FindOpenFABMAP.cmake
set(OPEN_FABMAP_INCLUDE_DIR "/opt/openFABMAP/include" CACHE STRING "Path to parent folder of openFABMAP headers")
set(OPEN_FABMAP_LIB "/opt/openFABMAP/build/lib/libopenFABMAP.so" CACHE STRING "Path to openFABMAP library")

# Macro for opencv version
find_package(OpenCV REQUIRED)
if (NOT (OpenCV_VERSION VERSION_LESS "2.4.0"))
    add_definitions(-DOPENCV2P4)
endif ()

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
        openFABMAP
        opencv-ndarray-conversion/conversion.cpp
        src/detectorsAndExtractors.cpp
        src/FabMapVocabulary.cpp
        src/ChowLiuTree.cpp
        src/openFABMAPPython.cpp
        src/PythonBindings.cpp)

include_directories(
        src
        ${OPEN_FABMAP_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIR}
        opencv-ndarray-conversion)

target_link_libraries(
        openFABMAP
        PRIVATE
        ${OPEN_FABMAP_LIB}
        ${OpenCV_LIBS})
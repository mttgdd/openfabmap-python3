cmake_minimum_required(VERSION 2.6)
project(OPENFABMAP_PYTHON_BINDING)

#Set the openFABMAP directory path
#TODO: Convert this to a proper FindOpenFABMAP.cmake
set(OPEN_FABMAP_INCLUDE_DIR "/opt/openFABMAP/include" CACHE STRING "Path to parent folder of openFABMAP headers")
set(OPEN_FABMAP_LIB "/opt/openFABMAP/build/lib/libopenFABMAP.so" CACHE STRING "Path to openFABMAP library")

# Macro for opencv version
find_package(OpenCV REQUIRED)
if (NOT (OpenCV_VERSION VERSION_LESS "2.4.0"))
    add_definitions(-DOPENCV2P4)
endif ()

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
        src/detectorsAndExtractors.cpp
        src/FabMapVocabluary.cpp
        src/ChowLiuTreeWrapper.cpp
        src/openFABMAPPython.cpp
        src/PythonBindings.cpp)

include_directories(
        src
        ${OPEN_FABMAP_INCLUDE_DIR})

target_link_libraries(
        openFABMAP
        PRIVATE
        ${OPEN_FABMAP_LIB}
        ${OpenCV_LIBS})
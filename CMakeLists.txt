cmake_minimum_required(VERSION 2.6)
project(OPENFABMAP_PYTHON_BINDING)

#Set the openFABMAP directory path
#TODO: Convert this to a proper FindOpenFABMAP.cmake
set(OPEN_FABMAP_INCLUDE_DIR "/opt/openFABMAP/include" CACHE STRING "Path to parent folder of openFABMAP headers")
set(OPEN_FABMAP_LIB "/opt/openFABMAP/build/lib/libopenFABMAP.so" CACHE STRING "Path to openFABMAP library")

find_package(OpenCV REQUIRED)

# Macro for opencv version
if (NOT (OpenCV_VERSION VERSION_LESS "2.4.0"))
    add_definitions(-DOPENCV2P4)
endif ()

add_subdirectory(pybind11)

pybind11_add_module(openFABMAP src/detectorsAndExtractors.cpp
        src/FabMapVocabluary.cpp
        src/ChowLiuTree.cpp
        src/openFABMAPPython.cpp
        src/PythonBindings.cpp)

set_target_properties(openFABMAP
        PROPERTIES
        SUFFIX ".so")

include_directories(
        src
        ${OPEN_FABMAP_INCLUDE_DIR})

target_link_libraries(openFABMAP PRIVATE ${OPEN_FABMAP_LIB})

set(PY_INSTALL_DIR "/usr/local/lib/python${PYTHON_VERSION}/dist-packages" CACHE STRING "Install location for library file.")
install(TARGETS openFABMAP DESTINATION ${PY_INSTALL_DIR})
# openfabmap-python

Pybind11 bindings for openFABMAP as well as conversions to allow pre-loaded images as numpy arrays to be fed into the FABMAP API.

This repository was initially forked from [openfabmap-python](<https://github.com/jskinn/openfabmap-python>) but pybind11 replaced boost-python due to Python2 reaching end of life.

# Requirements

The major requirements are:

* [openFABMAP](https://github.com/arrenglover/openfabmap)
* [OpenCV](https://github.com/opencv/opencv), with nonfree additions

Other requirements, such as [pybind11](https://github.com/pybind/pybind11) and [opencv-ndarray-conversion](https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md) are managed as git submodules.
# Installation

```bash
git clone git@github.com:mttgdd/openfabmap-python.git src
cd src
git submodule update --init
mkdir build
cd build
cmake  ../src -DOPEN_FABMAP_INCLUDE_DIR=/path/to/openfabmap/src/include -DOPEN_FABMAP_LIB=/path/to/libopenFABMAP.a
make -j
```

# References

* <https://github.com/arrenglover/openfabmap>
* <https://github.com/jskinn/openfabmap-python>
* <https://github.com/pybind/pybind11>
* <https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md>


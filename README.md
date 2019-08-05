# openfabmap_python3

Pybind11 bindings for openFABMAP as well as conversions to allow pre-loaded images as numpy arrays to be fed into the FABMAP API.

This repository was initially forked from [openfabmap-python](<https://github.com/jskinn/openfabmap-python>) but pybind11 was selected to replace [boost-python](https://github.com/boostorg/python) due to Python2 reaching [end of life](https://legacy.python.org/dev/peps/pep-0373/).

# Requirements

The major requirement is:

* [OpenCV](https://github.com/opencv/opencv), with nonfree additions

Other requirements, such as [openFABMAP](https://github.com/arrenglover/openfabmap), [pybind11](https://github.com/pybind/pybind11), and [opencv-ndarray-conversion](https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md) are managed as git submodules.

# Installation

OpenCV needs to be installed separately. Once that is done, configure the install for your preferred python version.

```bash
git clone git@github.com:mttgdd/openfabmap-python.git src
cd src
git submodule update --init
mkdir build
cd build
cmake  ../src -DPYTHON_EXECUTABLE=/usr/local/bin/python3.7
make -j
```

Test that installation was successful as follows:

```bash
python -c "from lib import openfabmap_python3"
```

# Examples

## Building a vocabulary

Open an interactive python session from within the installation directory for this library:

```bash
cd build/lib
python
```

Then, the binding module can be imported:

```python
>>> import openfabmap_python3 as of
```

The wrapper for building a vocabulary is configured and initialised using a dictionary:

```python
>>> SETTINGS = dict()
>>> SETTINGS["VocabTrainOptions"] = dict()
>>> SETTINGS["VocabTrainOptions"]["ClusterSize"] = 0.45
>>> vb = of.VocabularyBuilder(SETTINGS)
```

Now, you have the option of adding the features extracted from images to the vocabulary in two ways. The first involves loading the image with OpenCV's C++ methods:

```python
png_file = "example.png"
vb.load_and_add_training_image(png_file)
```

The second allows for a numpy array to be passed directly to the vocabulary builder wrapper. 

```python
>>> from PIL import Image
>>> png_file = "example.png"
>>> img = Image.open(image_file)
>>> vb.add_training_image(img)
```

This functionality allows image manipulation in Python prior to feature extraction (e.g. cropping, rotating, etc).

# References

* <https://github.com/arrenglover/openfabmap>
* <https://github.com/jskinn/openfabmap-python>
* <https://github.com/pybind/pybind11>
* <https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md>


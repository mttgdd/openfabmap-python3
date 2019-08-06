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

# Usage Instructions

## Loading the bindings

Open an interactive python session from within the installation directory for this library:

```bash
cd build/lib
python
```

Then, the binding module can be imported:

```python
>>> import openfabmap_python3 as of
```

## Configuration

The wrapper methods are configured by a Python dictionary, an example of which is shown below:

```python
>>> SETTINGS = dict()
>>> SETTINGS["VocabTrainOptions"] = dict()
>>> SETTINGS["VocabTrainOptions"]["ClusterSize"] = 0.45
```

## Image Manipulation

You have the option of delegating image manipulation and feature extraction to OpenCV's native C++ methods or precomputed from your Python routine. In the first case (where for example you are adding a training image to the vocabulary builder):

```python
>>> png_file = "example.png"
>>> vb.load_and_add_training_image(png_file)
```

In the second case (once again adding a training image to the vocabulary) we use the numpy array convertors of [opencv-ndarray-conversion](https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md): 

```python
>>> from PIL import Image
>>> png_file = "example.png"
>>> img = Image.open(image_file)
>>> vb.add_training_image(img)
```

This functionality allows image manipulation in Python prior to feature extraction (e.g. cropping, rotating, etc), or even feature extraction in Python.

## Building a vocabulary

The wrapper for building a vocabulary is configured and initialised using a dictionary:

```python
>>> vb = of.VocabularyBuilder(SETTINGS)
```

Then, add a number of images to the vocabulary using ```add_training_image``` or ```load_and_add_training_image``` before building the vocabulary:

```python
>>> vb.build_vocabulary() 
```

Likewise, ```add_training_image``` or ```load_and_add_training_image``` are then used to populate the Chowliu tree structures before that model is built:

```python
# inspect the ctor in ChowLiuTree.cpp to see which configuration parameters are required
>>> clt = of.ChowLiuTree(SETTINGS)
>>> clt.build_chow_liu_tree()
```

Finally, the model (including the vocabulary) can be saved to disk using ```save``` (and indeed loaded from disk using ```load```).

# References

* <https://github.com/arrenglover/openfabmap>
* <https://github.com/jskinn/openfabmap-python>
* <https://github.com/pybind/pybind11>
* <https://github.com/yati-sagade/opencv-ndarray-conversion/blob/master/README.md>


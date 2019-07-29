#ifndef OPEN_FABMAP_PYTHON_H
#define OPEN_FABMAP_PYTHON_H

#include "ChowLiuTree.h"
#include "FabMapVocabulary.h"
#include <Python.h>
#include <fabmap.hpp>
#include <memory>
#include <vector>

namespace pyof2 {

class OpenFABMAPPython {
public:
  OpenFABMAPPython(std::shared_ptr<ChowLiuTree> chowLiuTree,
                   pybind11::dict settings = pybind11::dict());
  virtual ~OpenFABMAPPython();

  bool loadAndProcessImage(std::string imageFile);
  bool ProcessImage(const pybind11::array_t<uchar> &frame);

private:
  bool ProcessImageInternal(const cv::Mat &frame);

public:
  int getLastMatch() const;
  pybind11::list getAllLoopClosures() const;

private:
  std::shared_ptr<FabMapVocabulary> vocabulary;
  std::shared_ptr<of2::FabMap> fabmap;

  int imageIndex;
  int lastMatch;
  pybind11::list loopClosures;
};

} // namespace pyof2

#endif // OPEN_FABMAP_PYTHON_H

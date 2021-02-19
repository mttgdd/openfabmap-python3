#ifndef OPEN_FABMAP_PYTHON_H
#define OPEN_FABMAP_PYTHON_H

#include "ChowLiuTree.h"
#include "FabMapVocabulary.h"
#include <Python.h>
#include <fabmap.hpp>
#include <memory>
#include <vector>

namespace ofpy3 {

class OpenFABMAPPython {
public:
  OpenFABMAPPython(std::shared_ptr<ChowLiuTree> chowLiuTree,
                   pybind11::dict settings = pybind11::dict());
  virtual ~OpenFABMAPPython();

  void addDesc(const pybind11::array_t<float> & qImgDesc_arr);

  bool loadAndProcessImage(std::string imageFile);
  bool ProcessImage(const pybind11::array_t<uchar> &frame);
  bool ProcessDesc(const pybind11::array_t<float> & desc_arr, bool addQ);

private:
  bool ProcessImageInternal(const cv::Mat &frame);

public:
  int getLastMatch() const;
  double getLastLikelihood() const;
  pybind11::list getBestLoopClosures() const;
  pybind11::dict getAllLoopClosures() const;

private:
  std::shared_ptr<FabMapVocabulary> vocabulary;
  std::shared_ptr<of2::FabMap> fabmap;

  int imageIndex;
  int lastMatch;
  double lastLikelihood;
  pybind11::list bestLoopClosures;
  pybind11::dict allLoopClosures;
};

} // namespace ofpy3

#endif // OPEN_FABMAP_PYTHON_H

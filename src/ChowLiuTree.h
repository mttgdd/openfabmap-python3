#ifndef CHOWLIUTREE_H
#define CHOWLIUTREE_H

#include "FabMapVocabulary.h"
#include <string>

#include <pybind11/pybind11.h>

namespace ofpy3 {

class ChowLiuTree {
public:
  ChowLiuTree(std::shared_ptr<FabMapVocabulary> vocabulary,
              pybind11::dict settings);
  ChowLiuTree(std::shared_ptr<FabMapVocabulary> vocabulary, cv::Mat chowLiuTree,
              cv::Mat fabmapTrainData, pybind11::dict settings);
  virtual ~ChowLiuTree();

  // These function are exposed to python
  bool loadAndAddTrainingImage(std::string imagePath);
  bool addTrainingImage(const pybind11::array_t<uchar> &frame);
  void buildChowLiuTree();

  bool addTrainingDesc(
      const pybind11::array_t<float> &desc_arr);

 private:
  bool addTrainingImageInternal(const cv::Mat &frame);

public:
  void save(std::string filename) const;
  static std::shared_ptr<ChowLiuTree> load(pybind11::dict settings,
                                           std::string filename);

  bool isTreeBuilt() const;
  std::shared_ptr<FabMapVocabulary> getVocabulary() const;
  cv::Mat getChowLiuTree() const;
  cv::Mat getTrainingData() const;

private:
  std::shared_ptr<FabMapVocabulary> vocabulary;
  cv::Mat chowLiuTree;
  cv::Mat fabmapTrainData;
  double lowerInformationBound;
  bool treeBuilt;
};

} // namespace ofpy3

#endif // CHOWLIUTREE_H

#ifndef FABMAPVOCABLUARY_H
#define FABMAPVOCABLUARY_H

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pyof2 {

class FabMapVocabluary {
public:
  FabMapVocabluary(cv::Ptr<cv::FeatureDetector> detector,
                   cv::Ptr<cv::DescriptorExtractor> extractor,
                   cv::Mat vocabluary);
  virtual ~FabMapVocabluary();

  cv::Mat getVocabluary() const;
  cv::Mat generateBOWImageDescs(const cv::Mat &frame) const;

  void save(cv::FileStorage fileStorage) const;
  static std::shared_ptr<FabMapVocabluary> load(const pybind11::dict &settings,
                                                cv::FileStorage fileStorage);

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Mat vocab;
};

class FabMapVocabluaryBuilder {
public:
  FabMapVocabluaryBuilder(pybind11::dict settings = pybind11::dict());
  virtual ~FabMapVocabluaryBuilder();

  // These function are exposed to python
  bool loadAndAddTrainingImage(std::string imagePath);
  bool addTrainingImage(const pybind11::array_t<uchar> &frame);
  std::shared_ptr<FabMapVocabluary> buildVocabluary();

 private:
  bool addTrainingImageInternal(const cv::Mat &frame);

 private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;

  cv::Mat vocabTrainData;
  double clusterRadius;
};

} // namespace pyof2

#endif // FABMAPVOCABLUARY_H

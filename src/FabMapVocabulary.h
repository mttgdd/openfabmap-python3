#ifndef FABMAPVOCABULARY_H
#define FABMAPVOCABULARY_H

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pyof2 {

class FabMapVocabulary {
public:
  FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                   cv::Ptr<cv::DescriptorExtractor> extractor,
                   cv::Mat vocabulary);
  virtual ~FabMapVocabulary();

  cv::Mat getVocabulary() const;
  cv::Mat generateBOWImageDescs(const cv::Mat &frame) const;

  void save(cv::FileStorage fileStorage) const;
  static std::shared_ptr<FabMapVocabulary> load(const pybind11::dict &settings,
                                                cv::FileStorage fileStorage);

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Mat vocab;
};

class FabMapVocabularyBuilder {
public:
  FabMapVocabularyBuilder(pybind11::dict settings = pybind11::dict());
  virtual ~FabMapVocabularyBuilder();

  // These function are exposed to python
  bool loadAndAddTrainingImage(std::string imagePath);
  bool addTrainingImage(const pybind11::array_t<uchar> &frame);
  std::shared_ptr<FabMapVocabulary> buildVocabulary();

private:
  bool addTrainingImageInternal(const cv::Mat &frame);

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;

  cv::Mat vocabTrainData;
  double clusterRadius;
};

} // namespace pyof2

#endif // FABMAPVOCABULARY_H

#ifndef FABMAPVOCABULARY_H
#define FABMAPVOCABULARY_H

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ofpy3 {

class FabMapVocabulary {
public:
  FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector,
                   cv::Ptr<cv::DescriptorExtractor> extractor,
                   cv::Mat vocabulary);
  FabMapVocabulary(cv::Mat vocabulary);
  FabMapVocabulary();
  virtual ~FabMapVocabulary() = default;

  cv::Mat getVocabulary() const;
  cv::Mat generateBOWImageDescs(const cv::Mat &frame) const;
  cv::Mat generateBOWImageDescsInternal(cv::Mat desc) const;
  pybind11::array_t<float> generateBOWImageDescsExt(const pybind11::array_t<float> &desc_arr) const;

  void compute(
      cv::Ptr<cv::DescriptorMatcher> dmatcher, cv::Mat keypointDescriptors,
      cv::Mat &_imgDescriptor ) const;

  void convert();

  void save(cv::FileStorage fileStorage) const;
  void saveToFile(const std::string filename) const;

  static std::shared_ptr<FabMapVocabulary> load(
      const pybind11::dict &settings,
      cv::FileStorage fileStorage);
  static std::shared_ptr<FabMapVocabulary> loadFromFile(
      const std::string filename);

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Mat vocab;
};

class FabMapVocabularyBuilder {
public:
  explicit FabMapVocabularyBuilder(pybind11::dict settings = pybind11::dict());
  virtual ~FabMapVocabularyBuilder() = default;

  // These function are exposed to python
  void initDetectorExtractor(pybind11::dict settings);
  bool loadAndAddTrainingImage(std::string imagePath);
  bool addTrainingImage(const pybind11::array_t<uchar> &frame);
  void addTrainingDescs(const pybind11::array_t<float> &descs_arr);

  std::shared_ptr<FabMapVocabulary> buildVocabulary();

private:
  bool addTrainingImageInternal(const cv::Mat &frame);
  void addTrainingDescsInternal(const cv::Mat &descs);

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;

  cv::Mat vocabTrainData;
  double clusterRadius;
};

} // namespace ofpy3

#endif // FABMAPVOCABULARY_H

#ifdef OPENCV2P4
//#include <opencv2/nonfree/nonfree.hpp>
#endif
#include "FabMapVocabulary.h"
#include "detectorsAndExtractors.h"
#include <bowmsctrainer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <conversion.h>
#include <iostream>

// ----------------- FabMapVocabulary -----------------

ofpy3::FabMapVocabulary::FabMapVocabulary(
    cv::Ptr<cv::FeatureDetector> detector,
    cv::Ptr<cv::DescriptorExtractor> extractor, cv::Mat vocabulary)
    : detector(std::move(detector)), extractor(std::move(extractor)),
      vocab(std::move(vocabulary)) {}

cv::Mat ofpy3::FabMapVocabulary::getVocabulary() const { return vocab; }

cv::Mat
ofpy3::FabMapVocabulary::generateBOWImageDescs(const cv::Mat &frame) const {
  // use a FLANN matcher to generate bag-of-words representations
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("FlannBased");
  cv::BOWImgDescriptorExtractor bide(extractor, matcher);
  bide.setVocabulary(vocab);

  cv::Mat bow;
  std::vector<cv::KeyPoint> kpts;

  detector->detect(frame, kpts);
  bide.compute(frame, kpts, bow);
  return bow;
}

void ofpy3::FabMapVocabulary::save(cv::FileStorage fileStorage) const {
  // Note that this is a partial save, assume that the settings are saved
  // elsewhere.
  fileStorage << "Vocabulary" << vocab;
}

std::shared_ptr<ofpy3::FabMapVocabulary>
ofpy3::FabMapVocabulary::load(const pybind11::dict &settings,
                              cv::FileStorage fileStorage) {
  cv::Mat vocab;
  fileStorage["Vocabulary"] >> vocab;

  return std::make_shared<ofpy3::FabMapVocabulary>(
      ofpy3::generateDetector(settings), ofpy3::generateExtractor(settings),
      vocab);
}

// ----------------- FabMapVocabularyBuilder -----------------

ofpy3::FabMapVocabularyBuilder::FabMapVocabularyBuilder(pybind11::dict settings)
    : vocabTrainData(), clusterRadius(0.45) {
  if (settings.contains("VocabTrainOptions")) {
    pybind11::dict trainSettings = settings["VocabTrainOptions"];
    if (trainSettings.contains("ClusterSize")) {
      clusterRadius = trainSettings["ClusterSize"].cast<double>();
    }
  }
}

void ofpy3::FabMapVocabularyBuilder::initDetectorExtractor(
    pybind11::dict settings) {
  detector = ofpy3::generateDetector(settings);
  extractor = ofpy3::generateExtractor(settings);
}

bool ofpy3::FabMapVocabularyBuilder::loadAndAddTrainingImage(
    std::string imagePath) {
  cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
  return addTrainingImageInternal(frame);
}

bool ofpy3::FabMapVocabularyBuilder::addTrainingImage(
    const pybind11::array_t<uchar> &frame) {
  NDArrayConverter cvt;
  cv::Mat mat{cvt.toMat(frame.ptr())};
  return addTrainingImageInternal(mat);
}

void ofpy3::FabMapVocabularyBuilder::addTrainingDescs(
    const pybind11::array_t<uchar> &descs) {
  NDArrayConverter cvt;
  cv::Mat mat{cvt.toMat(descs.ptr())};
  addTrainingDescsInternal(mat);
}

bool ofpy3::FabMapVocabularyBuilder::addTrainingImageInternal(
    const cv::Mat &frame) {
  cv::Mat descs, feats;
  std::vector<cv::KeyPoint> kpts;

  if (frame.data) {
    // detect & extract features
    detector->detect(frame, kpts);
    extractor->compute(frame, kpts, descs);

    // add all descriptors to the training data
    addTrainingDescsInternal(descs);
    return true;
  }
  return false;
}

void ofpy3::FabMapVocabularyBuilder::addTrainingDescsInternal(
    const cv::Mat &descs) {
  vocabTrainData.push_back(descs);
}

std::shared_ptr<ofpy3::FabMapVocabulary>
ofpy3::FabMapVocabularyBuilder::buildVocabulary() {
  // Build the vocab
  of2::BOWMSCTrainer trainer(clusterRadius);
  trainer.add(vocabTrainData);
  cv::Mat vocab = trainer.cluster();

  // Return the vocab object
  return std::make_shared<ofpy3::FabMapVocabulary>(detector, extractor,
                                                   std::move(vocab));
}

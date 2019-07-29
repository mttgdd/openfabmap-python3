#ifdef OPENCV2P4
//#include <opencv2/nonfree/nonfree.hpp>
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bowmsctrainer.hpp>
#include "FabMapVocabulary.h"
#include "detectorsAndExtractors.h"

#include <conversion.h>
#include <iostream>

// ----------------- FabMapVocabulary -----------------

pyof2::FabMapVocabulary::FabMapVocabulary(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, cv::Mat vocabulary) :
        detector(std::move(detector)),
        extractor(std::move(extractor)),
        vocab(std::move(vocabulary))
{
    
}

pyof2::FabMapVocabulary::~FabMapVocabulary()
{
    
}
    
cv::Mat pyof2::FabMapVocabulary::getVocabulary() const
{
    return vocab;
}

cv::Mat pyof2::FabMapVocabulary::generateBOWImageDescs(const cv::Mat& frame) const
{
    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);
    
    cv::Mat bow;
    std::vector<cv::KeyPoint> kpts;

    detector->detect(frame, kpts);
    bide.compute(frame, kpts, bow);
    return bow;
}

void pyof2::FabMapVocabulary::save(cv::FileStorage fileStorage) const
{
    // Note that this is a partial save, assume that the settings are saved elsewhere.
    fileStorage << "Vocabulary" << vocab;
}

std::shared_ptr<pyof2::FabMapVocabulary> pyof2::FabMapVocabulary::load(const pybind11::dict& settings, cv::FileStorage fileStorage)
{
    cv::Mat vocab;
    fileStorage["Vocabulary"] >> vocab;
    
    return std::make_shared<pyof2::FabMapVocabulary>(
        pyof2::generateDetector(settings),
        pyof2::generateExtractor(settings),
        vocab);
}

// ----------------- FabMapVocabularyBuilder -----------------

pyof2::FabMapVocabularyBuilder::FabMapVocabularyBuilder(pybind11::dict settings) :
        detector(pyof2::generateDetector(settings)),
        extractor(pyof2::generateExtractor(settings)),
        vocabTrainData(),
        clusterRadius(0.45)
{
    if (settings.contains("VocabTrainOptions"))
    {
        pybind11::dict trainSettings = settings["VocabTrainOptions"];
        if (trainSettings.contains("ClusterSize"))
        {
            clusterRadius = trainSettings["ClusterSize"].cast<double>();
        }
    }
}

pyof2::FabMapVocabularyBuilder::~FabMapVocabularyBuilder()
{
    
}

bool pyof2::FabMapVocabularyBuilder::loadAndAddTrainingImage(std::string imagePath)
{
    cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
    return addTrainingImageInternal(frame);
}

bool pyof2::FabMapVocabularyBuilder::addTrainingImage(const pybind11::array_t<uchar> &frame)
{
  NDArrayConverter cvt;
  cv::Mat mat { cvt.toMat(frame.ptr()) };
  return addTrainingImageInternal(mat);
}

bool pyof2::FabMapVocabularyBuilder::addTrainingImageInternal(const cv::Mat &frame)
{
    cv::Mat descs, feats;
    std::vector<cv::KeyPoint> kpts;

    if (frame.data)
    {
        //detect & extract features
        detector->detect(frame, kpts);
        extractor->compute(frame, kpts, descs);

        //add all descriptors to the training data
        vocabTrainData.push_back(descs);
        return true;
    }
    return false;
}

std::shared_ptr<pyof2::FabMapVocabulary> pyof2::FabMapVocabularyBuilder::buildVocabulary()
{
    // Build the vocab
    of2::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);
    cv::Mat vocab = trainer.cluster();
    
    // Return the vocab object
    return std::make_shared<pyof2::FabMapVocabulary>(detector, extractor, std::move(vocab));
}


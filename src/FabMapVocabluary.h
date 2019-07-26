#ifndef FABMAPVOCABLUARY_H
#define FABMAPVOCABLUARY_H

#include <string>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pybind11/pybind11.h>

namespace pyof2
{

class FabMapVocabluary
{
public:
    FabMapVocabluary(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, cv::Mat vocabluary);
    virtual ~FabMapVocabluary();
    
    cv::Mat getVocabluary() const;
    cv::Mat generateBOWImageDescs(const cv::Mat& frame) const;
    
    void save(cv::FileStorage fileStorage) const;
    static std::shared_ptr<FabMapVocabluary> load(const pybind11::dict& settings, cv::FileStorage fileStorage);
    
private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Mat vocab;
};
    
class FabMapVocabluaryBuilder
{
public:    
    FabMapVocabluaryBuilder(pybind11::dict settings = pybind11::dict());
    virtual ~FabMapVocabluaryBuilder();
    
    // These function are exposed to python
    bool addTrainingImage(std::string imagePath);
    std::shared_ptr<FabMapVocabluary> buildVocabluary();
    
private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;

    cv::Mat vocabTrainData;
    double clusterRadius;
};

}

#endif // FABMAPVOCABLUARY_H

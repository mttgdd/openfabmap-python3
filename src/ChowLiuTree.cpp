#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chowliutree.hpp>
#include "ChowLiuTree.h"
#include <conversion.h>

// ----------------- ChowLiuTree -----------------

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabulary> vocabulary, pybind11::dict settings) :
    ChowLiuTree(vocabulary, cv::Mat(), cv::Mat(), settings)
{
    
}

pyof2::ChowLiuTree::ChowLiuTree(std::shared_ptr<FabMapVocabulary> vocabulary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, pybind11::dict settings) :
    vocabulary(vocabulary),
    chowLiuTree(std::move(chowLiuTree)),
    fabmapTrainData(std::move(fabmapTrainData)),
    lowerInformationBound(0.0005),
    treeBuilt(!this->chowLiuTree.empty())
{
    if (settings.contains("ChowLiuOptions"))
    {
        pybind11::dict trainSettings = settings["ChowLiuOptions"];
        if (trainSettings.contains("LowerInfoBound"))
        {
            lowerInformationBound = trainSettings["LowerInfoBound"].cast<double>();
        }
    }
}

pyof2::ChowLiuTree::~ChowLiuTree()
{
    
}

bool pyof2::ChowLiuTree::loadAndAddTrainingImage(std::string imagePath)
{
    cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
    return addTrainingImageInternal(frame);
}

bool pyof2::ChowLiuTree::addTrainingImage(const pybind11::array_t<uchar> &frame)
{
    NDArrayConverter cvt;
    cv::Mat mat { cvt.toMat(frame.ptr()) };
    return addTrainingImageInternal(mat);
}

bool pyof2::ChowLiuTree::addTrainingImageInternal(const cv::Mat &frame)
{
    if (frame.data)
    {
        cv::Mat bow = vocabulary->generateBOWImageDescs(frame);
        fabmapTrainData.push_back(std::move(bow));
        treeBuilt = false;
        return true;
    }
    return false;
}

void pyof2::ChowLiuTree::buildChowLiuTree()
{
    of2::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    chowLiuTree = tree.make(lowerInformationBound);
    treeBuilt = true;
}

bool pyof2::ChowLiuTree::isTreeBuilt() const
{
    return treeBuilt;
}

std::shared_ptr<pyof2::FabMapVocabulary> pyof2::ChowLiuTree::getVocabulary() const
{
    return vocabulary;
}
    
cv::Mat pyof2::ChowLiuTree::getChowLiuTree() const
{
    return chowLiuTree;
}

cv::Mat pyof2::ChowLiuTree::getTrainingData() const
{
    return fabmapTrainData;
}
    
void pyof2::ChowLiuTree::save(std::string filename) const
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::WRITE);
    vocabulary->save(fs);
    if (treeBuilt)
    {
        fs << "ChowLiuTree" << chowLiuTree;
        fs << "FabMapTrainingData" << fabmapTrainData;
    }
    fs.release();
}

std::shared_ptr<pyof2::ChowLiuTree> pyof2::ChowLiuTree::load(pybind11::dict settings, std::string filename)
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::READ);
    
    std::shared_ptr<pyof2::FabMapVocabulary> vocab = pyof2::FabMapVocabulary::load(settings, fs);
    
    cv::Mat chowLiuTree;
    fs["ChowLiuTree"] >> chowLiuTree;
    
    cv::Mat fabmapTrainData;
    fs["FabMapTrainingData"] >> fabmapTrainData;
    
    fs.release();
    
    return std::make_shared<pyof2::ChowLiuTree>(vocab, chowLiuTree, fabmapTrainData, settings);
}

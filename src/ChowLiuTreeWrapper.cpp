#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chowliutree.hpp>
#include "ChowLiuTreeWrapper.h"

// ----------------- ChowLiuTree -----------------

pyof2::ChowLiuTreeWrapper::ChowLiuTreeWrapper(std::shared_ptr<FabMapVocabluary> vocabluary, pybind11::dict settings) :
    ChowLiuTreeWrapper(vocabluary, cv::Mat(), cv::Mat(), settings)
{
    
}

pyof2::ChowLiuTreeWrapper::ChowLiuTreeWrapper(std::shared_ptr<FabMapVocabluary> vocabluary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, pybind11::dict settings) :
    vocabluary(vocabluary),
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

pyof2::ChowLiuTreeWrapper::~ChowLiuTreeWrapper()
{
    
}

bool pyof2::ChowLiuTreeWrapper::addTrainingImage(std::string imagePath)
{
    cv::Mat frame = cv::imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
    if (frame.data)
    {
        cv::Mat bow = vocabluary->generateBOWImageDescs(frame);
        fabmapTrainData.push_back(std::move(bow));
        treeBuilt = false;
        return true;
    }
    return false;
}

void pyof2::ChowLiuTreeWrapper::buildChowLiuTree()
{
    of2::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    chowLiuTree = tree.make(lowerInformationBound);
    treeBuilt = true;
}

bool pyof2::ChowLiuTreeWrapper::isTreeBuilt() const
{
    return treeBuilt;
}

std::shared_ptr<pyof2::FabMapVocabluary> pyof2::ChowLiuTreeWrapper::getVocabluary() const
{
    return vocabluary;
}
    
cv::Mat pyof2::ChowLiuTreeWrapper::getChowLiuTree() const
{
    return chowLiuTree;
}

cv::Mat pyof2::ChowLiuTreeWrapper::getTrainingData() const
{
    return fabmapTrainData;
}
    
void pyof2::ChowLiuTreeWrapper::save(std::string filename) const
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::WRITE);
    vocabluary->save(fs);
    if (treeBuilt)
    {
        fs << "ChowLiuTree" << chowLiuTree;
        fs << "FabMapTrainingData" << fabmapTrainData;
    }
    fs.release();
}

std::shared_ptr<pyof2::ChowLiuTreeWrapper> pyof2::ChowLiuTreeWrapper::load(pybind11::dict settings, std::string filename)
{
    cv::FileStorage fs;	
    fs.open(filename, cv::FileStorage::READ);
    
    std::shared_ptr<pyof2::FabMapVocabluary> vocab = pyof2::FabMapVocabluary::load(settings, fs);
    
    cv::Mat chowLiuTree;
    fs["ChowLiuTree"] >> chowLiuTree;
    
    cv::Mat fabmapTrainData;
    fs["FabMapTrainingData"] >> fabmapTrainData;
    
    fs.release();
    
    return std::make_shared<pyof2::ChowLiuTreeWrapper>(vocab, chowLiuTree, fabmapTrainData, settings);
}

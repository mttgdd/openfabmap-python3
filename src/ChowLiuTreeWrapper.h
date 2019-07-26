#ifndef CHOWLIUTREE_H
#define CHOWLIUTREE_H

#include <string>
#include "FabMapVocabluary.h"

#include <pybind11/pybind11.h>

namespace pyof2
{

class ChowLiuTreeWrapper
{
public:
    ChowLiuTreeWrapper(std::shared_ptr<FabMapVocabluary> vocabluary, pybind11::dict settings);
    ChowLiuTreeWrapper(std::shared_ptr<FabMapVocabluary> vocabluary, cv::Mat chowLiuTree, cv::Mat fabmapTrainData, pybind11::dict settings);
    virtual ~ChowLiuTreeWrapper();
    
    // These function are exposed to python
    bool addTrainingImage(std::string imagePath);
    void buildChowLiuTree();
    
    void save(std::string filename) const;
    static std::shared_ptr<ChowLiuTreeWrapper> load(pybind11::dict settings, std::string filename);
    
    bool isTreeBuilt() const;
    std::shared_ptr<FabMapVocabluary> getVocabluary() const;
    cv::Mat getChowLiuTree() const;
    cv::Mat getTrainingData() const;
    
private:
    std::shared_ptr<FabMapVocabluary> vocabluary;
    cv::Mat chowLiuTree;
    cv::Mat fabmapTrainData;
    double lowerInformationBound;
    bool treeBuilt;
};

}

#endif // CHOWLIUTREE_H

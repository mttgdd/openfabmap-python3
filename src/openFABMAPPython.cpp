/*//////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/] -or-
// [https://github.com/arrenglover/openfabmap]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote
//    products derived from this software without specific prior written
///   permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability,or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//////////////////////////////////////////////////////////////////////////////*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "openFABMAPPython.h"
#include <conversion.h>

// ----------------- OpenFABMAPPython -----------------

pyof2::OpenFABMAPPython::OpenFABMAPPython(std::shared_ptr<pyof2::ChowLiuTree> chowLiuTree, pybind11::dict settings) :
        vocabulary(chowLiuTree->getVocabulary()),
        fabmap(),
        imageIndex(0),
        lastMatch(-1),
        loopClosures()
{
    // Build the chow liu tree, if it hasn't been already.
    if (!chowLiuTree->isTreeBuilt())
    {
        chowLiuTree->buildChowLiuTree();
    }
    pybind11::dict openFabMapOptions;
    if (settings.contains("openFabMapOptions")) {
        openFabMapOptions = settings["openFabMapOptions"];
    }
    
    //create options flags
    std::string newPlaceMethod = "Meanfield";
    std::string bayesMethod = "Naive";
    bool simpleMotionModel = false;
    
    if (openFabMapOptions.contains("NewPlaceMethod")) {
        newPlaceMethod = openFabMapOptions["NewPlaceMethod"].cast<std::string>();
    }
    if (openFabMapOptions.contains("BayesMethod")) {
        bayesMethod = openFabMapOptions["BayesMethod"].cast<std::string>();
    }
    if (openFabMapOptions.contains("SimpleMotion")) {
        simpleMotionModel = openFabMapOptions["SimpleMotion"].cast<bool>();
    }
    
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2::FabMap::SAMPLED;
    } else {
        options |= of2::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2::FabMap::CHOW_LIU;
    } else {
        options |= of2::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2::FabMap::MOTION_MODEL;
    }

    //create an instance of the desired type of FabMap
    std::string fabMapVersion = "FABMAP2";
    if (openFabMapOptions.contains("FabMapVersion")) {
        fabMapVersion = openFabMapOptions["FabMapVersion"].cast<std::string>();
    }
    
    // Read common settings
    double PzGe = 0.39;
    double PzGne = 0.0;
    int numSamples = 3000;
    if (openFabMapOptions.contains("PzGe")) {
        PzGe = openFabMapOptions["PzGe"].cast<double>();
    }
    if (openFabMapOptions.contains("PzGne")) {
        PzGne = openFabMapOptions["PzGne"].cast<double>();
    }
    if (openFabMapOptions.contains("SimpleMotion")) {
        numSamples = openFabMapOptions["SimpleMotion"].cast<int>();
    }
    
    // Create the appropriate FABMAP object
    if(fabMapVersion == "FABMAP1") {
        fabmap = std::make_shared<of2::FabMap1>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples);
    } else if(fabMapVersion == "FABMAPLUT") {
        int precision = 6;
        if (openFabMapOptions.contains("PzGe")) {
            precision = openFabMapOptions["PzGe"].cast<int>();
        }
        
        fabmap = std::make_shared<of2::FabMapLUT>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples, precision);
    } else if(fabMapVersion == "FABMAPFBO") {
        double rejectionThreshold = 1e-8;
        double PsGd = 1e-8;
        int bisectionStart = 512;
        int bisectionIts = 9;
        
        if (openFabMapOptions.contains("RejectionThreshold")) {
            rejectionThreshold = openFabMapOptions["RejectionThreshold"].cast<double>();
        }
        if (openFabMapOptions.contains("PsGd")) {
            PsGd = openFabMapOptions["PsGd"].cast<double>();
        }
        if (openFabMapOptions.contains("BisectionStart")) {
            bisectionStart = openFabMapOptions["BisectionStart"].cast<int>();
        }
        if (openFabMapOptions.contains("BisectionIts")) {
            bisectionIts = openFabMapOptions["BisectionIts"].cast<int>();
        }
        
        fabmap = std::make_shared<of2::FabMapFBO>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples, rejectionThreshold, PsGd, bisectionStart, bisectionIts);
    } else {    // Default to FABMAP2
        fabmap = std::make_shared<of2::FabMap2>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options);
    }

    //add the training data for use with the sampling method
    fabmap->addTraining(chowLiuTree->getTrainingData());
}

pyof2::OpenFABMAPPython::~OpenFABMAPPython()
{
    
}

bool pyof2::OpenFABMAPPython::loadAndProcessImage(std::string imageFile)
{
    cv::Mat frame = cv::imread(imageFile, CV_LOAD_IMAGE_UNCHANGED);
    return ProcessImageInternal(frame);
}

bool pyof2::OpenFABMAPPython::ProcessImage(const pybind11::array_t<uchar> &frame) {
    NDArrayConverter cvt;
    cv::Mat mat { cvt.toMat(frame.ptr()) };
    return ProcessImageInternal(mat);
}

bool pyof2::OpenFABMAPPython::ProcessImageInternal(const cv::Mat &frame)
{
    if (frame.data)
    {
        cv::Mat bow = vocabulary->generateBOWImageDescs(frame);
        if (!bow.empty())
        {
            std::vector<of2::IMatch> matches;
            fabmap->localize(bow, matches, true);

            double bestLikelihood = 0.0;
            int bestMatchIndex = -1;
            for (std::vector<of2::IMatch>::iterator iter = matches.begin(); iter != matches.end(); ++iter)
            {
                if (iter->likelihood > bestLikelihood)
                {
                    bestLikelihood = iter->likelihood;
                    bestMatchIndex = iter->imgIdx;
                }
            }
            lastMatch = bestMatchIndex;
            loopClosures.append(pybind11::make_tuple(imageIndex, bestMatchIndex));
            ++imageIndex;
            return true;
        }
        else {
            return false;
        }
        return true;
    }
    return false;
}


int pyof2::OpenFABMAPPython::getLastMatch() const
{
    return lastMatch;
}

pybind11::list pyof2::OpenFABMAPPython::getAllLoopClosures() const
{
    return loopClosures;
}

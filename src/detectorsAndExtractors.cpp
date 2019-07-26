#include <opencv2/core/core.hpp>
#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif
#include "detectorsAndExtractors.h"

// ------------------- DETECTORS -------------------

cv::Ptr<cv::FeatureDetector> createSTAR(const pybind11::dict& settings)
{
    int maxSize = 32;
    int responseThreshold = 30;
    int lineThreshold = 10;
    int lineBinarized = 8;
    int suppressNonmaxSize = 5;
    
    if (settings.contains("MaxSize")) {
        maxSize = settings["MaxSize"].cast<int>();
    }
    if (settings.contains("Response")) {
        responseThreshold = settings["Response"].cast<int>();
    }
    if (settings.contains("LineThreshold")) {
        lineThreshold = settings["LineThreshold"].cast<int>();
    }
    if (settings.contains("LineBinarized")) {
        lineBinarized = settings["LineBinarized"].cast<int>();
    }
    if (settings.contains("Suppression")) {
        suppressNonmaxSize = settings["Suppression"].cast<int>();
    }
    
    return cv::makePtr<cv::StarFeatureDetector>(maxSize, responseThreshold, lineThreshold, lineBinarized, suppressNonmaxSize);
}

cv::Ptr<cv::FeatureDetector> createFAST(const pybind11::dict& settings)
{
    int threshold = 10;
    bool nonmaxSuppression = true;
    
    if (settings.contains("Threshold")) {
        threshold = settings["Threshold"].cast<int>();
    }
    if (settings.contains("NonMaxSuppression")) {
        nonmaxSuppression = settings["NonMaxSuppression"].cast<bool>();
    }
    
    return cv::makePtr<cv::FastFeatureDetector>(threshold, nonmaxSuppression);
}

cv::Ptr<cv::FeatureDetector> createSURF(const pybind11::dict& settings)
{
    double hessianThreshold = 400;
    int nOctaves = 4;
    int nOctaveLayers = 2;
    bool extended = true;
    bool upright = false;
    
    if (settings.contains("HessianThreshold")) {
        hessianThreshold = settings["HessianThreshold"].cast<double>();
    }
    if (settings.contains("NumOctaves")) {
        nOctaves = settings["NumOctaves"].cast<int>() ;
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = settings["NumOctaveLayers"].cast<int>() ;
    }
    if (settings.contains("Extended")) {
        extended = settings["Extended"].cast<bool>() ;
    }
    if (settings.contains("Upright")) {
        upright = settings["Upright"].cast<bool>() ;
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SURF>(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
#else
    return cv::makePtr<cv::SurfFeatureDetector>(hessianThreshold, nOctaves, nOctaveLayers, upright);
#endif
}

cv::Ptr<cv::FeatureDetector> createSIFT(const pybind11::dict& settings)
{
    int numFeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    
    if (settings.contains("NumFeatures")) {
        numFeatures = settings["NumFeatures"].cast<int>();
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = settings["NumOctaveLayers"].cast<int>();
    }
    if (settings.contains("ContrastThreshold")) {
        contrastThreshold = settings["ContrastThreshold"].cast<double>();
    }
    if (settings.contains("EdgeThreshold")) {
        edgeThreshold = settings["EdgeThreshold"].cast<double>();
    }
    if (settings.contains("Sigma")) {
        sigma = settings["Sigma"].cast<double>();
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SIFT>(numFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
    return cv::makePtr<cv::SiftFeatureDetector>(contrastThreshold, edgeThreshold);
#endif
}

cv::Ptr<cv::FeatureDetector> createMSER(const pybind11::dict& settings)
{
    int delta = 5;
    int minArea = 60;
    int maxArea = 14400;
    double maxVariation = 0.25;
    double minDiversity = 0.2;
    double maxEvolution = 200;
    double areaThreshold = 1.01;
    double minMargin = 0.003;
    int edgeBlurSize = 5;
    
    if (settings.contains("Delta")) {
        delta = settings["Delta"].cast<int>();
    }
    if (settings.contains("MinArea")) {
        minArea = settings["MinArea"].cast<int>();
    }
    if (settings.contains("MaxArea")) {
        maxArea = settings["MaxArea"].cast<int>();
    }
    if (settings.contains("MaxVariation")) {
        maxVariation = settings["MaxVariation"].cast<double>();
    }
    if (settings.contains("MinDiversity")) {
        minDiversity = settings["MinDiversity"].cast<double>();
    }
    if (settings.contains("MaxEvolution")) {
        maxEvolution = settings["MaxEvolution"].cast<double>();
    }
    if (settings.contains("AreaThreshold")) {
        areaThreshold = settings["AreaThreshold"].cast<double>();
    }
    if (settings.contains("MinMargin")) {
        minMargin = settings["MinMargin"].cast<double>();
    }
    if (settings.contains("EdgeBlurSize")) {
        edgeBlurSize = settings["EdgeBlurSize"].cast<int>();
    }
    
    return cv::makePtr<cv::MserFeatureDetector>(delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution, areaThreshold, minMargin, edgeBlurSize);
}

/**
 * Generates a feature detector based on options in the settings dict.
 * Does some fiddling for the setttings structure.
 * Will work with no settings specified, defaults to a STAR detector in STATIC detector mode.
 * Individual detector settings default to as in the OpenCV documentation, or as in the
 * sample openFABMAP settings where no OpenCV default.
 * 
 * @param settings A Python dict of settings, the full settings object.
 * @return A cv::FeatureDetector pointer, as a cv::Ptr (for OpenCV compatibility)
 */
cv::Ptr<cv::FeatureDetector> pyof2::generateDetector(const pybind11::dict &settings) {
    // Get the feature settings
    pybind11::dict featureOptions;
    if (settings.contains("FeatureOptions"))
    {
        featureOptions = settings["FeatureOptions"];
    }
    
    // Read the settings, with default values.
    std::string detectorMode = "STATIC";
    std::string detectorType = "STAR";
    if (featureOptions.contains("DetectorMode")) {
        detectorMode = featureOptions["DetectorMode"].cast<std::string>();
    }
    if (featureOptions.contains("DetectorType")) {
        detectorType = featureOptions["DetectorType"].cast<std::string>();
    }
    
    // 
    if(detectorMode == "ADAPTIVE") {

        if(detectorType != "STAR" && detectorType != "SURF" && detectorType != "FAST") {
            //Adaptive Detectors only work with STAR, SURF and FAST
            detectorType = "STAR";
        }
        
        // Get the settings for adaptive features
        pybind11::dict adaptiveOptions;
        if (featureOptions.contains("Adaptive"))
        {
            adaptiveOptions = featureOptions["Adaptive"];
        }
        
        // Defaults from the OpenCV documentation
        int minFeatures = 400;
        int maxFeatures = 500;
        int maxIters = 5;
        if (adaptiveOptions.contains("MinFeatures")) {
            minFeatures = adaptiveOptions["MinFeatures"].cast<int>();
        }
        if (adaptiveOptions.contains("MaxFeatures")) {
            maxFeatures = adaptiveOptions["MaxFeatures"].cast<int>();
        }
        if (adaptiveOptions.contains("MaxIters")) {
            maxIters = adaptiveOptions["MaxIters"].cast<int>();
        }
        return cv::makePtr<cv::DynamicAdaptedFeatureDetector>(cv::AdjusterAdapter::create(detectorType), minFeatures, maxFeatures, maxIters);

    } else {
        pybind11::dict detectorOptions;
        if(detectorType == "FAST") {
            if (featureOptions.contains("FastDetector"))
            {
                detectorOptions = featureOptions["FastDetector"];
            }
            return createFAST(detectorOptions);
        } else if(detectorType == "SURF") {
            if (featureOptions.contains("SurfDetector"))
            {
                detectorOptions = featureOptions["SurfDetector"];
            }
            return createSURF(detectorOptions);
        } else if(detectorType == "SIFT") {
            if (featureOptions.contains("SiftDetector"))
            {
                detectorOptions = featureOptions["SiftDetector"];
            }
            return createSIFT(detectorOptions);
        } else if(detectorType == "MSER") {
            if (featureOptions.contains("MSERDetector"))
            {
                detectorOptions = featureOptions["MSERDetector"];
            }
            return createMSER(detectorOptions);
        } else {
            if (featureOptions.contains("StarDetector"))
            {
                detectorOptions = featureOptions["StarDetector"];
            }
            return createSTAR(detectorOptions);
        }
    }
}

// ------------------- EXTRACTORS -------------------

cv::Ptr<cv::DescriptorExtractor> createSIFTExtractor(const pybind11::dict& settings)
{
    int numFeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    
    if (settings.contains("NumFeatures")) {
        numFeatures = settings["NumFeatures"].cast<int>();
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = settings["NumOctaveLayers"].cast<int>();
    }
    if (settings.contains("ContrastThreshold")) {
        contrastThreshold = settings["ContrastThreshold"].cast<double>();
    }
    if (settings.contains("EdgeThreshold")) {
        edgeThreshold = settings["EdgeThreshold"].cast<double>();
    }
    if (settings.contains("Sigma")) {
        sigma = settings["Sigma"].cast<double>();
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SIFT>(numFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
    return cv::makePtr<cv::SiftDescriptorExtractor>();
#endif
}

cv::Ptr<cv::DescriptorExtractor> createSURFExtractor(const pybind11::dict& settings)
{
    double hessianThreshold = 400;
    int nOctaves = 4;
    int nOctaveLayers = 2;
    bool extended = true;
    bool upright = false;
    
    if (settings.contains("HessianThreshold")) {
        hessianThreshold = settings["HessianThreshold"].cast<double>();
    }
    if (settings.contains("NumOctaves")) {
        nOctaves = settings["NumOctaves"].cast<int>();
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = settings["NumOctaveLayers"].cast<int>();
    }
    if (settings.contains("Extended")) {
        extended = settings["Extended"].cast<bool>();
    }
    if (settings.contains("Upright")) {
        upright = settings["Upright"].cast<bool>();
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SURF>(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
#else
    return cv::makePtr<cv::SurfDescriptorExtractor>(nOctaves, nOctaveLayers, extended, upright);
#endif
}

/**
 * Generates a feature detector based on options in the settings file
 * 
 * @param settings A python dict of settings,
 * @return 
 */
cv::Ptr<cv::DescriptorExtractor> pyof2::generateExtractor(const pybind11::dict &settings)
{
    // Get the feature settings
    pybind11::dict featureOptions;
    if (settings.contains("FeatureOptions"))
    {
        featureOptions = settings["FeatureOptions"];
    }
    
    std::string extractorType = "SURF";
    if (featureOptions.contains("ExtractorType")) {
        extractorType = featureOptions["ExtractorType"].cast<std::string>();
    }
    
    pybind11::dict detectorOptions;
    if(extractorType == "SIFT") {
        if (featureOptions.contains("SiftDetector"))
        {
            detectorOptions = featureOptions["SiftDetector"];
        }
        return createSIFTExtractor(detectorOptions);
    } else {
        if (featureOptions.contains("SurfDetector"))
        {
            detectorOptions = featureOptions["SurfDetector"];
        }
        return createSURFExtractor(detectorOptions);
    }
}
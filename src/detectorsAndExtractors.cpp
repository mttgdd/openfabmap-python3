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
        nOctaves = boost::python::extract<int>(settings.get("NumOctaves"));
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.contains("Extended")) {
        extended = boost::python::extract<bool>(settings.get("Extended"));
    }
    if (settings.contains("Upright")) {
        upright = boost::python::extract<bool>(settings.get("Upright"));
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
        numFeatures = boost::python::extract<int>(settings.get("NumFeatures"));
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.contains("ContrastThreshold")) {
        contrastThreshold = boost::python::extract<double>(settings.get("ContrastThreshold"));
    }
    if (settings.contains("EdgeThreshold")) {
        edgeThreshold = boost::python::extract<double>(settings.get("EdgeThreshold"));
    }
    if (settings.contains("Sigma")) {
        sigma = boost::python::extract<double>(settings.get("Sigma"));
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
        delta = boost::python::extract<int>(settings.get("Delta"));
    }
    if (settings.contains("MinArea")) {
        minArea = boost::python::extract<int>(settings.get("MinArea"));
    }
    if (settings.contains("MaxArea")) {
        maxArea = boost::python::extract<int>(settings.get("MaxArea"));
    }
    if (settings.contains("MaxVariation")) {
        maxVariation = boost::python::extract<double>(settings.get("MaxVariation"));
    }
    if (settings.contains("MinDiversity")) {
        minDiversity = boost::python::extract<double>(settings.get("MinDiversity"));
    }
    if (settings.contains("MaxEvolution")) {
        maxEvolution = boost::python::extract<double>(settings.get("MaxEvolution"));
    }
    if (settings.contains("AreaThreshold")) {
        areaThreshold = boost::python::extract<double>(settings.get("AreaThreshold"));
    }
    if (settings.contains("MinMargin")) {
        minMargin = boost::python::extract<double>(settings.get("MinMargin"));
    }
    if (settings.contains("EdgeBlurSize")) {
        edgeBlurSize = boost::python::extract<int>(settings.get("EdgeBlurSize"));
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
        featureOptions = boost::python::extract<pybind11::dict>(settings.get("FeatureOptions"));
    }
    
    // Read the settings, with default values.
    std::string detectorMode = "STATIC";
    std::string detectorType = "STAR";
    if (featureOptions.contains("DetectorMode")) {
        detectorMode = boost::python::extract<std::string>(featureOptions.get("DetectorMode"));
    }
    if (featureOptions.contains("DetectorType")) {
        detectorType = boost::python::extract<std::string>(featureOptions.get("DetectorType"));
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
            adaptiveOptions = boost::python::extract<pybind11::dict>(featureOptions.get("Adaptive"));
        }
        
        // Defaults from the OpenCV documentation
        int minFeatures = 400;
        int maxFeatures = 500;
        int maxIters = 5;
        if (adaptiveOptions.contains("MinFeatures")) {
            minFeatures = boost::python::extract<int>(adaptiveOptions.get("MinFeatures"));
        }
        if (adaptiveOptions.contains("MaxFeatures")) {
            maxFeatures = boost::python::extract<int>(adaptiveOptions.get("MaxFeatures"));
        }
        if (adaptiveOptions.contains("MaxIters")) {
            maxIters = boost::python::extract<int>(adaptiveOptions.get("MaxIters"));
        }
        return cv::makePtr<cv::DynamicAdaptedFeatureDetector>(cv::AdjusterAdapter::create(detectorType), minFeatures, maxFeatures, maxIters);

    } else {
        pybind11::dict detectorOptions;
        if(detectorType == "FAST") {
            if (featureOptions.contains("FastDetector"))
            {
                detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("FastDetector"));
            }
            return createFAST(detectorOptions);
        } else if(detectorType == "SURF") {
            if (featureOptions.contains("SurfDetector"))
            {
                detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("SurfDetector"));
            }
            return createSURF(detectorOptions);
        } else if(detectorType == "SIFT") {
            if (featureOptions.contains("SiftDetector"))
            {
                detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("SiftDetector"));
            }
            return createSIFT(detectorOptions);
        } else if(detectorType == "MSER") {
            if (featureOptions.contains("MSERDetector"))
            {
                detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("MSERDetector"));
            }
            return createMSER(detectorOptions);
        } else {
            if (featureOptions.contains("StarDetector"))
            {
                detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("StarDetector"));
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
        numFeatures = boost::python::extract<int>(settings.get("NumFeatures"));
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.contains("ContrastThreshold")) {
        contrastThreshold = boost::python::extract<double>(settings.get("ContrastThreshold"));
    }
    if (settings.contains("EdgeThreshold")) {
        edgeThreshold = boost::python::extract<double>(settings.get("EdgeThreshold"));
    }
    if (settings.contains("Sigma")) {
        sigma = boost::python::extract<double>(settings.get("Sigma"));
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
        hessianThreshold = boost::python::extract<double>(settings.get("HessianThreshold"));
    }
    if (settings.contains("NumOctaves")) {
        nOctaves = boost::python::extract<int>(settings.get("NumOctaves"));
    }
    if (settings.contains("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.contains("Extended")) {
        extended = boost::python::extract<bool>(settings.get("Extended"));
    }
    if (settings.contains("Upright")) {
        upright = boost::python::extract<bool>(settings.get("Upright"));
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
        featureOptions = boost::python::extract<pybind11::dict>(settings.get("FeatureOptions"));
    }
    
    std::string extractorType = "SURF";
    if (featureOptions.contains("ExtractorType")) {
        extractorType = boost::python::extract<std::string>(featureOptions.get("ExtractorType"));
    }
    
    pybind11::dict detectorOptions;
    if(extractorType == "SIFT") {
        if (featureOptions.contains("SiftDetector"))
        {
            detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("SiftDetector"));
        }
        return createSIFTExtractor(detectorOptions);
    } else {
        if (featureOptions.contains("SurfDetector"))
        {
            detectorOptions = boost::python::extract<pybind11::dict>(featureOptions.get("SurfDetector"));
        }
        return createSURFExtractor(detectorOptions);
    }
}
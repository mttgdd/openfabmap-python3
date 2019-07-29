#ifndef DETECTORS_AND_EXTRACTORS_H
#define DETECTORS_AND_EXTRACTORS_H

#include <Python.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pybind11/pybind11.h>

namespace pyof2 {
cv::Ptr<cv::FeatureDetector> generateDetector(const pybind11::dict &settings);
cv::Ptr<cv::DescriptorExtractor>
generateExtractor(const pybind11::dict &settings);
} // namespace pyof2

#endif // DETECTORS_AND_EXTRACTORS_H

#include "ChowLiuTreeWrapper.h"
#include "FabMapVocabluary.h"
#include "openFABMAPPython.h"

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(openFABMAP, m) {

  PyEval_InitThreads();

  pybind11::class_<pyof2::FabMapVocabluary,
                   std::shared_ptr<pyof2::FabMapVocabluary>>(m, "Vocabluary");

  pybind11::class_<pyof2::FabMapVocabluaryBuilder,
                   std::shared_ptr<pyof2::FabMapVocabluaryBuilder>>(
      m, "VocabluaryBuilder")
      .def(pybind11::init<pybind11::dict>())
      .def("add_training_image",
           &pyof2::FabMapVocabluaryBuilder::addTrainingImage)
      .def("build_vocabluary",
           &pyof2::FabMapVocabluaryBuilder::buildVocabluary);

  pybind11::class_<pyof2::ChowLiuTreeWrapper, std::shared_ptr<pyof2::ChowLiuTreeWrapper>>(
      m, "ChowLiuTreeWrapper")
      .def(pybind11::init<std::shared_ptr<pyof2::FabMapVocabluary>,
                          pybind11::dict>())
      .def("add_training_image", &pyof2::ChowLiuTreeWrapper::addTrainingImage)
      .def("build_chow_liu_tree", &pyof2::ChowLiuTreeWrapper::buildChowLiuTree)
      .def("save", &pyof2::ChowLiuTreeWrapper::save)
      .def("load", &pyof2::ChowLiuTreeWrapper::load);

  pybind11::class_<pyof2::OpenFABMAPPython,
                   std::shared_ptr<pyof2::OpenFABMAPPython>>(m, "OpenFABMAPPython")
      .def(
          pybind11::init<std::shared_ptr<pyof2::ChowLiuTreeWrapper>, pybind11::dict>())
      .def("load_and_process_image",
           &pyof2::OpenFABMAPPython::loadAndProcessImage)
      .def("get_last_match", &pyof2::OpenFABMAPPython::getLastMatch)
      .def("get_all_loop_closures",
           &pyof2::OpenFABMAPPython::getAllLoopClosures);
}
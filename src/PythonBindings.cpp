#include "ChowLiuTree.h"
#include "FabMapVocabulary.h"
#include "openFABMAPPython.h"

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(openfabmap_python3, m) {

  PyEval_InitThreads();

  pybind11::class_<ofpy3::FabMapVocabulary,
                   std::shared_ptr<ofpy3::FabMapVocabulary>>(m, "Vocabulary");

  pybind11::class_<ofpy3::FabMapVocabularyBuilder,
                   std::shared_ptr<ofpy3::FabMapVocabularyBuilder>>(
      m, "VocabularyBuilder")
      .def(pybind11::init<pybind11::dict>())
      .def("add_training_image",
           &ofpy3::FabMapVocabularyBuilder::addTrainingImage)
      .def("load_and_add_training_image",
           &ofpy3::FabMapVocabularyBuilder::loadAndAddTrainingImage)
      .def("add_training_descs",
           &ofpy3::FabMapVocabularyBuilder::addTrainingDescs)
      .def("build_vocabulary",
           &ofpy3::FabMapVocabularyBuilder::buildVocabulary);

  pybind11::class_<ofpy3::ChowLiuTree, std::shared_ptr<ofpy3::ChowLiuTree>>(
      m, "ChowLiuTree")
      .def(pybind11::init<std::shared_ptr<ofpy3::FabMapVocabulary>,
                          pybind11::dict>())
      .def("add_training_image", &ofpy3::ChowLiuTree::addTrainingImage)
      .def("load_and_add_training_image",
           &ofpy3::ChowLiuTree::loadAndAddTrainingImage)
      .def("build_chow_liu_tree", &ofpy3::ChowLiuTree::buildChowLiuTree)
      .def("save", &ofpy3::ChowLiuTree::save)
      .def("load", &ofpy3::ChowLiuTree::load);

  pybind11::class_<ofpy3::OpenFABMAPPython,
                   std::shared_ptr<ofpy3::OpenFABMAPPython>>(m, "OpenFABMAP")
      .def(
          pybind11::init<std::shared_ptr<ofpy3::ChowLiuTree>, pybind11::dict>())
      .def("load_and_process_image",
           &ofpy3::OpenFABMAPPython::loadAndProcessImage)
      .def("process_image", &ofpy3::OpenFABMAPPython::ProcessImage)
      .def("get_last_match", &ofpy3::OpenFABMAPPython::getLastMatch)
      .def("get_best_loop_closures",
           &ofpy3::OpenFABMAPPython::getBestLoopClosures)
      .def("get_all_loop_closures",
           &ofpy3::OpenFABMAPPython::getAllLoopClosures);
}
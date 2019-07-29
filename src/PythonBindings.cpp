#include "ChowLiuTree.h"
#include "FabMapVocabulary.h"
#include "openFABMAPPython.h"

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(openFABMAP, m) {

  PyEval_InitThreads();

  pybind11::class_<pyof2::FabMapVocabulary,
                   std::shared_ptr<pyof2::FabMapVocabulary>>(m, "Vocabulary");

  pybind11::class_<pyof2::FabMapVocabularyBuilder,
                   std::shared_ptr<pyof2::FabMapVocabularyBuilder>>(
      m, "VocabularyBuilder")
      .def(pybind11::init<pybind11::dict>())
      .def("add_training_image",
           &pyof2::FabMapVocabularyBuilder::addTrainingImage)
      .def("load_and_add_training_image",
           &pyof2::FabMapVocabularyBuilder::loadAndAddTrainingImage)
      .def("build_vocabulary",
           &pyof2::FabMapVocabularyBuilder::buildVocabulary);

  pybind11::class_<pyof2::ChowLiuTree, std::shared_ptr<pyof2::ChowLiuTree>>(
      m, "ChowLiuTree")
      .def(pybind11::init<std::shared_ptr<pyof2::FabMapVocabulary>,
                          pybind11::dict>())
      .def("add_training_image", &pyof2::ChowLiuTree::addTrainingImage)
      .def("load_and_add_training_image",
           &pyof2::ChowLiuTree::loadAndAddTrainingImage)
      .def("build_chow_liu_tree", &pyof2::ChowLiuTree::buildChowLiuTree)
      .def("save", &pyof2::ChowLiuTree::save)
      .def("load", &pyof2::ChowLiuTree::load);

  pybind11::class_<pyof2::OpenFABMAPPython,
                   std::shared_ptr<pyof2::OpenFABMAPPython>>(m, "OpenFABMAP")
      .def(
          pybind11::init<std::shared_ptr<pyof2::ChowLiuTree>, pybind11::dict>())
      .def("load_and_process_image",
           &pyof2::OpenFABMAPPython::loadAndProcessImage)
      .def("process_image", &pyof2::OpenFABMAPPython::ProcessImage)
      .def("get_last_match", &pyof2::OpenFABMAPPython::getLastMatch)
      .def("get_all_loop_closures",
           &pyof2::OpenFABMAPPython::getAllLoopClosures);
}
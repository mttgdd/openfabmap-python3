#include "ChowLiuTree.h"
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
      .def("load_and_add_training_image",
           &pyof2::FabMapVocabluaryBuilder::loadAndAddTrainingImage)
      .def("build_vocabluary",
           &pyof2::FabMapVocabluaryBuilder::buildVocabluary);

  pybind11::class_<pyof2::ChowLiuTree, std::shared_ptr<pyof2::ChowLiuTree>>(
      m, "ChowLiuTree")
      .def(pybind11::init<std::shared_ptr<pyof2::FabMapVocabluary>,
                          pybind11::dict>())
      .def("add_training_image", &pyof2::ChowLiuTree::addTrainingImage)
      .def("load_and_add_training_image", &pyof2::ChowLiuTree::loadAndAddTrainingImage)
      .def("build_chow_liu_tree", &pyof2::ChowLiuTree::buildChowLiuTree)
      .def("save", &pyof2::ChowLiuTree::save)
      .def("load", &pyof2::ChowLiuTree::load);

  pybind11::class_<pyof2::OpenFABMAPPython,
                   std::shared_ptr<pyof2::OpenFABMAPPython>>(m, "OpenFABMAP")
      .def(
          pybind11::init<std::shared_ptr<pyof2::ChowLiuTree>, pybind11::dict>())
      .def("load_and_process_image",
           &pyof2::OpenFABMAPPython::loadAndProcessImage)
      .def("process_image",
           &pyof2::OpenFABMAPPython::ProcessImage)
      .def("get_last_match", &pyof2::OpenFABMAPPython::getLastMatch)
      .def("get_all_loop_closures",
           &pyof2::OpenFABMAPPython::getAllLoopClosures);
}
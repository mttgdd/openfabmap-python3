#include "ChowLiuTree.h"
#include "FabMapVocabluary.h"
#include "openFABMAPPython.h"

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// BOOST_PYTHON_MODULE(openFABMAP)
//{
//    boost::python::class_<pyof2::FabMapVocabluary,
//    std::shared_ptr<pyof2::FabMapVocabluary>>(
//            "Vocabluary", boost::python::no_init);
//    boost::python::register_ptr_to_python<std::shared_ptr<pyof2::FabMapVocabluary
//    > >();
//
//    boost::python::class_<pyof2::FabMapVocabluaryBuilder,
//    std::shared_ptr<pyof2::FabMapVocabluaryBuilder>>(
//            "VocabluaryBuilder", boost::python::init<boost::python::dict>())
//        .def("add_training_image",
//        &pyof2::FabMapVocabluaryBuilder::addTrainingImage)
//        .def("build_vocabluary",
//        &pyof2::FabMapVocabluaryBuilder::buildVocabluary);
//
//    boost::python::class_<pyof2::ChowLiuTree,
//    std::shared_ptr<pyof2::ChowLiuTree>>(
//            "ChowLiuTree",
//            boost::python::init<std::shared_ptr<pyof2::FabMapVocabluary>,
//            boost::python::dict>())
//        .def("add_training_image", &pyof2::ChowLiuTree::addTrainingImage)
//        .def("build_chow_liu_tree", &pyof2::ChowLiuTree::buildChowLiuTree)
//        .def("save", &pyof2::ChowLiuTree::save)
//        .def("load", &pyof2::ChowLiuTree::load)
//        .staticmethod("load");
//
//    boost::python::class_<pyof2::OpenFABMAPPython,
//    std::shared_ptr<pyof2::OpenFABMAPPython>>(
//            "OpenFABMAP", boost::python::init<
//                std::shared_ptr<pyof2::ChowLiuTree>,
//                boost::python::dict>())
//        .def("load_and_process_image",
//        &pyof2::OpenFABMAPPython::loadAndProcessImage) .def("get_last_match",
//        &pyof2::OpenFABMAPPython::getLastMatch) .def("get_all_loop_closures",
//        &pyof2::OpenFABMAPPython::getAllLoopClosures);
//}

class TestClass {
public:
  TestClass() = default;

public:
  void Test() { std::cout << "test!" << std::endl; }
};

PYBIND11_MODULE(openFABMAP, m) {

  PyEval_InitThreads();

  pybind11::class_<TestClass>(m, "TestClass")
      .def(pybind11::init<>())
      .def("Test", &TestClass::Test);
}
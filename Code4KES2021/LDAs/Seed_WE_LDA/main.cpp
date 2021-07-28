#include "lda.h"
#include "zlabels.h"
#include "seed_lda.h"
#include<pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(LDAs, m) {
	
	m.doc() = "..."; 

	py::class_<LDA>(m, "LDA").def(py::init()).def("setData", &LDA::SetData).def("train", &LDA::train).def("phi", &LDA::Phi).def("theta", &LDA::Theta).def("wordTopicMatrix", &LDA::WordTopicMatrix);

	py::class_<Zlabels>(m, "Zlabel").def(py::init()).def("setData", &Zlabels::SetData).def("train", &Zlabels::train).def("phi", &Zlabels::Phi).def("theta", &Zlabels::Theta).def("wordTopicMatrix", &Zlabels::WordTopicMatrix);

	py::class_<SeededLDA>(m, "SeededLDA").def(py::init()).def("setData", &SeededLDA::SetData).def("train", &SeededLDA::train).def("phi", &SeededLDA::Phi).def("theta", &SeededLDA::Theta).def("wordTopicMatrix", &SeededLDA::WordTopicMatrix);

}



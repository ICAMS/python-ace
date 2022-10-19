#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>

#include "ace-evaluator/ace_arraynd.h"
#include "ace-evaluator/ace_timing.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace/ace_b_evaluator.h"
#include "ace-evaluator/ace_recursive.h"
#include "ace-evaluator/ace_version.h"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::map<string, string>)


namespace py = pybind11;
using namespace std;

string get_ace_evaluator_version() {
    stringstream ss;
    ss << VERSION_YEAR << "." << VERSION_MONTH << "." << VERSION_DAY;
    return ss.str();
}

/*
ACEEvaluator has a derived ACECTildeEvaluator, both are bound to make pybind11
aware of the relationship.

NOte: There are many virtual overrides - one must be careful if these need
to be exposed. Then trampoline classes are need to properly do this. For now,
since the overridden methods are not used, it is fine.
*/
PYBIND11_MODULE(evaluator, m) {
    py::options options;
    options.disable_function_signatures();

    m.def("get_ace_evaluator_version", &get_ace_evaluator_version);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ACEEvaluator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//base class bindings
    py::class_<ACEEvaluator>(m, "ACEEvaluator")
            .def_property("element_type_mapping",
                          [](const ACEEvaluator &e) { return e.element_type_mapping.to_vector(); },
                          [](ACEEvaluator &e, vector<int> v) { e.element_type_mapping = v; });



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ACEBEvaluator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<ACEBEvaluator, ACEEvaluator>(m, "ACEBEvaluator")
            .def(py::init<>())
            .def(py::init<ACEBBasisSet &>(), py::arg("bBasisSet"))
            .def(py::init<BBasisConfiguration &>(), py::arg("bBasisConfiguration"))
            .def("load_active_set", &ACEBEvaluator::load_active_set, py::arg("asi_filename"),
                 py::arg("is_linear") = true, py::arg("is_auto_determine") = true)
            .def("set_active_set", &ACEBEvaluator::set_active_set, py::arg("species_type_active_set_inv"))
            .def("set_basis", &ACEBEvaluator::set_basis, R"mydelimiter(

    Set a basis to the evaluator

    Parameters
    ----------
    basis : ACECTildeBasisSet object

    Returns
    -------
    None

    )mydelimiter");


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ACECTildeEvaluator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<ACECTildeEvaluator, ACEEvaluator>(m, "ACECTildeEvaluator", R"mydelimiter(

    )mydelimiter")
            .def(py::init<>())
            .def(py::init<ACECTildeBasisSet &>(), py::arg("cTildeBasisSet"))
            .def("set_basis", &ACECTildeEvaluator::set_basis, R"mydelimiter(

    Set a basis to the evaluator

    Parameters
    ----------
    basis : ACECTildeBasisSet object

    Returns
    -------
    None

    )mydelimiter")
            .def_property("element_type_mapping",
                          [](const ACECTildeEvaluator &e) { return e.element_type_mapping.to_vector(); },
                          [](ACECTildeEvaluator &e, vector<int> v) { e.element_type_mapping = v; });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ACERecursiveEvaluator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<ACERecursiveEvaluator, ACEEvaluator>(m, "ACERecursiveEvaluator")
            .def(py::init<>())
            .def(py::init<ACECTildeBasisSet &>())
            .def("set_recursive", &ACERecursiveEvaluator::set_recursive)
            .def("set_basis", &ACERecursiveEvaluator::set_basis,
                 py::arg("cTildeBasisSet"),
                 py::arg("heuristic") = 0);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


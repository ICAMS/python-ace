//
// Created by Lysogorskiy Yury on 28.06.24.
//


#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "ace/grace_fs_evaluator.h"
#include "extra/grace_fs_calculator.h"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(grace_fs, m) {
    py::options options;
    options.disable_function_signatures();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRACEFSBasisSet
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<GRACEFSBasisSet>(m, "GRACEFSBasisSet")
            .def(py::init<>())
            .def(py::init<string>(), py::arg("yaml_file_name"))
            .def_readonly("nelements", &GRACEFSBasisSet::nelements)
            .def_property_readonly("elements_name", [](const GRACEFSBasisSet &b) {
                vector<string> res(b.nelements);
                for (int i = 0; i < b.nelements; i++)
                    res[i] = b.elements_name[i];
                return res;
            })
            .def_readonly("elements_to_index_map", &GRACEFSBasisSet::elements_to_index_map)
            .def_readonly("ndensitymax", &GRACEFSBasisSet::ndensitymax)
            .def_readonly("nradbase", &GRACEFSBasisSet::nradbase)
            .def_readonly("lmax", &GRACEFSBasisSet::lmax)
            .def_readonly("nradmax", &GRACEFSBasisSet::nradmax)
            .def_readonly("cutoffmax", &GRACEFSBasisSet::cutoffmax)
            .def_property_readonly("nfuncs", [](const GRACEFSBasisSet &b) {
                vector<int> res(b.nelements);
                for (int i = 0; i < b.nelements; i++)
                    res[i] = b.basis[i].size();
                return res;
            });



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRACEFSBEvaluator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//base class bindings
    py::class_<GRACEFSBEvaluator>(m, "GRACEFSBEvaluator")
            .def_property("element_type_mapping",
                          [](const GRACEFSBEvaluator &e) { return e.element_type_mapping.to_vector(); },
                          [](GRACEFSBEvaluator &e, vector<int> v) { e.element_type_mapping = v; })
            .def(py::init<>())
            .def("set_basis", &GRACEFSBEvaluator::set_basis, R"delim(

    Set a basis to the evaluator

    Parameters
    ----------
    basis : GRACEFSBasisSet object

    Returns
    -------
    None

    )delim")
            .def("load_active_set", &GRACEFSBEvaluator::load_active_set, py::arg("asi_filename"))
    ;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRACEFSCalculator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<GRACEFSCalculator>(m, "GRACEFSCalculator", R"mydelimiter(

    )mydelimiter")
            .def(py::init<GRACEFSBEvaluator &>(), py::arg("GRACEFSCalculator"))
//            .def("set_evaluator", &ACECalculator::set_evaluator)
            .def("compute", &GRACEFSCalculator::compute, py::arg("atomic_environment"), py::arg("compute_projections") = false, py::arg("verbose") = false)
            .def_property_readonly("forces", [](const GRACEFSCalculator &calc) { return calc.forces.to_vector(); })
            .def_readonly("energy", &GRACEFSCalculator::energy)
            .def_property_readonly("energies", [](const GRACEFSCalculator &calc) { return calc.energies.to_vector(); })
            .def_property_readonly("virial", [](const GRACEFSCalculator &calc) { return calc.virial.to_vector(); })
            .def_readonly("projections", &GRACEFSCalculator::projections)
            .def_readonly("gamma_grade",&GRACEFSCalculator::gamma_grade)
//            .def_readonly("rhos", &ACECalculator::rhos)
//            .def_readonly("dF_drhos", &ACECalculator::dF_drhos)
//            .def_readonly("dE_dc", &ACECalculator::dE_dc)
//            .def_readonly("forces_bfuncs",&ACECalculator::forces_bfuncs)
            ;
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


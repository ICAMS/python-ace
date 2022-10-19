//
// Created by Lysogorskiy Yury on 11.05.2020.
//
#include "ace_radial_helper.h"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "ace-evaluator/ace_radial.h"
#include "ace-evaluator/ace_types.h"


namespace py = pybind11;
using namespace std;


py::tuple ACERadialFunctions_getstate(const AbstractRadialBasis *radial_functions) {
    return py::make_tuple(
            radial_functions->nradbase,              //0
            radial_functions->lmax,                  //1
            radial_functions->nradial,               //2
            radial_functions->deltaSplineBins,       //3
            radial_functions->nelements,             //4
            radial_functions->prehc.to_vector(),     //5
            radial_functions->lambdahc.to_vector(),  //6
            radial_functions->lambda.to_vector(),    //7
            radial_functions->cut.to_vector(),       //8
            radial_functions->dcut.to_vector(),      //9
            radial_functions->crad.to_vector(),       //10
            radial_functions->radbasenameij.to_vector(),     //11
            radial_functions->inner_cutoff_type, //12
            radial_functions->cut_in.to_vector(), //13
            radial_functions->dcut_in.to_vector() //14
    );
}

ACERadialFunctions *ACERadialFunctions_setstate(const py::tuple &t) {
    if (t.size() != 15)
        throw std::runtime_error("Invalid state of ACERadialFunctions-tuple");

    NS_TYPE nradbase = t[0].cast<NS_TYPE>();        //0
    LS_TYPE lmax = t[1].cast<LS_TYPE>();                 //1
    NS_TYPE nradial = t[2].cast<NS_TYPE>();               //2
    DOUBLE_TYPE deltaSplineBins = t[3].cast<DOUBLE_TYPE>();                  //3
    SPECIES_TYPE nelements = t[4].cast<SPECIES_TYPE>();             //4

    auto prehc = t[5].cast<vector<vector<DOUBLE_TYPE>>>();     //5
    auto lambdahc = t[6].cast<vector<vector<DOUBLE_TYPE>>>();  //6
    auto lambda = t[7].cast<vector<vector<DOUBLE_TYPE>>>();    //7
    auto cut = t[8].cast<vector<vector<DOUBLE_TYPE>>>();//8
    auto dcut = t[9].cast<vector<vector<DOUBLE_TYPE>>>();//9
    auto crad = t[10].cast<vector<vector<vector<vector<vector<DOUBLE_TYPE>>>>>>();       //10
    auto radbasename = t[11].cast<vector<vector<string>>>();            // 11
    auto inner_cutoff_type = t[12].cast<string>(); // 12
    auto cut_in = t[13].cast<vector<vector<DOUBLE_TYPE>>>(); // 13
    auto dut_in = t[14].cast<vector<vector<DOUBLE_TYPE>>>(); // 13

    ACERadialFunctions *radial_functions = new ACERadialFunctions(nradbase, lmax, nradial, deltaSplineBins, nelements,
                                                                  radbasename);

    radial_functions->prehc = prehc;
    radial_functions->lambdahc = lambdahc;
    radial_functions->lambda = lambda;
    radial_functions->cut = cut;
    radial_functions->dcut = dcut;
    radial_functions->crad = crad;
    radial_functions->inner_cutoff_type=inner_cutoff_type;
    radial_functions->cut_in = cut_in;
    radial_functions->dcut_in = dut_in;

    radial_functions->setuplookupRadspline();
    return radial_functions;
}
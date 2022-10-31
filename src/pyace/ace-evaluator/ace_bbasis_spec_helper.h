//
// Created by Lysogorskiy Yury on 11.05.2020.
//

#ifndef PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H
#define PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ace/ace_b_basis.h"
#include "ace-evaluator/ace_utils.h"

namespace py = pybind11;
using namespace std;

string BBasisFunctionsSpecificationBlock_repr_(const BBasisFunctionsSpecificationBlock &block);

string BBasisConfiguration_repr(BBasisConfiguration &config);

py::tuple BBasisFunctionSpecification_getstate(const BBasisFunctionSpecification &spec);

BBasisFunctionSpecification BBasisFunctionSpecification_setstate(const py::tuple &t);

py::tuple BBasisFunctionsSpecificationBlock_getstate(const BBasisFunctionsSpecificationBlock &block);

BBasisFunctionsSpecificationBlock BBasisFunctionsSpecificationBlock_setstate(const py::tuple &tuple);

py::tuple BBasisConfiguration_getstate(const BBasisConfiguration &config);

BBasisConfiguration BBasisConfiguration_setstate(const py::tuple &tuple);


// ACEBBasisSet pickling
py::tuple ACEBBasisSet_getstate(const ACEBBasisSet &bbasisSet);

ACEBBasisSet ACEBBasisSet_setstate(const py::tuple &tuple);

#endif //PYACE_ACE_BBASIS_FUNC_SPEC_HELPER_H

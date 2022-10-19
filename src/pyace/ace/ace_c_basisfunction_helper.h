//
// Created by Lysogorskiy Yury on 11.05.2020.
//

#ifndef PYACE_ACE_C_BASISFUNCTION_HELPER_H
#define PYACE_ACE_C_BASISFUNCTION_HELPER_H
#include <pybind11/pybind11.h>

#include "ace-evaluator/ace_c_basisfunction.h"
#include "ace/ace_b_basisfunction.h"

namespace py = pybind11;
using namespace std;

vector<SPECIES_TYPE> get_mus(const ACEAbstractBasisFunction &func );
vector<NS_TYPE> get_ns(const ACEAbstractBasisFunction &func );
vector<LS_TYPE> get_ls(const ACEAbstractBasisFunction &func );
vector<vector<MS_TYPE>> get_ms_combs(const ACEAbstractBasisFunction &func );

vector<DOUBLE_TYPE> get_gen_cgs(const ACEBBasisFunction &func );
vector<DOUBLE_TYPE> get_coeff(const ACEBBasisFunction &func );
vector<LS_TYPE> get_LS(const ACEBBasisFunction &func );

vector<vector<DOUBLE_TYPE>> get_ctildes(const ACECTildeBasisFunction &func );

py::tuple ACECTildeBasisFunction_getstate(const ACECTildeBasisFunction &func);
ACECTildeBasisFunction ACECTildeBasisFunction_setstate(const py::tuple &tuple);



#endif //PYACE_ACE_C_BASISFUNCTION_HELPER_H

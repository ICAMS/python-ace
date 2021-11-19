//
// Created by Lysogorskiy Yury on 11.05.2020.
//

#ifndef PYACE_ACE_C_BASIS_HELPER_H
#define PYACE_ACE_C_BASIS_HELPER_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ace_c_basisfunction.h"
#include "ace_c_basis.h"
#include "ace_b_basis.h"

namespace py = pybind11;
using namespace std;

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis_rank1(const ACEBBasisSet &basis);

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis(const ACEBBasisSet &basis);


vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis_rank1(const ACECTildeBasisSet &basis);

vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis(const ACECTildeBasisSet &basis);

py::tuple ACECTildeBasisSet_getstate(const ACECTildeBasisSet &cbasisSet);

ACECTildeBasisSet ACECTildeBasisSet_setstate(const py::tuple &t);


py::tuple ACEEmbeddingSpecification_getstate(const ACEEmbeddingSpecification &embeddingSpecification);

ACEEmbeddingSpecification ACEEmbeddingSpecification_setstate(const py::tuple &t);


py::tuple ACEBondSpecification_getstate(const ACEBondSpecification &bondSpecification);

ACEBondSpecification ACEBondSpecification_setstate(const py::tuple &t);


#endif //PYACE_ACE_C_BASIS_HELPER_H

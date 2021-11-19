//
// Created by Lysogorskiy Yury on 11.05.2020.
//

#include "ace_c_basisfunction_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>



namespace py = pybind11;
using namespace std;

vector<SPECIES_TYPE> get_mus(const ACEAbstractBasisFunction &func ) {
    vector<SPECIES_TYPE> mus(func.rank);
    for (RANK_TYPE r = 0; r < func.rank; ++r) mus[r] = func.mus[r];
    return mus;
}

vector<NS_TYPE> get_ns(const ACEAbstractBasisFunction &func ) {
    vector<NS_TYPE> ns(func.rank);
    for (RANK_TYPE r = 0; r < func.rank; ++r) ns[r] = func.ns[r];
    return ns;
}

vector<LS_TYPE> get_ls(const ACEAbstractBasisFunction &func ) {
    vector<LS_TYPE> ls(func.rank);
    for (RANK_TYPE r = 0; r < func.rank; ++r) ls[r] = func.ls[r];
    return ls;
}

vector<LS_TYPE> get_LS(const ACEBBasisFunction &func ) {
    vector<LS_TYPE> LS(func.rankL);
    for (RANK_TYPE r = 0; r < func.rankL; ++r) LS[r] = func.LS[r];
    return LS;
}

vector<vector<MS_TYPE>> get_ms_combs(const ACEAbstractBasisFunction &func ) {
    vector<vector<MS_TYPE>> ms_combs(func.num_ms_combs);
    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++){
        ms_combs[m_ind].resize(func.rank);
        for (RANK_TYPE r = 0; r < func.rank; ++r)
            ms_combs[m_ind][r] = func.ms_combs[m_ind * func.rank + r];
    }
    return ms_combs;
}

vector<DOUBLE_TYPE> get_gen_cgs(const ACEBBasisFunction &func ) {
    vector<DOUBLE_TYPE> gen_cgs(func.num_ms_combs);
    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++){
        gen_cgs[m_ind] = func.gen_cgs[m_ind];
    }
    return gen_cgs;
}

vector<DOUBLE_TYPE> get_coeff(const ACEBBasisFunction &func ) {
    vector<DOUBLE_TYPE> coeff(func.ndensity);
    for(DENSITY_TYPE p = 0; p < func.ndensity; p++)
        coeff[p] = func.coeff[p];
    return coeff;
}

vector<vector<DOUBLE_TYPE>> get_ctildes(const ACECTildeBasisFunction &func ) {
    vector<vector<DOUBLE_TYPE>> ctildes(func.num_ms_combs);
    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++){
        ctildes[m_ind].resize(func.ndensity);
        for(DENSITY_TYPE p = 0; p < func.ndensity; p++)
            ctildes[m_ind][p] = func.ctildes[m_ind * func.ndensity + p];
    }
    return ctildes;
}

py::tuple ACECTildeBasisFunction_getstate(const ACECTildeBasisFunction &func) {
    return py::make_tuple(
            (int)func.rank, //0
            func.ndensity, //1
            func.mu0, //2
            get_mus(func), //3
            get_ns(func),//4
            get_ls(func), //5
            get_ms_combs(func), //6
            get_ctildes(func), //7
            func.is_half_ms_basis //8
    );
}

ACECTildeBasisFunction ACECTildeBasisFunction_setstate(const py::tuple &tuple) {
    if (tuple.size()!=9)
        throw std::runtime_error("Invalid state of ACECTildeBasisFunction-tuple");
    RANK_TYPE r;
    ACECTildeBasisFunction func;
    func.rank = tuple[0].cast<int>();
    func.ndensity = tuple[1].cast<DENSITY_TYPE>();
    func.mu0 = tuple[2].cast<SPECIES_TYPE>();
    vector<SPECIES_TYPE> mus = tuple[3].cast<vector<SPECIES_TYPE>>();
    vector<NS_TYPE> ns= tuple[4].cast<vector<NS_TYPE>>();
    vector<LS_TYPE> ls= tuple[5].cast<vector<LS_TYPE>>();
    vector<vector<MS_TYPE>> ms_combs= tuple[6].cast<vector<vector<MS_TYPE>>>();
    vector<vector<DOUBLE_TYPE>> ctildes= tuple[7].cast<vector<vector<DOUBLE_TYPE>>>();
    func.is_half_ms_basis= tuple[8].cast<bool>();

    func.mus = new SPECIES_TYPE [func.rank];
    for (r = 0; r < func.rank; ++r) func.mus[r] = mus[r];

    func.ns = new NS_TYPE[func.rank];
    for (r = 0; r < func.rank; ++r) func.ns[r] = ns[r];

    func.ls = new LS_TYPE[func.rank];
    for (r = 0; r < func.rank; ++r) func.ls[r]=ls[r];

    func.num_ms_combs = ms_combs.size();
    func.ms_combs = new MS_TYPE [func.num_ms_combs * func.rank];
    func.ctildes = new DOUBLE_TYPE[func.num_ms_combs * func.ndensity];

    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++){
        for (r = 0; r < func.rank; ++r)
            func.ms_combs[m_ind * func.rank + r] =  ms_combs[m_ind][r];

        for(DENSITY_TYPE p = 0; p < func.ndensity; p++)
            func.ctildes[m_ind * func.ndensity + p] = ctildes[m_ind][p];
    }

    return func;
}


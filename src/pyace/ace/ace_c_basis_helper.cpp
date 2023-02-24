//
// Created by Lysogorskiy Yury on 11.05.2020.
//
#include "ace_c_basis_helper.h"
#include "ace/ace_b_basisfunction.h"
#include "ace_radial_helper.h"


namespace py = pybind11;
using namespace std;

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis_rank1(const ACEBBasisSet &basis) {
    vector<vector<ACEBBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size_rank1[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE func_ind = 0; func_ind < size; func_ind++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis_rank1[mu][func_ind].is_proxy;
            basis.basis_rank1[mu][func_ind].is_proxy = false;
            res[mu][func_ind] = basis.basis_rank1[mu][func_ind];
            basis.basis_rank1[mu][func_ind].is_proxy = old_proxy;
        }
    }
    return res;
}

vector<vector<ACEBBasisFunction>> ACEBBasisSet_get_basis(const ACEBBasisSet &basis) {
    vector<vector<ACEBBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis[mu][t].is_proxy;
            basis.basis[mu][t].is_proxy = false;
            res[mu][t] = basis.basis[mu][t];
            basis.basis[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}


vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis_rank1(const ACECTildeBasisSet &basis) {
    vector<vector<ACECTildeBasisFunction>> res;

    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size_rank1[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis_rank1[mu][t].is_proxy;
            basis.basis_rank1[mu][t].is_proxy = false;
            res[mu][t] = basis.basis_rank1[mu][t];
            basis.basis_rank1[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}

vector<vector<ACECTildeBasisFunction>> ACECTildeBasisSet_get_basis(const ACECTildeBasisSet &basis) {
    vector<vector<ACECTildeBasisFunction>> res;
    res.resize(basis.nelements);
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        SHORT_INT_TYPE size = basis.total_basis_size[mu];
        res[mu].resize(size);
        for (SHORT_INT_TYPE t = 0; t < size; t++) {
            //turn off the proxying for proper copying when export to Python
            bool old_proxy = basis.basis[mu][t].is_proxy;
            basis.basis[mu][t].is_proxy = false;
            res[mu][t] = basis.basis[mu][t];
            basis.basis[mu][t].is_proxy = old_proxy;
        }
    }
    return res;
}

py::tuple ACECTildeBasisSet_getstate(const ACECTildeBasisSet &cbasisSet) {
    vector<string> elements_name(cbasisSet.nelements);
    for (SPECIES_TYPE mu = 0; mu < cbasisSet.nelements; ++mu)
        elements_name[mu] = cbasisSet.elements_name[mu];

    auto tuple = py::make_tuple(
            cbasisSet.lmax,  //0
            cbasisSet.nradbase, //1
            cbasisSet.nradmax, //2
            cbasisSet.nelements, //3
            cbasisSet.rankmax, //4
            cbasisSet.ndensitymax, //5
            cbasisSet.cutoffmax, //6
            cbasisSet.deltaSplineBins, //7
            cbasisSet.map_embedding_specifications, //8
            ACERadialFunctions_getstate(cbasisSet.radial_functions), //9
            elements_name, //10
            ACECTildeBasisSet_get_basis_rank1(cbasisSet), //11
            ACECTildeBasisSet_get_basis(cbasisSet),//12
            cbasisSet.E0vals.to_vector(), //13,
            cbasisSet.map_bond_specifications //14
    );
    return tuple;
}

ACECTildeBasisSet ACECTildeBasisSet_setstate(const py::tuple &t) {
    if (t.size() != 15)
        throw std::runtime_error("Invalid state of ACECTildeBasisSet-tuple");

    ACECTildeBasisSet new_cbasis;
    new_cbasis.lmax = t[0].cast<LS_TYPE>();  //0
    new_cbasis.nradbase = t[1].cast<NS_TYPE>(); //1
    new_cbasis.nradmax = t[2].cast<NS_TYPE>(); //2
    new_cbasis.nelements = t[3].cast<SPECIES_TYPE>(); //3
    new_cbasis.rankmax = t[4].cast<RANK_TYPE>(); //4
    new_cbasis.ndensitymax = t[5].cast<DENSITY_TYPE>(); //5
    new_cbasis.cutoffmax = t[6].cast<DOUBLE_TYPE>(); //6
    new_cbasis.deltaSplineBins = t[7].cast<DOUBLE_TYPE>(); //7
    new_cbasis.map_embedding_specifications = t[8].cast < map < SPECIES_TYPE, ACEEmbeddingSpecification >> ();//8
    new_cbasis.spherical_harmonics.init(new_cbasis.lmax);
    new_cbasis.radial_functions = ACERadialFunctions_setstate(t[9].cast<py::tuple>()); //9
    auto elements_name = t[10].cast<vector<string>>(); //10
    auto basis_rank1 = t[11].cast<vector<vector<ACECTildeBasisFunction>>>(); //11
    auto basis = t[12].cast<vector<vector<ACECTildeBasisFunction>>>(); //12
    new_cbasis.E0vals = t[13].cast<vector<DOUBLE_TYPE>>();//13

    new_cbasis.map_bond_specifications = t[14].cast < map < pair < SPECIES_TYPE, SPECIES_TYPE >, ACEBondSpecification >
                                                                                                 > ();//14

    new_cbasis.elements_name = new string[elements_name.size()];
    for (int i = 0; i < elements_name.size(); i++) {
        new_cbasis.elements_name[i] = elements_name[i];
    }

    new_cbasis.total_basis_size_rank1 = new int[new_cbasis.nelements];
    new_cbasis.basis_rank1 = new ACECTildeBasisFunction *[new_cbasis.nelements];
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; ++mu) {
        SHORT_INT_TYPE size = basis_rank1[mu].size();
        new_cbasis.total_basis_size_rank1[mu] = size;
        new_cbasis.basis_rank1[mu] = new ACECTildeBasisFunction[size];
    }
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < new_cbasis.total_basis_size_rank1[mu]; ++func_ind) {
            new_cbasis.basis_rank1[mu][func_ind] = basis_rank1[mu][func_ind];
        }

    new_cbasis.total_basis_size = new int[new_cbasis.nelements];
    new_cbasis.basis = new ACECTildeBasisFunction *[new_cbasis.nelements];
    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; ++mu) {
        SHORT_INT_TYPE size = basis[mu].size();
        new_cbasis.total_basis_size[mu] = size;
        new_cbasis.basis[mu] = new ACECTildeBasisFunction[size];
    }

    for (SPECIES_TYPE mu = 0; mu < new_cbasis.nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < new_cbasis.total_basis_size[mu]; ++func_ind) {
            new_cbasis.basis[mu][func_ind] = basis[mu][func_ind];
        }

    new_cbasis.pack_flatten_basis();
    return new_cbasis;
}


py::tuple ACEEmbeddingSpecification_getstate(const ACEEmbeddingSpecification &spec) {
    return py::make_tuple(spec.ndensity,//0
                          spec.FS_parameters,//1
                          spec.npoti,//2
                          spec.rho_core_cutoff,//3
                          spec.drho_core_cutoff//4
    );
}

ACEEmbeddingSpecification ACEEmbeddingSpecification_setstate(const py::tuple &t) {
    if (t.size() != 5)
        throw std::runtime_error("Invalid state of ACEEmbeddingSpecification-tuple");
    ACEEmbeddingSpecification spec;
    spec.ndensity = t[0].cast<DENSITY_TYPE>();
    spec.FS_parameters = t[1].cast<vector<DOUBLE_TYPE>>();
    spec.npoti = t[2].cast<string>();
    spec.rho_core_cutoff = t[3].cast<DOUBLE_TYPE>();
    spec.drho_core_cutoff = t[4].cast<DOUBLE_TYPE>();
    return spec;
}


py::tuple ACEBondSpecification_getstate(const ACEBondSpecification &spec) {
    return py::make_tuple(
            spec.nradmax,           //0
            spec.lmax,              //1
            spec.nradbasemax,       //2

            spec.radbasename,       //3
            spec.radparameters,     //4
            spec.radcoefficients,   //5
            spec.prehc,             //6
            spec.lambdahc,          //7
            spec.rcut,              //8
            spec.dcut,              //9

            spec.rcut_in,           //10
            spec.dcut_in,           //11
            spec.inner_cutoff_type  //12
    );
}

ACEBondSpecification ACEBondSpecification_setstate(const py::tuple &t) {
    if (t.size() != 13)
        throw std::runtime_error("Invalid state of ACEBondSpecification-tuple");
    ACEBondSpecification spec;

    spec.nradmax = t[0].cast<NS_TYPE>();           //0
    spec.lmax = t[1].cast<LS_TYPE>();               //1
    spec.nradbasemax = t[2].cast<NS_TYPE>();        //2

    spec.radbasename = t[3].cast<string>();        //3
    spec.radparameters = t[4].cast<vector<DOUBLE_TYPE>>();      //4
    spec.radcoefficients = t[5].cast<vector<vector<vector<DOUBLE_TYPE>>>>();    //5
    spec.prehc = t[6].cast<DOUBLE_TYPE>();              //6
    spec.lambdahc = t[7].cast<DOUBLE_TYPE>();           //7
    spec.rcut = t[8].cast<DOUBLE_TYPE>();               //8
    spec.dcut = t[9].cast<DOUBLE_TYPE>();                //9

    spec.rcut_in = t[10].cast<DOUBLE_TYPE>();           //10
    spec.dcut_in = t[11].cast<DOUBLE_TYPE>();           //11
    spec.inner_cutoff_type = t[12].cast<string>();      //12

    return spec;
}
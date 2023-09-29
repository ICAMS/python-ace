//
// Created by Yury Lysogorskiy on 31.01.20.
//

#ifndef ACE_B_EVALUATOR_H
#define ACE_B_EVALUATOR_H


#include "ace-evaluator/ace_arraynd.h"
#include "ace-evaluator/ace_array2dlm.h"
#include "ace/ace_b_basis.h"
#include "ace-evaluator/ace_complex.h"
#include "ace-evaluator/ace_timing.h"
#include "ace-evaluator/ace_types.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_abstract_basis.h"

class ACEBEvaluator : public ACEEvaluator {

    Array2D<DOUBLE_TYPE> weights_rank1 = Array2D<DOUBLE_TYPE>("weights_rank1");
    Array4DLM<ACEComplex> weights = Array4DLM<ACEComplex>("weights");

#ifdef COMPUTE_B_GRAD
    // for B-derivatives
    Array3D<DOUBLE_TYPE> weights_rank1_dB = Array3D<DOUBLE_TYPE>("weights_rank1_dB");
    Array5DLM<ACEComplex> weights_dB = Array5DLM<ACEComplex>("weights_dB");
#endif
    //cache for grads: grad_phi(jj,n)=A2DLM(l,m)
    //(neigh_jj)(n=0..nr-1)
    Array2D<DOUBLE_TYPE> DG_cache = Array2D<DOUBLE_TYPE>("DG_cache");
    // (neigh_jj)(n=0..nr-1,l)
    Array3D<DOUBLE_TYPE> R_cache = Array3D<DOUBLE_TYPE>("R_cache");
    Array3D<DOUBLE_TYPE> DR_cache = Array3D<DOUBLE_TYPE>("DR_cache");
    // (neigh_jj)(l,m)
    Array3DLM<ACEComplex> Y_cache = Array3DLM<ACEComplex>("Y_cache");
    Array3DLM<ACEDYcomponent> DY_cache = Array3DLM<ACEDYcomponent>("dY_dense_cache");

    //hard-core repulsion
    //(neigh_jj)
    Array1D<DOUBLE_TYPE> DCR_cache = Array1D<DOUBLE_TYPE>("DCR_cache");


    Array1D<ACEComplex> dB_flatten = Array1D<ACEComplex>("dB_flatten");

    //pointer to the ACEBasisSet object
    ACEBBasisSet *basis_set = nullptr;

    ACEBBasisSet _basis_set;

    void init(ACEBBasisSet *basis_set);

    // active sets
    map<SPECIES_TYPE, Array2D<DOUBLE_TYPE>> A_active_set_inv;

    bool is_linear_extrapolation_grade = true;

    void resize_projections();

    void validate_ASI_square_shape(SPECIES_TYPE st, const vector<size_t> &shape);

    void validate_ASI_shape(const string &element_name, SPECIES_TYPE st, const vector<size_t> &shape);

public:

    ACEBEvaluator() = default;

    explicit ACEBEvaluator(ACEBBasisSet &bas) {
        set_basis(bas);
    }

    explicit ACEBEvaluator(BBasisConfiguration &bBasisConfiguration) {
        _basis_set.initialize_basis(bBasisConfiguration);
        set_basis(_basis_set);
    }

    //set the basis function to the ACE evaluator
    void set_basis(ACEBBasisSet &bas);

    //compute the energy and forces for atom_i
    //x - atomic positions [atom_ind][3]
    //type - atomic types [atom_ind]
    //jnum - number of neighbours of atom_i
    //jlist - list of neighbour indices. Indices are for arrays a and type
    //this will also update the energies(i) and neighbours_forces(jj, alpha) arrays
    void compute_atom(int i, DOUBLE_TYPE **x, const SPECIES_TYPE *type, const int jnum, const int *jlist) override;

    void resize_neighbours_cache(int max_jnum) override;

    void load_active_set(const string &asi_filename, bool is_linear = true, bool is_auto_determine = true);

    void set_active_set(const vector<vector<vector<DOUBLE_TYPE>>> &species_type_active_set_inv);

    bool get_is_linear_extrapolation_grade() { return this->is_linear_extrapolation_grade; }

    vector<int> get_func_ind_shift() override;

    int get_total_number_of_functions() override;

    vector<int> get_number_of_functions() override;
};


#endif //ACE_B_EVALUATOR_H

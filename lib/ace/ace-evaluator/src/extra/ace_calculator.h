//
// Created by Yury Lysogorskiy on 13.03.2020.
//

#ifndef ACE_CALCULATOR_H
#define ACE_CALCULATOR_H

#include "ace-evaluator/ace_evaluator.h"
#include "extra/ace_atoms.h"

class ACECalculator {
    ACEEvaluator *evaluator = nullptr;
public:
    //total energy of ACEAtomicEnvironment
    DOUBLE_TYPE energy = 0;
    //total forces array
    //forces(i,3), i = 0..num_of_atoms-1
    Array2D<DOUBLE_TYPE> forces = Array2D<DOUBLE_TYPE>("forces");

    //stresses
    Array1D<DOUBLE_TYPE> virial = Array1D<DOUBLE_TYPE>(6, "virial");

    //Per-atom energies
    //energies(i), i = 0..num_of_atoms-1
    Array1D<DOUBLE_TYPE> energies = Array1D<DOUBLE_TYPE>("energies");

    ACECalculator() = default;

    ACECalculator(ACEEvaluator &aceEvaluator) {
        set_evaluator(aceEvaluator);
    }
    void set_evaluator(ACEEvaluator &aceEvaluator);

    //compute the energies and forces for each atoms in atomic_environment
    //results are stored in forces and energies arrays
    void compute(ACEAtomicEnvironment &atomic_environment,  bool compute_projections = false, bool compute_b_grad = false, bool verbose = false);
#ifdef EXTRA_C_PROJECTIONS
    vector<vector<DOUBLE_TYPE>> projections;
    vector<vector<DOUBLE_TYPE>> rhos;
    vector<vector<DOUBLE_TYPE>> dF_drhos;
    vector<vector<DOUBLE_TYPE>> dE_dc;
    vector<DOUBLE_TYPE> gamma_grade;
#endif
#ifdef COMPUTE_B_GRAD
    vector<vector<vector<DOUBLE_TYPE>>> forces_bfuncs;
    //shape [natoms; ntotal_funcs (over all species types); 3]
#endif
};


#endif //ACE_CALCULATOR_H

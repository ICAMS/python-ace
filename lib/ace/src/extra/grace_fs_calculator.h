//
// Created by Yury Lysogorskiy on 31.05.23.
//

#ifndef GRACE_FS_CALCULATOR_H
#define GRACE_FS_CALCULATOR_H

#include "ace/grace_fs_evaluator.h"
#include "extra/ace_atoms.h"

class GRACEFSCalculator {
    GRACEFSBEvaluator &evaluator;
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

    explicit GRACEFSCalculator(GRACEFSBEvaluator &aceEvaluator) : evaluator(aceEvaluator) {};


    //compute the energies and forces for each atoms in atomic_environment
    //results are stored in forces and energies arrays
    void compute(ACEAtomicEnvironment &atomic_environment, bool compute_projections = false, bool verbose = false);
#ifdef EXTRA_C_PROJECTIONS
    vector<vector<DOUBLE_TYPE>> projections;
//    vector<vector<DOUBLE_TYPE>> rhos;
//    vector<vector<DOUBLE_TYPE>> dF_drhos;
//    vector<vector<DOUBLE_TYPE>> dE_dc;
    vector<DOUBLE_TYPE> gamma_grade;
#endif
};

#endif //GRACE_FS_CALCULATOR_H

//
// Created by Yury Lysogorskiy on 22.02.20.
//

#ifndef ACE_UTILS_TEST_H
#define ACE_UTILS_TEST_H

#include <cmath>
#include "ace_types.h"
#include "ace_atoms.h"
#include "ace_evaluator.h"
#include "ace_utils.h"
#include "ace_calculator.h"


void print_input_structure_for_fortran(ACEAtomicEnvironment &atomic_environment);

void check_sum_of_forces(ACEAtomicEnvironment &ae, ACECalculator &aceCalculator, DOUBLE_TYPE threshold = 1e-10);

void check_cube_diagonal_forces_symmetry(ACEAtomicEnvironment &ae, ACECalculator &ace);

void compare_forces(DOUBLE_TYPE analytic_force, DOUBLE_TYPE numeric_force, DOUBLE_TYPE rel_threshold);

void check_numeric_force(ACEAtomicEnvironment &ae, ACECalculator &ace, DOUBLE_TYPE rel_threshold = 1e-5,
                         DOUBLE_TYPE dr = 1e-8,
                         int atom_ind_freq = 10);

#endif //ACE_UTILS_TEST_H

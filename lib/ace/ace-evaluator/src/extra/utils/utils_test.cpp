#include "gtest/gtest.h"
#include "extra/utils/utils_test.h"
#include "extra/ace_calculator.h"


void compare_forces(DOUBLE_TYPE analytic_force, DOUBLE_TYPE numeric_force, DOUBLE_TYPE rel_threshold) {
    ASSERT_LE(absolute_relative_error(numeric_force, analytic_force), rel_threshold)
                                << "numeric forces and analytical force not consistent";
}

void
check_numeric_force(ACEAtomicEnvironment &ae, ACECalculator &ace, DOUBLE_TYPE rel_threshold, DOUBLE_TYPE dr,
                    int atom_ind_freq) {
    ace.compute(ae);
    auto original_forces = ace.forces;
    DOUBLE_TYPE E0 = ace.energy;
    DOUBLE_TYPE max_rel_error = 0;

    for (int i = 0; i < ae.n_atoms_real; i += atom_ind_freq) {
        for (int alpha = 0; alpha < 3; alpha++) {
            auto analytic_force = original_forces(i, alpha);
            if (abs(analytic_force) < 1e-12) continue;
            ACEAtomicEnvironment ae_def = ae;
            ae_def.x[i][alpha] += dr;
            ace.compute(ae_def);
            DOUBLE_TYPE E1 = ace.energy;
            DOUBLE_TYPE numeric_force = -(E1 - E0) / dr;
            printf("\n");
            printf("numeric_force(%d, %d) = %.15f, analytic_force(%d, %d) = %.15g\n", i, alpha, numeric_force, i, alpha,
                   analytic_force);
            if (absolute_relative_error(numeric_force, analytic_force) > max_rel_error)
                max_rel_error = absolute_relative_error(numeric_force, analytic_force);
            compare_forces(analytic_force, numeric_force, rel_threshold);

        }
    }
    printf("Maximum relative force error = %g\n", max_rel_error);
}


void print_input_structure_for_fortran(ACEAtomicEnvironment &atomic_environment) {
    printf("dimer-z regularclusters FINISHED niter 8\n");
    printf("groupstart T path /scratch/drautrmy/work.regularclusters/Al.FHI-PBE.ecut=tight.mag=1.smearing=0.10/1-body-000001\n");
    printf("FHI VERSION 171019\n");
    printf("SPIN channels 1\n");
    printf("XC pbe\n");
    printf("SMEARING  gaussian 0.10\n");
    printf("CLUSTER\n");
    printf("POS %d\n", atomic_environment.n_atoms_real);
    for (int i_at = 0; i_at < atomic_environment.n_atoms_real; i_at++) {
        printf("Al %f %f %f\n", atomic_environment.x[i_at][0],
               atomic_environment.x[i_at][1],
               atomic_environment.x[i_at][2]);
    }
    printf("ENERGY\n");
    printf("Total energy uncorrected 1.0\n");
    printf("Total energy corrected 1.0\n");
    printf("Electronic free energy 1.0\n");
    printf("FORCES\n");
    for (int i_at = 0; i_at < atomic_environment.n_atoms_real; i_at++) {
        printf("%d 0.0 0.0 0.0\n", i_at + 1);
    }
    printf("Date     :  20180112, Time     :  073942.889\n");
    printf("CPU time 0.677\n");
    printf("                     ---------------------------------\n");
    printf("dblastline\n");
}

void check_sum_of_forces(ACEAtomicEnvironment &ae, ACECalculator &aceCalculator, DOUBLE_TYPE threshold) {
    DOUBLE_TYPE fx = 0, fy = 0, fz = 0;
    DOUBLE_TYPE fx_abs = 0, fy_abs = 0, fz_abs = 0;

    for (int i_at = 0; i_at < ae.n_atoms_real; ++i_at) {
        fx += aceCalculator.forces(i_at, 0);
        fy += aceCalculator.forces(i_at, 1);
        fz += aceCalculator.forces(i_at, 2);

        fx_abs += abs(aceCalculator.forces(i_at, 0));
        fy_abs += abs(aceCalculator.forces(i_at, 1));
        fz_abs += abs(aceCalculator.forces(i_at, 2));
    }

    fx_abs /= ae.n_atoms_real;
    fy_abs /= ae.n_atoms_real;
    fz_abs /= ae.n_atoms_real;
    if (fx_abs > 0)
        ASSERT_LE(abs(fx) / fx_abs, threshold) << "Sum of forces_x !=0";
    if (fy_abs > 0)
        ASSERT_LE(abs(fy) / fy_abs, threshold) << "Sum of forces_y !=0";
    if (fz_abs > 0)
        ASSERT_LE(abs(fz) / fz_abs, threshold) << "Sum of forces_z !=0";
}


void check_cube_diagonal_forces_symmetry(ACEAtomicEnvironment &ae, ACECalculator &ace) {
    int first = 0;
    int last = ae.n_atoms_real - 1;
    printf("F(first atom) = (%.15g, %.15g, %.15g)\n", ace.forces(first, 0), ace.forces(first, 1), ace.forces(first, 2));
    printf("F(last  atom) = (%.15g, %.15g, %.15g)\n", ace.forces(last, 0), ace.forces(last, 1), ace.forces(last, 2));
    for (int i = 0; i < 3; i++) {
        ASSERT_LE(absolute_relative_error(ace.forces(first, i), -ace.forces(last, i)), 2e-10) << "non-symmetric forces";
    }

    ASSERT_LE(absolute_relative_error(ace.forces(first, 0), ace.forces(first, 1)), 2e-10) << "non-symmetric forces";
    ASSERT_LE(absolute_relative_error(ace.forces(first, 0), ace.forces(first, 2)), 2e-10) << "non-symmetric forces";
    ASSERT_LE(absolute_relative_error(ace.forces(first, 1), ace.forces(first, 2)), 2e-10) << "non-symmetric forces";

    ASSERT_LE(absolute_relative_error(ace.forces(last, 0), ace.forces(last, 1)), 2e-10) << "non-symmetric forces";
    ASSERT_LE(absolute_relative_error(ace.forces(last, 0), ace.forces(last, 2)), 2e-10) << "non-symmetric forces";
    ASSERT_LE(absolute_relative_error(ace.forces(last, 1), ace.forces(last, 2)), 2e-10) << "non-symmetric forces";
}
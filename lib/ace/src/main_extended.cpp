#include <iostream>

#include "ace_b_basis.h"
#include "ace_calculator.h"
#include "ace_utils.h"


using namespace std;

int main(int argc, char *argv[]) {

    string filename;

    if (argc < 2) {
        cout << "needs one command line argument - name of input file" << endl;
        return 0;
    } else {
        filename = argv[1];
    }

    ACEBBasisSet basis;
    ACECalculator ace;

    //parse the main information
    basis.load(filename);
    //basis.load(filename);
    cout << "Basis functions constructed." << endl;

    ACECTildeEvaluator eval;
    auto new_basis = basis.to_ACECTildeBasisSet();
    eval.set_basis(new_basis);
    ace.set_evaluator(eval);
    cout << "===========================" << endl;
    cout << "MAIN SIMULATION" << endl;
    cout << "===========================" << endl << endl;


//    ACEAtomicEnvironment ae = create_linear_chain(3);
//    ae.x[0][0] = -1;
//    ae.x[0][1] = -1;
//    ae.x[0][2] = -1;
//
//    ae.x[1][0] = 0;
//    ae.x[1][1] = 0;
//    ae.x[1][2] = 0;
//
//    ae.x[2][0] = 1;
//    ae.x[2][1] = 1;
//    ae.x[2][2] = 1;
    //ae.n_atoms_tot = 2;
    //ae.compute_neighbour_list(12.);

    ACEAtomicEnvironment ae = create_cube(3., 9.);
    //ACEAtomicEnvironment ae = create_linear_chain(2);
    //print_input_structure_for_fortran(ae);

    cout << "Number of atoms " << ae.n_atoms_real << endl;

    const int num_iter = 50;
    std::vector<double> timings(num_iter);
    for (int iter = 0; iter < num_iter; iter++) {
        ace.compute(ae, false);
        timings[iter] = (double) eval.per_atom_calc_timer.as_microseconds() / ae.n_atoms_real;
    }
    double min_timing = timings[0], ave_timing = 0, ave_dispersion = 0;
    for (int iter = 0; iter < num_iter; ++iter) {
        if (timings[iter] < min_timing)
            min_timing = timings[iter];
        ave_timing += timings[iter];
    }
    ave_timing /= num_iter;
    for (int iter = 0; iter < num_iter; ++iter) {
        ave_dispersion += sqr(timings[iter] - ave_timing);
    }
    ave_dispersion /= num_iter;
    cout << "Best of " << num_iter << ": " << min_timing << " microsec/atom" << endl;
    cout << "Average: " << ave_timing << "+-" << sqrt(ave_dispersion) << " microsec/atom" << endl;


    ace.compute(ae, true);
    cout << "Calculation complete. Final values: " << endl;

    int i_at;
    DOUBLE_TYPE fx = 0, fy = 0, fz = 0;
    for (i_at = 0; i_at < ae.n_atoms_real; ++i_at) {
//        printf("Force_i(%d)=(%2.15f, %2.15f, %2.15f)\n", i_at,
//               ace.forces(i_at, 0), ace.forces(i_at, 1), ace.forces(i_at, 2));
        fx += ace.forces(i_at, 0);
        fy += ace.forces(i_at, 1);
        fz += ace.forces(i_at, 2);
    }
    printf("Sum of forces: %g %g %g\n", fx, fy, fz);
    printf("Total energy = %2.15f\n", ace.energy);
    printf("Force ARE = %g\n", absolute_relative_error(ace.forces(0, 0), ace.forces(0, 2)));


    return 0;
}


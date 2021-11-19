#include "ace_calculator.h"
#include "ace_timing.h"
#include "ace_utils.h"

using namespace std;

int main(int argc, char *argv[]) {

    string filename;

    if (argc < 2) {
        printf("needs one command line argument - name of input file\n");
        return 0;
    } else {
        filename = argv[1];
    }

    ACECTildeBasisSet basis;
    ACECTildeEvaluator acecTildeEvaluator;
    ACECalculator aceCalculator;

    //parse the main information
    basis.load(filename);
    printf("Basis functions loaded\n");

    acecTildeEvaluator.set_basis(basis);
    aceCalculator.set_evaluator(acecTildeEvaluator);

    printf("===========================\n");
    printf("MAIN SIMULATION\n");
    printf("===========================\n");


    ACEAtomicEnvironment ae = create_cube(3., 9.);
    // ACEAtomicEnvironment ae = create_linear_chain(2.);
//
//    ae.x[0][0] = 0.; ae.x[0][1] = 0.; ae.x[0][2] = 0.;
//    ae.x[1][0] = 0.; ae.x[1][1] = 0.; ae.x[1][2] = 2.;
//    ae.x[2][0] = 0.; ae.x[2][1] = 1.; ae.x[2][2] = 2.;

    printf("Number of atoms: %d\n", ae.n_atoms_real);

    const int num_iter = 50;
    std::vector<double> timings(num_iter);
    for (int iter = 0; iter < num_iter; iter++) {
        aceCalculator.compute(ae, false);
        timings[iter] = (double) acecTildeEvaluator.per_atom_calc_timer.as_microseconds() / ae.n_atoms_real;
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
    printf("Best of %d: %f microsec/atom\n", num_iter, min_timing);
    printf("Average: %f +- %f microsec/atom\n", ave_timing, sqrt(ave_dispersion));

    aceCalculator.compute(ae, true);


    int i_at;
    DOUBLE_TYPE fx = 0, fy = 0, fz = 0;
    for (i_at = 0; i_at < ae.n_atoms_real; ++i_at) {
//        printf("Force_i(%d)=(%2.15f, %2.15f, %2.15f)\n", i_at,
//               ace.forces(i_at, 0), ace.forces(i_at, 1), ace.forces(i_at, 2));
        fx += aceCalculator.forces(i_at, 0);
        fy += aceCalculator.forces(i_at, 1);
        fz += aceCalculator.forces(i_at, 2);
    }
    printf("Sum of forces: %g %g %g\n", fx, fy, fz);
    printf("Total energy = %2.15f\n", aceCalculator.energy);
    printf("Force ARE = %g\n", absolute_relative_error(aceCalculator.forces(0, 0), aceCalculator.forces(0, 2)));
    return 0;
}


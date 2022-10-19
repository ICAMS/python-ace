#include "ace/ace_clebsch_gordan.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "wigner/wigner_3nj.hpp"

using namespace std;

double wigner3j(LS_TYPE j1, LS_TYPE m1, LS_TYPE j2, LS_TYPE m2, LS_TYPE J, LS_TYPE M) {
    if (m1 + m2 + M != 0) return 0;
    wigner::Wigner3jSeriesJ<double, int> w3j;
    w3j.compute(j1, j2, m1, m2);
    LS_TYPE jmin = w3j.nmin();
    LS_TYPE jmax = w3j.nmax();

    if (J < jmin || J > jmax) return 0;
    double res = w3j.get(J);
    return res;
}

double anotherClebschGordan(LS_TYPE j1, LS_TYPE m1, LS_TYPE j2, LS_TYPE m2, LS_TYPE J,
                            LS_TYPE M) {
    if (m1 + m2 != M) return 0;
    LS_TYPE jmin = abs(j1 - j2);
    LS_TYPE jmax = abs(j1 + j2);
    if (J > jmax || J < jmin)
        return 0;
    if (abs(m1) > j1) {
        stringstream s;
        char buf[1024];
        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
        s << buf;
        s << "Non-sense coefficient C_L: |m1|>l1";
        throw invalid_argument(s.str());
    }
    if (abs(m2) > j2) {
        stringstream s;
        char buf[1024];
        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
        s << buf;
        s << "Non-sense coefficient: |m2|>l2";
        throw invalid_argument(s.str());
    }
    if (abs(M) > J) {
        stringstream s;
        char buf[1024];
        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
        s << buf;
        s << "Non-sense coefficient: |M|>L";
        throw invalid_argument(s.str());
    }

    double factor = pow(-1.0, -j1 + j2 - M);
    double sqrtarg = sqrt(2.0 * J + 1.0);
    double w3j = wigner3j(j1, m1, j2, m2, J, -M);

    return factor * sqrtarg * w3j;
}


/**
Constructor for ClebschGordan. Dynamically initialises all the arrays.

@param lmax, int

The value of lmax

@returns None
*/
ACEClebschGordan::ACEClebschGordan(LS_TYPE lm) {
}


void ACEClebschGordan::init(LS_TYPE lm) {
}

/**
Destructor for ClebschGordan. Frees the memory of all the arrays.

@param None

@returns None
*/
ACEClebschGordan::~ACEClebschGordan() {
}


/**
Precomputes factorials up to 4*lmax+1 to use in the Racha's formula and other constants

@param None

@returns None
*/
void ACEClebschGordan::pre_compute() {
}


double ACEClebschGordan::clebsch_gordan(LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M) const {
    return anotherClebschGordan(j1, m1, j2, m2, J, M);
}


/**
Function that computes \f$ P_{lm} \f$ for the corresponding lmax value
Input is the \f$ \cos(\theta) \f$ value which can be calculated from an atom
and its neighbors.

For each \f$ \cos(\theta) \f$, this computes the whole range of \f$ P_{lm} \f$ values
and its derivatives upto the lmax specified, which is a member of the class.

@param costheta, double

@returns None
*/
double ACEClebschGordan::_compute_cbl(LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M) {
//   Racha's formula used to calculate Clebsch-Gordan coefficients

    DOUBLE_TYPE cg_value;
    DOUBLE_TYPE ph;
    LS_TYPE a, b;

    DOUBLE_TYPE prefac1 = 0;
    DOUBLE_TYPE prefac2 = 0;

    LS_TYPE J2 = (2 * J + 1);
    MS_TYPE j1j2mJ = j1 + j2 - J;
    MS_TYPE j1mj2J = j1 - j2 + J;
    MS_TYPE mj1j2J = -j1 + j2 + J;
    MS_TYPE JM = J + M;
    MS_TYPE JmM = J - M;
    MS_TYPE j1m1 = j1 + m1;
    MS_TYPE j1mm1 = j1 - m1;
    MS_TYPE j2m2 = j2 + m2;
    MS_TYPE j2mm2 = j2 - m2;
    MS_TYPE j1j2J1 = j1 + j2 + J + 1;
    MS_TYPE Jmj2m1 = J - j2 + m1;
    MS_TYPE Jmj1mm2 = J - j1 - m2;


    prefac1 = (J2 * fac(j1j2mJ) * fac(j1mj2J) * fac(mj1j2J) * fac(JM) * fac(JmM) * fac(j1m1) * fac(j1mm1) * fac(j2m2) *
               fac(j2mm2)) / fac(j1j2J1);


    a = max(max(0, j2 - J - m1), j1 - J + m2);
    b = min(min(j1 + j2 - J, j1 - m1), j2 + m2);

    for (MS_TYPE z = a; z <= b; z++) {
        ph = (z % 2 == 0 ? 1 : -1);
        prefac2 = prefac2 + (DOUBLE_TYPE) ph /
                            (fac(z) * fac(j1j2mJ - z) * fac(j1mm1 - z) * fac(j2m2 - z) * fac(Jmj2m1 + z) *
                             fac(Jmj1mm2 + z));

    }

    cg_value = sqrt(prefac1) * prefac2;
    return cg_value;

}

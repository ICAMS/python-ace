#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "ace_clebsch_gordan.h"
#include <sstream>

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
//    init(lm);
}


void ACEClebschGordan::init(LS_TYPE lm) {
//    if (lm<lmax)
//        return;;
//
//    lmax = lm;
//
//    //calculate the total size required
//    //eq (12) of Ref. [1] Jour. Mol. Str. THEOCHEM 715 (2005) 177-181
//    //1 + (l1*(1 + l1)*(-2 + l1*(5 + 3*l1)))/12 + (l2*(1 + l2))/2 + ((1 + l1)*(2 + l1)*(l1 + m1))/2 + m2
//
//    // maximum index F1 for columns from eq.(12)
//    F1max = (lmax * (lmax + 1) * (lmax * (3 * lmax + 5) - 2)) / 12 + (lmax + 1) * (lmax + 2) * (lmax + lmax) / 2 +
//            (lmax * (lmax + 1)) / 2 + lmax + 1;
//    F1max = F1max + 1; // for the first element in the last row
//
//    // maximum index F2 for rows from eq.(13)
//    F2max = lmax + 1;
//    F2max = F2max + 1; // for the first element in the last row
//
//    cgcoeff_len = F1max * F2max;
//    factorial_len = 4 * lmax + 2;
//#ifdef DEBUG_CLEBSCH
//    cout<<"F1max="<<F1max<< " F2max="<<F2max<<endl;
//    cout<<"factorial_len="<<factorial_len<<endl;
//    cout<<"cgcoeff_len="<<cgcoeff_len<<endl;
//#endif
//    fac.init(factorial_len, "fac");
//
//    cgcoeff.init(cgcoeff_len, "cgcoeff");
//    //fill in arrays with 0s, just to be sure for different compiler in same behaviour
//    fac.fill(0);
//    cgcoeff.fill(0);
//    pre_compute();
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
//
//    // the maximum factorial that can be computed is 20! with unsigned long long int
//    fac(0) = 1; // 0!
//    fac(1) = 1; // 1!
//
//    for (LS_TYPE l = 2; l <= 4 * lmax + 1; l++) {
//        if (fac(l - 1) * l < (std::numeric_limits<double>::max() - 1)) {
//            fac(l) = fac(l - 1) * (l);
//        } else {
//            stringstream s;
//            s << "Overflow! lmax = " << lmax << ", l=" << l << "! is too large";
//            throw invalid_argument(s.str());
//        }
//    }
//
//    DOUBLE_TYPE cg_value;
//    int i, j;
//
//    for (LS_TYPE j1 = 0; j1 <= lmax; j1++)
//        for (LS_TYPE j2 = 0; j2 <= j1; j2++)
//            for (MS_TYPE m1 = -j1; m1 <= j1; m1++)
//                for (MS_TYPE m2 = 0; m2 <= j2; m2++)
//                    for (LS_TYPE J = abs(j1 - j2); J <= abs(j1 + j2); J++)
//                        for (MS_TYPE M = -J; M <= J; M++) {
//
//
//                            if ((M != m1 + m2) || ((j1 + j2 + J) % 2 != 0))
//                                continue;
//
//                            i = compact_cg_get_i_coeff(j1, m1, j2, m2);
//                            j = compact_cg_get_j_coeff(j1, m1, j2, m2, J);
//                            cg_value = _compute_cbl(j1, m1, j2, m2, J, M);
//#ifdef DEBUG_CLEBSCH
//                            cout << "Cahing: CG("<< j1<<","<< m1<<"),("<<j2<<","<<m2<<"),("<<J<<","<<M<<")="<<cg_value <<endl;
//                            cout<<"goes to: i="<<i<<" j="<<j<<endl<<endl;
//#endif
//                            if (i * F2max + j < cgcoeff_len)
//                                cgcoeff(i * F2max + j) = cg_value;
//                            else {
//                                stringstream s;
//                                char buf[1024];
//                                sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
//                                s << buf;
//                                s << "cgcoeff out ouf range index: i=" << i << ", j=" << j << ", index=i*F2max+j="
//                                  << i * F2max + j << ", but max_ind = " << cgcoeff_len << endl
//                                  << "Probably Clebsh-Gordan coefficients were initialized for smaller lmax="<<lmax;
//
//                                throw invalid_argument(s.str());
//                            }
//                        }
}


double ACEClebschGordan::clebsch_gordan(LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M) const {
    return anotherClebschGordan(j1, m1, j2, m2, J, M);
//    int i, j;
//    DOUBLE_TYPE cg_inmem;
//#ifdef DEBUG_CLEBSCH
//    double cg_value;
//#endif
//
//    if ((M != m1 + m2) ) { //|| ((j1 + j2 + J) % 2 != 0)
//#ifdef DEBUG_CLEBSCH
//        cg_value = 0;
//        cout << "CG("<< j1<<","<< m1<<"),("<<j2<<","<<m2<<"),("<<J<<","<<M<<")="<<cg_value <<endl<<endl;
//#endif
//        return 0;
//    }
//    LS_TYPE jmin = abs(j1 - j2);
//    LS_TYPE jmax = abs(j1 + j2);
//    if (J > jmax || J < jmin)
//        return 0;
//
//    if (abs(m1) > j1) {
//        stringstream s;
//        char buf[1024];
//        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
//        s << buf;
//        s << "Non-sense coefficient C_L: |m1|>l1";
//        throw invalid_argument(s.str());
//    }
//    if (abs(m2) > j2) {
//        stringstream s;
//        char buf[1024];
//        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
//        s << buf;
//        s << "Non-sense coefficient: |m2|>l2";
//        throw invalid_argument(s.str());
//    }
//    if (abs(M) > J) {
//        stringstream s;
//        char buf[1024];
//        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
//        s << buf;
//        s << "Non-sense coefficient: |M|>L";
//        throw invalid_argument(s.str());
//    }
//
//    if (j2 > j1)
//        return clebsch_gordan(j2, m2, j1, m1, J, M);
//
//    if (m2 < 0)
//        return clebsch_gordan(j1, -m1, j2, -m2, J, -M);
//
//    i = compact_cg_get_i_coeff(j1, m1, j2, m2);
//    j = compact_cg_get_j_coeff(j1, m1, j2, m2, J);
//#ifdef DEBUG_CLEBSCH
//    cout<<"goes from: i="<<i<<" j="<<j<<endl;
//#endif
//    if (i * F2max + j < cgcoeff_len) {
//        cg_inmem = cgcoeff(i * F2max + j);
//#ifdef DEBUG_CLEBSCH
//        cg_value = _compute_cbl(j1, m1, j2, m2, J,M);
//        if (cg_value != cg_inmem)
//            cout<<"WARNING!!!"<<endl;
//        cout << "CG("<< j1<<","<< m1<<"),("<<j2<<","<<m2<<"),("<<J<<","<<M<<")="<<cg_value<<"(actual)"<<cg_inmem <<"(in mem)"<<endl<<endl;
//#endif
//        return cg_inmem;
//    } else {
//        stringstream s;
//        char buf[1024];
//        sprintf(buf, "C_L(%d|%d,%d)_M(%d|%d,%d): ", J, j1, j2, M, m1, m2);
//        s << buf;
//        s << "cgcoeff out ouf range index: i=" << i << ", j=" << j << ", index=i*F2max+j=" << i * F2max + j
//          << ", but max_ind = " << cgcoeff_len << endl;
//        s << "Probably Clebsh-Gordan coefficients were initialized for smaller lmax="<<lmax;
//        throw invalid_argument(s.str());
//    }
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

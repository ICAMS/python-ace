#ifndef ACE_CLEBSCH_GORDAN_H
#define ACE_CLEBSCH_GORDAN_H

#include <cmath>
#include <iostream>

#include "wigner_3nj.hpp"

#include "ace_types.h"
#include "ace_arraynd.h"

using namespace std;

double wigner3j(LS_TYPE j1, LS_TYPE m1, LS_TYPE j2, LS_TYPE m2, LS_TYPE J, LS_TYPE M);

double anotherClebschGordan(LS_TYPE j1, LS_TYPE m1, LS_TYPE j2, LS_TYPE m2, LS_TYPE J, LS_TYPE M);

/**
Class to store the Clebsch-Gordan coefficients through Racha's formula. \n
The coefficients are one-dimensional arrays of length (F1max+1)*(F2max+1). \n
*/
class ACEClebschGordan {
protected:

    void pre_compute();

    double _compute_cbl(LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M);
    //static int compact_cg_get_j_coeff(int j1, int m1, int j2, int m2, int J);
    //static int compact_cg_get_i_coeff(int j1, int m1, int j2, int m2);

public:
    /**
    int, the number of spherical harmonics to be found
    */
    LS_TYPE lmax = -1;

    int cgcoeff_len = 0;
    int factorial_len = 0;
    int F1max = -1, F2max = -1;

    //Array to store the factorials up to (4*lmax)
    //double *fac = nullptr;
    Array1D<DOUBLE_TYPE> fac;
    //Array to store the Clebsch-Gordan coefficients
    //DOUBLE_TYPE *cgcoeff = nullptr;
    Array1D<DOUBLE_TYPE> cgcoeff;

    ACEClebschGordan() = default;

    explicit ACEClebschGordan(LS_TYPE lmax);

    void init(LS_TYPE lm);

    ~ACEClebschGordan();

    DOUBLE_TYPE clebsch_gordan(LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M) const;
};


#endif
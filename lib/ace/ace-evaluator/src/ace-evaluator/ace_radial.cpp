/*
 * Performant implementation of atomic cluster expansion and interface to LAMMPS
 *
 * Copyright 2021  (c) Yury Lysogorskiy^1, Cas van der Oord^2, Anton Bochkarev^1,
 * Sarath Menon^1, Matteo Rinaldi^1, Thomas Hammerschmidt^1, Matous Mrovec^1,
 * Aidan Thompson^3, Gabor Csanyi^2, Christoph Ortner^4, Ralf Drautz^1
 *
 * ^1: Ruhr-University Bochum, Bochum, Germany
 * ^2: University of Cambridge, Cambridge, United Kingdom
 * ^3: Sandia National Laboratories, Albuquerque, New Mexico, USA
 * ^4: University of British Columbia, Vancouver, BC, Canada
 *
 *
 * See the LICENSE file.
 * This FILENAME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <cmath>
#include <functional>
#include <stdexcept>

#include "ace-evaluator/ace_radial.h"

#define sqr(x) ((x)*(x))
const DOUBLE_TYPE pi = 3.14159265358979323846264338327950288419; // pi

namespace ACEZBL {

    static char const *const elements_pace[] = {
            "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
            "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
            "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
            "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
            "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
            "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
            "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};
    static constexpr int elements_num_pace = sizeof(elements_pace) / sizeof(const char *);

    static int AtomicNumberByName_pace(const char *elname) {
        for (int i = 1; i < elements_num_pace; i++)
            if (strcmp(elname, elements_pace[i]) == 0) return i;
        return -1;
    }


    // transformation coefficients to eV,  K = _e**2 / (4 * np.pi * _eps0) / 1e-10 / _e
    DOUBLE_TYPE K = 14.399645351950543;

    // ZBL coefficients from https://docs.lammps.org/pair_zbl.html
    static const double ZBL_phi_coefs[4] = {0.18175, 0.50986, 0.28022, 0.02817};
    static const double ZBL_phi_exps[4] = {-3.19980, -0.94229, -0.40290, -0.20162};

    inline DOUBLE_TYPE phi(DOUBLE_TYPE x) {
        DOUBLE_TYPE phi = 0;
        for (int k = 0; k < 4; k++) {
            phi += ZBL_phi_coefs[k] * exp(ZBL_phi_exps[k] * x);
        }
        return phi;
    }

    std::tuple<DOUBLE_TYPE, DOUBLE_TYPE, DOUBLE_TYPE> phi_and_deriv2(DOUBLE_TYPE x) {
        DOUBLE_TYPE phi = 0, dphi = 0, d2phi = 0;
        DOUBLE_TYPE e, t;
        for (int k = 0; k < 4; k++) {
            e = exp(ZBL_phi_exps[k] * x);
            t = ZBL_phi_coefs[k] * e;
            phi += t;
            t *= ZBL_phi_exps[k];
            dphi += t;
            t *= ZBL_phi_exps[k];
            d2phi += t;
        }
        return {phi, dphi, d2phi};
    }

    inline DOUBLE_TYPE fun_E_ij(DOUBLE_TYPE r, DOUBLE_TYPE a) {
        return 1 / r * phi(r / a);
    }

    std::tuple<DOUBLE_TYPE, DOUBLE_TYPE, DOUBLE_TYPE> fun_E_ij_and_deriv2(DOUBLE_TYPE r, DOUBLE_TYPE a) {
        DOUBLE_TYPE phi, dphi, d2phi;
        DOUBLE_TYPE E_ij, dE_ij, dE2_ij;

        std::tie(phi, dphi, d2phi) = ACEZBL::phi_and_deriv2(r / a);

        E_ij = 1 / r * phi;
        dE_ij = (-1 / (r * r)) * phi + 1 / r * dphi / a;
        dE2_ij = (2 / (r * r * r)) * phi + 2 * (-1 / (r * r)) * dphi / a + (1 / r) * d2phi / ((a * a));

        return {E_ij, dE_ij, dE2_ij};
    }

} // namespace ACEZBL


ACERadialFunctions::ACERadialFunctions(NS_TYPE nradb, LS_TYPE lmax, NS_TYPE nradial, DOUBLE_TYPE deltaSplineBins,
                                       SPECIES_TYPE nelements,
                                       vector<vector<string>> radbasename) {
    init(nradb, lmax, nradial, deltaSplineBins, nelements, radbasename);
}

void ACERadialFunctions::init(NS_TYPE nradb, LS_TYPE lmax, NS_TYPE nradial, DOUBLE_TYPE deltaSplineBins,
                              SPECIES_TYPE nelements, vector<vector<string>> radbasename) {
    this->nradbase = nradb;
    this->lmax = lmax;
    this->nradial = nradial;
    this->deltaSplineBins = deltaSplineBins;
    this->nelements = nelements;
    this->radbasenameij = radbasename;
    auto shape = this->radbasenameij.get_shape();

    if (shape.size() != 2 || shape.at(0) != (unsigned) nelements || shape.at(1) != (unsigned) nelements) {
        throw std::invalid_argument("`radbasename` array has wrong shape. It must be of shape (nelements, nelements)");
    }


    gr.init(nradbase, "gr");
    dgr.init(nradbase, "dgr");
    d2gr.init(nradbase, "d2gr");


    fr.init(nradial, lmax + 1, "fr");
    dfr.init(nradial, lmax + 1, "dfr");
    d2fr.init(nradial, lmax + 1, "d2fr");


    cheb.init(nradbase + 1, "cheb");
    dcheb.init(nradbase + 1, "dcheb");
    cheb2.init(nradbase + 1, "cheb2");


    splines_gk.init(nelements, nelements, "splines_gk");
    splines_rnl.init(nelements, nelements, "splines_rnl");
    splines_hc.init(nelements, nelements, "splines_hc");

    lambda.init(nelements, nelements, "lambda");
    lambda.fill(1.);

    cut.init(nelements, nelements, "cut");
    cut.fill(1.);

    dcut.init(nelements, nelements, "dcut");
    dcut.fill(1.);

    cut_in.init(nelements, nelements, "cut_in");
    cut_in.fill(0);

    dcut_in.init(nelements, nelements, "dcut_in");
    dcut_in.fill(1e-5);

    crad.init(nelements, nelements, nradial, (lmax + 1), nradbase, "crad");
    crad.fill(0.);

    //hard-core repulsion
    prehc.init(nelements, nelements, "prehc");
    prehc.fill(0.);

    lambdahc.init(nelements, nelements, "lambdahc");
    lambdahc.fill(1.);

}


/**
Function that computes Chebyshev polynomials of first and second kind
 to setup the radial functions and the derivatives

@param n, x

@returns cheb1, dcheb1
*/
void ACERadialFunctions::calcCheb(NS_TYPE n, DOUBLE_TYPE x) {
    if (n < 0) {
        char s[1024];
        sprintf(s, "The order n of the polynomials should be positive %d\n", n);
        throw std::invalid_argument(s);
    }
    DOUBLE_TYPE twox = 2.0 * x;
    cheb(0) = 1.;
    dcheb(0) = 0.;
    cheb2(0) = 1.;

    if (nradbase >= 1) {
        cheb(1) = x;
        cheb2(1) = twox;
    }
    for (NS_TYPE m = 1; m <= n - 1; m++) {
        cheb(m + 1) = twox * cheb(m) - cheb(m - 1);
        cheb2(m + 1) = twox * cheb2(m) - cheb2(m - 1);
    }
    for (NS_TYPE m = 1; m <= n; m++) {
        dcheb(m) = m * cheb2(m - 1);
    }
#ifdef DEBUG_RADIAL
    for ( NS_TYPE  m=0; m<=n; m++ ) {
        printf(" m %d cheb %f dcheb %f \n", m, cheb(m), dcheb(m));
    }
#endif
}

/**
 * Polynomial inner cutoff  function, descending from r_in-delta_in to r_in
 * @param r actual r
 * @param r_in inner cutoff
 * @param delta_in decay of inner cutoff
 * @param fc
 * @param dfc
 */
void cutoff_func_poly(DOUBLE_TYPE r, DOUBLE_TYPE r_in, DOUBLE_TYPE delta_in, DOUBLE_TYPE &fc, DOUBLE_TYPE &dfc) {
    if (r <= r_in - delta_in) {
        fc = 1;
        dfc = 0;
    } else if (r >= r_in) {
        fc = 0;
        dfc = 0;
    } else {
        DOUBLE_TYPE x = 1 - 2 * (1 + (r - r_in) / delta_in);
        fc = 0.5 + 7.5 / 2. * (x / 4. - pow(x, 3) / 6. + pow(x, 5) / 20.);
        dfc = -7.5 / delta_in * (0.25 - x * x / 2.0 + pow(x, 4) / 4.);
    }
}

/**
Function that computes radial basis.

@param lam, nradbase, cut, dcut, r

@returns gr, dgr
*/
void
ACERadialFunctions::radbase(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, string radbasename, DOUBLE_TYPE r,
                            DOUBLE_TYPE cut_in, DOUBLE_TYPE dcut_in) {
    // rest inner cutoff for ZBL core-repulsion type
    if (inner_cutoff_type == "zbl")
        cut_in = dcut_in==0;
    /*lam is given by the formula (24), that contains cut */
    if (r <= cut_in - dcut_in || r >= cut) {
        gr.fill(0);
        dgr.fill(0);
    } else { // cut_in < r < cut
        if (radbasename == "ChebExpCos") {
            chebExpCos(lam, cut, dcut, r);
        } else if (radbasename == "ChebPow") {
            chebPow(lam, cut, dcut, r);
        } else if (radbasename == "ChebLinear") {
            chebLinear(lam, cut, dcut, r);
        } else if (radbasename == "TEST_SBessel" || radbasename == "SBessel") {
            simplified_bessel(cut, r);
        } else if (radbasename.rfind("TEST_", 0) == 0) {
            test_zero_func(lam, cut, dcut, r);
        } else {
            throw invalid_argument("Unknown radial basis function name: " + radbasename);
        }

        //TODO: always take into account inner cutoff
        if (inner_cutoff_type == "distance" || inner_cutoff_type == "zbl") {
            //IMPORTANT!!! radial inner cutoff is shifted by dcut_in and
            // is applied now in a range from cut_in-dcut_in to cut_in-2*dcut_in
            //multiply by cutoff poly gr and dgr
            DOUBLE_TYPE fc, dfc;
            cutoff_func_poly(r, cut_in, dcut_in, fc, dfc); // ascending inner cutoff
            // cutoff_func_poly(r, 0, 0, fc, dfc) for ZBL
            fc = 1 - fc;
            dfc = -dfc;
            for (unsigned int i = 0; i < gr.get_dim(0); i++) {
                DOUBLE_TYPE new_gr = gr(i) * fc;
                DOUBLE_TYPE new_dgr = dgr(i) * fc + gr(i) * dfc;
                gr(i) = new_gr;
                dgr(i) = new_dgr;
            }
        }
    }
}

/***
 *  Radial function: ChebExpCos, cheb exp scaling including cos envelope
 * @param lam function parameter
 * @param cut cutoff distance
 * @param r function input argument
 * @return fills in gr and dgr arrays
 */
void
ACERadialFunctions::chebExpCos(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, DOUBLE_TYPE r) {
    DOUBLE_TYPE y2, y1, x, dx;
    DOUBLE_TYPE env, denv, fcut, dfcut;
    /* scaled distance x and derivative*/
    y1 = exp(-lam * r / cut);
    y2 = exp(-lam);
    x = 1.0 - 2.0 * ((y1 - y2) / (1 - y2));
    dx = 2 * (lam / cut) * (y1 / (1 - y2));
    /* calculation of Chebyshev polynomials from the recursion */
    calcCheb(nradbase, x);
    gr(0) = cheb(0);
    dgr(0) = dcheb(0) * dx;
    for (NS_TYPE n = 2; n <= nradbase; n++) {
        gr(n - 1) = 0.5 - 0.5 * cheb(n - 1);
        dgr(n - 1) = -0.5 * dcheb(n - 1) * dx;
    }
    env = 0.5 * (1.0 + cos(pi * r / cut));
    denv = -0.5 * sin(pi * r / cut) * pi / cut;
    for (NS_TYPE n = 0; n < nradbase; n++) {
        dgr(n) = gr(n) * denv + dgr(n) * env;
        gr(n) = gr(n) * env;
    }
    // for radtype = 3 a smooth cut is already included in the basis function
    dx = cut - dcut;
    if (r > dx) {
        fcut = 0.5 * (1.0 + cos(pi * (r - dx) / dcut));
        dfcut = -0.5 * sin(pi * (r - dx) / dcut) * pi / dcut;
        for (NS_TYPE n = 0; n < nradbase; n++) {
            dgr(n) = gr(n) * dfcut + dgr(n) * fcut;
            gr(n) = gr(n) * fcut;
        }
    }
}

/***
*  Radial function: ChebPow
* - argument of Chebyshev polynomials
* x = 2.0*( 1.0 - (1.0 - r/rcut)^lam ) - 1.0
* - radial function
* gr(n) = ( 1.0 - Cheb(n) )/2.0, n = 1,...,nradbase
* - the function fulfills:
* gr(n) = 0 at rcut
* dgr(n) = 0 at rcut for lam >= 1
* second derivative zero at rcut for lam >= 2
* -> the radial function does not require a separate cutoff function
* - corresponds to radial basis radtype=5 in Fortran code
*
* @param lam function parameter
* @param cut cutoff distance
* @param r function input argument
* @return fills in gr and dgr arrays
*/
void
ACERadialFunctions::chebPow(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, DOUBLE_TYPE r) {
    DOUBLE_TYPE y, dy, x, dx;
    /* scaled distance x and derivative*/
    y = (1.0 - r / cut);
    dy = pow(y, (lam - 1.0));
    y = dy * y;
    dy = -lam / cut * dy;

    x = 2.0 * (1.0 - y) - 1.0;
    dx = -2.0 * dy;
    calcCheb(nradbase, x);
    for (NS_TYPE n = 1; n <= nradbase; n++) {
        gr(n - 1) = 0.5 - 0.5 * cheb(n);
        dgr(n - 1) = -0.5 * dcheb(n) * dx;
    }
}


void
ACERadialFunctions::chebLinear(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, DOUBLE_TYPE r) {
    DOUBLE_TYPE x, dx;
    /* scaled distance x and derivative*/
    x = (1.0 - r / cut);
    dx = -1 / cut;
    calcCheb(nradbase, x);
    for (NS_TYPE n = 1; n <= nradbase; n++) {
        gr(n - 1) = 0.5 - 0.5 * cheb(n);
        dgr(n - 1) = -0.5 * dcheb(n) * dx;
    }
}

/**
 * sinc(x) = sin(x)/x
 * @param x
 * @return
 */
DOUBLE_TYPE sinc(DOUBLE_TYPE x) {
    return x != 0.0 ? sin(x) / x : 1;
}

/**
 * Derivative of d sinc(x) / dx = (cos(x)*x - sin(x))/x^2
 * @param x
 * @return
 */
DOUBLE_TYPE dsinc(DOUBLE_TYPE x) {
    return x != 0.0 ? (cos(x) * x - sin(x)) / (x * x) : 0;
}

/**
 * Auxiliary function $fn$ for simplified Bessel
 * @param x argument
 * @param rc cutoff
 * @param n degree
 * @return
 */
DOUBLE_TYPE simplified_bessel_aux(DOUBLE_TYPE x, DOUBLE_TYPE rc, int n) {
    return pow(-1, n) * sqrt(2) * pi / pow(rc, 1.5) * (n + 1) * (n + 2) / sqrt(sqr(n + 1) + sqr(n + 2)) *
           (sinc(x * (n + 1) * pi / rc) + sinc(x * (n + 2) * pi / rc));
}

/**
 * Derivative of the auxiliary function $fn$ for simplified Bessel
 * @param x argument
 * @param rc cutoff
 * @param n degree
 * @return
 */
DOUBLE_TYPE dsimplified_bessel_aux(DOUBLE_TYPE x, DOUBLE_TYPE rc, int n) {
    return pow(-1, n) * sqrt(2) * pi / pow(rc, 1.5) * (n + 1) * (n + 2) / sqrt(sqr(n + 1) + sqr(n + 2)) *
           (dsinc(x * (n + 1) * pi / rc) * (n + 1) * pi / rc +
            dsinc(x * (n + 2) * pi / rc) * (n + 2) * pi / rc);
}

/**
 * Simplified Bessel function
 * @param rc
 * @param x
 */
void ACERadialFunctions::simplified_bessel(DOUBLE_TYPE rc, DOUBLE_TYPE x) {
    if (x < rc) {
        gr(0) = simplified_bessel_aux(x, rc, 0);
        dgr(0) = dsimplified_bessel_aux(x, rc, 0);

        DOUBLE_TYPE d_prev = 1.0, en, dn;
        for (NS_TYPE n = 1; n < nradbase; n++) {
            en = sqr(n) * sqr(n + 2) / (4 * pow(n + 1, 4) + 1);
            dn = 1 - en / d_prev;
            gr(n) = 1 / sqrt(dn) * (simplified_bessel_aux(x, rc, n) + sqrt(en / d_prev) * gr(n - 1));
            dgr(n) = 1 / sqrt(dn) * (dsimplified_bessel_aux(x, rc, n) + sqrt(en / d_prev) * dgr(n - 1));
            d_prev = dn;
        }
    } else {
        gr.fill(0);
        dgr.fill(0);
    }
}


/**
 * Stub zero function (for testing purposes mostly), called when radbasename starts with "TEST_"
 * @param lam
 * @param cut
 * @param dcut
 * @param r
 */
void ACERadialFunctions::test_zero_func(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, DOUBLE_TYPE r) {
    gr.fill(0);
    dgr.fill(0);
}

/**
Function that computes radial functions.

@param nradbase, nelements, elei, elej

@returns fr, dfr
*/
void ACERadialFunctions::radfunc(SPECIES_TYPE elei, SPECIES_TYPE elej) {
    DOUBLE_TYPE frval, dfrval;
    for (NS_TYPE n = 0; n < nradial; n++) {
        for (LS_TYPE l = 0; l <= lmax; l++) {
            frval = 0.0;
            dfrval = 0.0;
            for (NS_TYPE k = 0; k < nradbase; k++) {
                frval += crad(elei, elej, n, l, k) * gr(k);
                dfrval += crad(elei, elej, n, l, k) * dgr(k);
            }
            fr(n, l) = frval;
            dfr(n, l) = dfrval;
        }
    }
}


void ACERadialFunctions::all_radfunc(SPECIES_TYPE mu_i, SPECIES_TYPE mu_j, DOUBLE_TYPE r) {
    DOUBLE_TYPE lam = lambda(mu_i, mu_j);
    DOUBLE_TYPE r_cut = cut(mu_i, mu_j);
    DOUBLE_TYPE dr_cut = dcut(mu_i, mu_j);

    DOUBLE_TYPE r_in = cut_in(mu_i, mu_j);
    DOUBLE_TYPE dr_in = dcut_in(mu_i, mu_j);

    // set up radial functions
    radbase(lam, r_cut, dr_cut, radbasenameij(mu_i, mu_j), r, r_in, dr_in); //update gr, dgr
    radfunc(mu_i, mu_j); // update fr(nr, l),  dfr(nr, l)
}


void ACERadialFunctions::setuplookupRadspline() {
    using namespace std::placeholders;
    DOUBLE_TYPE lam, r_cut, dr_cut, r_in, delta_in;
    DOUBLE_TYPE cr_c, dcr_c, pre, lamhc;
    string radbasename;

    // at r = rcut + eps the function and its derivatives is zero
    for (SPECIES_TYPE elei = 0; elei < nelements; elei++) {
        for (SPECIES_TYPE elej = 0; elej < nelements; elej++) {

            lam = lambda(elei, elej);
            r_cut = cut(elei, elej);
            dr_cut = dcut(elei, elej);
            r_in = cut_in(elei, elej);
            delta_in = dcut_in(elei, elej);
            radbasename = radbasenameij(elei, elej);

            splines_gk(elei, elej).setupSplines(gr.get_size(),
                                                std::bind(&ACERadialFunctions::radbase, this, lam, r_cut, dr_cut,
                                                          radbasename,
                                                          _1, r_in, delta_in),//update gr, dgr
                                                gr.get_data(),
                                                dgr.get_data(), deltaSplineBins, r_cut);

            splines_rnl(elei, elej).setupSplines(fr.get_size(),
                                                 std::bind(&ACERadialFunctions::all_radfunc, this, elei, elej,
                                                           _1), // update fr(nr, l),  dfr(nr, l)
                                                 fr.get_data(),
                                                 dfr.get_data(), deltaSplineBins, r_cut);

            if (inner_cutoff_type == "density" || inner_cutoff_type == "distance") {
                pre = prehc(elei, elej);
                lamhc = lambdahc(elei, elej);
                //            radcore(r, pre, lamhc, cutoff, cr_c, dcr_c, r_cut_in, dr_cut_in);
                splines_hc(elei, elej).setupSplines(1,
                                                    std::bind(&ACERadialFunctions::radcore, this,
                                                              _1, pre, lamhc, r_cut,
                                                              std::ref(cr_c), std::ref(dcr_c),
                                                              r_in, delta_in),
                                                    &cr_c,
                                                    &dcr_c, deltaSplineBins, r_cut);
            } else if (inner_cutoff_type == "zbl") {
                int Zi = ACEZBL::AtomicNumberByName_pace(elements[elei].c_str());
                if (Zi == -1) throw runtime_error("Element `" + elements[elei] + "` is not recognized");
                int Zj = ACEZBL::AtomicNumberByName_pace(elements[elej].c_str());
                if (Zj == -1) throw runtime_error("Element `" + elements[elej] + "` is not recognized");
                pre = prehc(elei, elej);

                //use OUTER cutoff, so ZBL always acts "on full force" by itself.
                // Switching from ACE to ZBL happens in the inner cutoff
                splines_hc(elei, elej).setupSplines(1,
                                                    std::bind(&ACERadialFunctions::ZBL, _1, Zi, Zj,
                                                              std::ref(cr_c), std::ref(dcr_c), r_cut, r_cut - dr_cut,
                                                              pre),
                                                    &cr_c,
                                                    &dcr_c, deltaSplineBins, r_cut);
            } else {
                throw runtime_error("Not implemented core-repulsion:" + inner_cutoff_type);
            }

        }
    }

}

/**
Function that gets radial function from look-up table using splines.

@param r, nradbase_c, nradial_c, lmax, mu_i, mu_j

@returns fr, dfr, gr, dgr, cr, dcr
*/
void
ACERadialFunctions::evaluate(DOUBLE_TYPE r, NS_TYPE nradbase_c, NS_TYPE nradial_c, SPECIES_TYPE mu_i,
                             SPECIES_TYPE mu_j, bool calc_second_derivatives) {
    auto &spline_gk = splines_gk(mu_i, mu_j);
    auto &spline_rnl = splines_rnl(mu_i, mu_j);
    auto &spline_hc = splines_hc(mu_i, mu_j);

    spline_gk.calcSplines(r, calc_second_derivatives); // populate  splines_gk.values, splines_gk.derivatives;
    for (NS_TYPE nr = 0; nr < nradbase_c; nr++) {
        gr(nr) = spline_gk.values(nr);
        dgr(nr) = spline_gk.derivatives(nr);
        if (calc_second_derivatives)
            d2gr(nr) = spline_gk.second_derivatives(nr);
    }

    spline_rnl.calcSplines(r, calc_second_derivatives);
    for (size_t ind = 0; ind < fr.get_size(); ind++) {
        fr.get_data(ind) = spline_rnl.values.get_data(ind);
        dfr.get_data(ind) = spline_rnl.derivatives.get_data(ind);
        if (calc_second_derivatives)
            d2fr.get_data(ind) = spline_rnl.second_derivatives.get_data(ind);
    }

    spline_hc.calcSplines(r, calc_second_derivatives);
    cr = spline_hc.values(0);
    dcr = spline_hc.derivatives(0);
    if (calc_second_derivatives)
        d2cr = spline_hc.second_derivatives(0);

}

/**
 pseudocode for hard core repulsion
in:
 r: distance
 pre: prefactor: read from input, depends on pair of atoms mu_i mu_j
 lambda: exponent: read from input, depends on pair of atoms mu_i mu_j
 cutoff: cutoff distance: read from input, depends on pair of atoms mu_i mu_j
out:
cr: hard core repulsion
dcr: derivative of hard core repulsion

 function
 \$f f_{core} = pre \exp( - \lambda r^2 ) / r   \$f

*/
void
ACERadialFunctions::radcore(DOUBLE_TYPE r, DOUBLE_TYPE pre, DOUBLE_TYPE lambda, DOUBLE_TYPE cutoff, DOUBLE_TYPE &cr,
                            DOUBLE_TYPE &dcr, DOUBLE_TYPE r_in, DOUBLE_TYPE delta_in) {

    DOUBLE_TYPE r2, lr2, y, x0, env, denv;

    // repulsion strictly positive and decaying
    pre = abs(pre);
    lambda = abs(lambda);

    r2 = r * r;
    lr2 = lambda * r2;
    if (lr2 < 50.0) {
        y = exp(-lr2);
        cr = pre * y / r;
        dcr = -pre * y * (2.0 * lr2 + 1.0) / r2;

        x0 = r / cutoff;
        env = 0.5 * (1.0 + cos(pi * x0));
        denv = -0.5 * sin(pi * x0) * pi / cutoff;
        dcr = cr * denv + dcr * env;
        cr = cr * env;
    } else {
        cr = 0.0;
        dcr = 0.0;
    }

    if (inner_cutoff_type == "distance") {
        // core repulsion became non-zero only within r < cut_in
        DOUBLE_TYPE fc, dfc;
        cutoff_func_poly(r, r_in, delta_in, fc, dfc);
        DOUBLE_TYPE new_cr = cr * fc;
        DOUBLE_TYPE new_dcr = dcr * fc + cr * dfc;
        cr = new_cr;
        dcr = new_dcr;
    }
}


void ACERadialFunctions::ZBL(DOUBLE_TYPE r, int Zi, int Zj, DOUBLE_TYPE &cr,
                             DOUBLE_TYPE &dcr, DOUBLE_TYPE cut_out, DOUBLE_TYPE cut_in, DOUBLE_TYPE prefactor) {
    // ZBL outer cutoff = cut_out
    // ZBL inner cutoff = cut_in
    DOUBLE_TYPE zbl_in = cut_in;
    DOUBLE_TYPE zbl_out = cut_out;
    if (r >= cut_out) {
        cr = 0;
        dcr = 0;
        return;
    }
    DOUBLE_TYPE a = 0.46850 / (pow(Zi, 0.23) + pow(Zj, 0.23));

    DOUBLE_TYPE E_ij, dE_ij, _t;
    std::tie(E_ij, dE_ij, _t) = ACEZBL::fun_E_ij_and_deriv2(r, a);

    DOUBLE_TYPE Ec, dEc, d2Ec;
    std::tie(Ec, dEc, d2Ec) = ACEZBL::fun_E_ij_and_deriv2(zbl_out, a);

    DOUBLE_TYPE drcut = zbl_out - zbl_in;

    DOUBLE_TYPE A = (-3 * dEc + drcut * d2Ec) / (drcut * drcut);
    DOUBLE_TYPE B = (2 * dEc - drcut * d2Ec) / (drcut * drcut * drcut);
    DOUBLE_TYPE C = -Ec + 1. / 2 * drcut * dEc - 1. / 12 * (drcut * drcut) * d2Ec;

    DOUBLE_TYPE S;
    if (r <= zbl_in) S = C;
    else if (r >= zbl_out) S = 0;
    else S = A / 3 * pow(r - zbl_in, 3) + B / 4 * pow(r - zbl_in, 4) + C;

    cr = prefactor * ACEZBL::K / 2 * Zi * Zj * (E_ij + S);

    // forces:
    // dS_dr = A * (d - self.cut_in) ** 2 + B * (d - self.cut_in) ** 3
    DOUBLE_TYPE dS_dr = 0;
    if (zbl_in < r && r < zbl_out) dS_dr = A * pow(r - zbl_in, 2) + B * pow(r - zbl_in, 3);

    dcr = prefactor * ACEZBL::K / 2 * Zi * Zj * (dE_ij + dS_dr);

    //poly cutoff
    DOUBLE_TYPE delta_in = cut_out - cut_in;
    DOUBLE_TYPE fc, dfc;
    cutoff_func_poly(r, cut_out, delta_in, fc, dfc);
    DOUBLE_TYPE new_cr = cr * fc;
    DOUBLE_TYPE new_dcr = dcr * fc + cr * dfc;
    cr = new_cr;
    dcr = new_dcr;
}

void
ACERadialFunctions::evaluate_range(vector<DOUBLE_TYPE> r_vec, NS_TYPE nradbase_c, NS_TYPE nradial_c, SPECIES_TYPE mu_i,
                                   SPECIES_TYPE mu_j) {
    if (nradbase_c > nradbase)
        throw invalid_argument("nradbase_c couldn't be larger than nradbase");
    if (nradial_c > nradial)
        throw invalid_argument("nradial_c couldn't be larger than nradial");
    if (mu_i > nelements)
        throw invalid_argument("mu_i couldn't be larger than nelements");
    if (mu_j > nelements)
        throw invalid_argument("mu_j couldn't be larger than nelements");

    gr_vec.resize(r_vec.size(), nradbase_c);
    dgr_vec.resize(r_vec.size(), nradbase_c);
    d2gr_vec.resize(r_vec.size(), nradbase_c);

    fr_vec.resize(r_vec.size(), fr.get_dim(0), fr.get_dim(1));
    dfr_vec.resize(r_vec.size(), fr.get_dim(0), fr.get_dim(1));
    d2fr_vec.resize(r_vec.size(), fr.get_dim(0), fr.get_dim(1));

    for (size_t i = 0; i < r_vec.size(); i++) {
        DOUBLE_TYPE r = r_vec[i];
        this->evaluate(r, nradbase_c, nradial_c, mu_i, mu_j, true);
        for (NS_TYPE nr = 0; nr < nradbase_c; nr++) {
            gr_vec(i, nr) = gr(nr);
            dgr_vec(i, nr) = dgr(nr);
            d2gr_vec(i, nr) = d2gr(nr);
        }

        for (NS_TYPE nr = 0; nr < nradial_c; nr++) {
            for (LS_TYPE l = 0; l <= lmax; l++) {
                fr_vec(i, nr, l) = fr(nr, l);
                dfr_vec(i, nr, l) = dfr(nr, l);
                d2fr_vec(i, nr, l) = d2fr(nr, l);

            }
        }
    }
}

void SplineInterpolator::setupSplines(int num_of_functions, RadialFunctions func,
                                      DOUBLE_TYPE *values,
                                      DOUBLE_TYPE *dvalues, DOUBLE_TYPE deltaSplineBins, DOUBLE_TYPE cutoff) {

    this->deltaSplineBins = deltaSplineBins;
    if (deltaSplineBins <= 0) {
        throw invalid_argument("deltaSplineBins should be positive");
    }
    this->cutoff = cutoff;
    this->ntot = static_cast<int>(cutoff / deltaSplineBins);

    DOUBLE_TYPE r, c[4];
    this->num_of_functions = num_of_functions;
    this->values.resize(num_of_functions);
    this->derivatives.resize(num_of_functions);
    this->second_derivatives.resize(num_of_functions);

    Array1D<DOUBLE_TYPE> f1g(num_of_functions);
    Array1D<DOUBLE_TYPE> f1gd1(num_of_functions);
    f1g.fill(0);
    f1gd1.fill(0);

    nlut = ntot;
    DOUBLE_TYPE f0, f1, f0d1, f1d1;
    int idx;

    // cutoff is global cutoff
    rscalelookup = (DOUBLE_TYPE) nlut / cutoff;
    invrscalelookup = 1.0 / rscalelookup;

    lookupTable.init(ntot + 1, num_of_functions, 4);
    if ((values == nullptr) && (num_of_functions > 0))
        throw invalid_argument("SplineInterpolator::setupSplines: values could not be null");
    if ((dvalues == nullptr) && (num_of_functions > 0))
        throw invalid_argument("SplineInterpolator::setupSplines: dvalues could not be null");

    if (num_of_functions == 0)
        return;

    for (int n = nlut; n >= 1; n--) {
        r = invrscalelookup * DOUBLE_TYPE(n);
        func(r); //populate values and dvalues arrays
        for (int func_id = 0; func_id < num_of_functions; func_id++) {
            f0 = values[func_id];
            f1 = f1g(func_id);
            f0d1 = dvalues[func_id] * invrscalelookup;
            f1d1 = f1gd1(func_id);
            // evaluate coefficients
            c[0] = f0;
            c[1] = f0d1;
            c[2] = 3.0 * (f1 - f0) - f1d1 - 2.0 * f0d1;
            c[3] = -2.0 * (f1 - f0) + f1d1 + f0d1;
            // store coefficients
            for (idx = 0; idx <= 3; idx++)
                lookupTable(n, func_id, idx) = c[idx];

            // evaluate function values and derivatives at current position
            f1g(func_id) = c[0];
            f1gd1(func_id) = c[1];
        }
    }


}


void SplineInterpolator::calcSplines(DOUBLE_TYPE r, bool calc_second_derivatives) {
    DOUBLE_TYPE wl, wl2, wl3, w2l1, w3l2, w4l2;
    DOUBLE_TYPE c[4];
    int func_id, idx;
    DOUBLE_TYPE x = r * rscalelookup;
    int nl = static_cast<int>(floor(x));

    if (nl <= 0)
        throw std::invalid_argument("Encountered very small distance. Stopping.");

    if (nl < nlut) {
        wl = x - DOUBLE_TYPE(nl);
        wl2 = wl * wl;
        wl3 = wl2 * wl;
        w2l1 = 2.0 * wl;
        w3l2 = 3.0 * wl2;
        w4l2 = 6.0 * wl;
        for (func_id = 0; func_id < num_of_functions; func_id++) {
            for (idx = 0; idx <= 3; idx++) {
                c[idx] = lookupTable(nl, func_id, idx);
            }
            values(func_id) = c[0] + c[1] * wl + c[2] * wl2 + c[3] * wl3;
            derivatives(func_id) = (c[1] + c[2] * w2l1 + c[3] * w3l2) * rscalelookup;
            if (calc_second_derivatives)
                second_derivatives(func_id) = (c[2] + c[3] * w4l2) * rscalelookup * rscalelookup * 2;
        }
    } else { // fill with zeroes
        values.fill(0);
        derivatives.fill(0);
        if (calc_second_derivatives)
            second_derivatives.fill(0);
    }
}

void SplineInterpolator::calcSplines(DOUBLE_TYPE r, SHORT_INT_TYPE func_ind) {
    DOUBLE_TYPE wl, wl2, wl3, w2l1, w3l2;
    DOUBLE_TYPE c[4];
    int idx;
    DOUBLE_TYPE x = r * rscalelookup;
    int nl = static_cast<int>(floor(x));

    if (nl <= 0)
        throw std::invalid_argument("Encountered very small distance. Stopping.");

    if (nl < nlut) {
        wl = x - DOUBLE_TYPE(nl);
        wl2 = wl * wl;
        wl3 = wl2 * wl;
        w2l1 = 2.0 * wl;
        w3l2 = 3.0 * wl2;

        for (idx = 0; idx <= 3; idx++) {
            c[idx] = lookupTable(nl, func_ind, idx);
        }
        values(func_ind) = c[0] + c[1] * wl + c[2] * wl2 + c[3] * wl3;
        derivatives(func_ind) = (c[1] + c[2] * w2l1 + c[3] * w3l2) * rscalelookup;

    } else { // fill with zeroes
        values(func_ind) = 0;
        derivatives(func_ind) = 0;
    }
}


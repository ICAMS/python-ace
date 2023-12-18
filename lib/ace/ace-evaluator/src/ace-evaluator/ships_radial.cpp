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

// Created by Christoph Ortner  on 03.06.2020

#include "ace-evaluator/ships_radial.h"

#include <functional>
#include <cmath>
#include <string>


using namespace std;

void SHIPsRadPolyBasis::_init(DOUBLE_TYPE r0, int p, DOUBLE_TYPE rcut,
                              DOUBLE_TYPE xl, DOUBLE_TYPE xr,
                              int pl, int pr, size_t maxn) {
    this->p = p;
    this->r0 = r0;
    this->rcut = rcut;
    this->xl = xl;
    this->xr = xr;
    this->pl = pl;
    this->pr = pr;
    this->maxn = maxn;
    this->A.resize(maxn);
    this->B.resize(maxn);
    this->C.resize(maxn);
    this->P.resize(maxn);
    this->dP_dr.resize(maxn);
}


void SHIPsRadPolyBasis::fread(FILE *fptr) {
    int res; //for fscanf result
    int maxn, p, pl, pr, ntests;
    double r0, xl, xr, a, b, c, rcut;

    // transform parameters
    res = fscanf(fptr, "transform parameters: p=%d r0=%lf\n", &p, &r0);
    if (res != 2)
        throw invalid_argument("Couldn't read line: transform parameters: p=%d r0=%lf");
    // cutoff parameters
    res = fscanf(fptr, "cutoff parameters: rcut=%lf xl=%lf xr=%lf pl=%d pr=%d\n",
                 &rcut, &xl, &xr, &pl, &pr);
    if (res != 5)
        throw invalid_argument("Couldn't read cutoff parameters: rcut=%lf xl=%lf xr=%lf pl=%d pr=%d");
    // basis size
    res = fscanf(fptr, "recursion coefficients: maxn = %d\n", &maxn);
    if (res != 1)
        throw invalid_argument("Couldn't read recursion coefficients: maxn = %d");
    // initialize and allocate
    this->_init(r0, p, rcut, xl, xr, pl, pr, maxn);

    // read basis coefficients
    for (int i = 0; i < maxn; i++) {
        res = fscanf(fptr, " %lf %lf %lf\n", &a, &b, &c);
        if (res != 3)
            throw invalid_argument("Couldn't read line: A_n B_n C_n");
        this->A(i) = DOUBLE_TYPE(a);
        this->B(i) = DOUBLE_TYPE(b);
        this->C(i) = DOUBLE_TYPE(c);
    }

    // // check there are no consistency tests (I don't have time to fix this now)
    // res = fscanf(fptr, "tests: ntests = %d\n", &ntests);
    // if (res != 1)
    //     throw invalid_argument("Couldn't read line: tests: ntests = %d");
    // if (ntests != 0)
    //     throw invalid_argument("must have ntests = 0!");

    // ---------------------------------------------------------------------
    // run the consistency test this could be moved into a separate function
    double r, Pn, dPn;
    double err = 0.0;

    res = fscanf(fptr, "tests: ntests = %d\n", &ntests);
    if (res != 1)
        throw invalid_argument("Couldn't read line: tests: ntests = %d");
    for (int itest = 0; itest < ntests; itest++) {
        // read an r argument
        res = fscanf(fptr, " r=%lf\n", &r);
        // printf("r = %lf \n", r);
        if (res != 1)
            throw invalid_argument("Couldn't read line: r=%lf");
        // printf("test %d, r=%f, maxn=%d \n", itest, r, maxn);
        // evaluate the basis
        this->calcP(r, maxn, SPECIES_TYPE(0), SPECIES_TYPE(0));
        // compare against the stored values
        for (int n = 0; n < maxn; n++) {
            res = fscanf(fptr, " %lf %lf\n", &Pn, &dPn);
            if (res != 2)
                throw invalid_argument("Couldn't read test value line: %lf %lf");
            err = max(err, abs(Pn - this->P(n)) + abs(dPn - this->dP_dr(n)));
            // printf("   %d  %e   %e \n", int(n), 
            //         abs(Pn - this->P(n)), 
            //         abs(dPn - this->dP_dr(n)));
        }
    }
    if (ntests > 0)
        printf("Maximum Test error = %e\n", err);
    // ---------------------------------------------------------------------

}


void SHIPsRadPolyBasis::read_YAML(YAML_PACE::Node node) {
    auto rcut = node["rcut"].as<DOUBLE_TYPE>();
    auto xl = node["xl"].as<DOUBLE_TYPE>();
    auto pl = node["pl"].as<int>();
    auto r0 = node["r0"].as<DOUBLE_TYPE>();
    auto xr = node["xr"].as<DOUBLE_TYPE>();

    auto maxn = node["maxn"].as<int>();
    auto pr = node["pr"].as<int>();

    auto p = node["p"].as<int>();

    this->_init(r0, p, rcut, xl, xr, pl, pr, maxn);

    auto recursion_coefficients = node["recursion_coefficients"]; //.as<vector<vector<DOUBLE_TYPE>>>();

    if (!recursion_coefficients["A"])
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `recursion_coefficients::A` is not presented");
    if (!recursion_coefficients["B"])
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `recursion_coefficients::B` is not presented");

    if (!recursion_coefficients["C"])
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `recursion_coefficients::C` is not presented");

    auto rec_A = recursion_coefficients["A"].as<vector<DOUBLE_TYPE>>();
    auto rec_B = recursion_coefficients["B"].as<vector<DOUBLE_TYPE>>();
    auto rec_C = recursion_coefficients["C"].as<vector<DOUBLE_TYPE>>();

    if (this->maxn != rec_A.size())
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `maxn` doesn't correspond to the size of `recursion_coefficients:A` list");
    if (this->maxn != rec_B.size())
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `maxn` doesn't correspond to the size of `recursion_coefficients:B` list");
    if (this->maxn != rec_C.size())
        throw runtime_error(
                "SHIPsRadPolyBasis::read_YAML: `maxn` doesn't correspond to the size of `recursion_coefficients:C` list");

    // read basis coefficients
    for (int i = 0; i < maxn; i++) {
        this->A(i) = rec_A.at(i);
        this->B(i) = rec_B.at(i);
        this->C(i) = rec_C.at(i);
    }

}

size_t SHIPsRadPolyBasis::get_maxn() {
    return this->maxn;
}


// Julia code: ((1+r0)/(1+r))^p
void SHIPsRadPolyBasis::transform(const DOUBLE_TYPE r, DOUBLE_TYPE &x_out, DOUBLE_TYPE &dx_out) const {
    x_out = pow((1 + r0) / (1 + r), p); // ==pow( (1 + r) / (1 + r0), -p );
    dx_out = -p * pow((1 + r) / (1 + r0), -p - 1) / (1 + r0);
}

void SHIPsRadPolyBasis::fcut(const DOUBLE_TYPE x, DOUBLE_TYPE &f_out, DOUBLE_TYPE &df_out) const {
    if (((x < xl) && (pl > 0)) || ((x > xr) && (pr > 0))) {
        f_out = 0.0;
        df_out = 0.0;
    } else {
        f_out = pow(x - xl, pl) * pow(x - xr, pr);
        df_out = pl * pow(x - xl, pl - 1) * pow(x - xr, pr) + pow(x - xl, pl) * pr * pow(x - xr, pr - 1);
    }
}

/* ------------------------------------------------------------------------
Julia Code
P[1] = J.A[1] * _fcut_(J.pl, J.tl, J.pr, J.tr, t)
if length(J) == 1; return P; end
P[2] = (J.A[2] * t + J.B[2]) * P[1]
@inbounds for n = 3:length(J)
  P[n] = (J.A[n] * t + J.B[n]) * P[n-1] + J.C[n] * P[n-2]
end
return P
------------------------------------------------------------------------ */

void SHIPsRadPolyBasis::calcP(DOUBLE_TYPE r, size_t maxn,
                              SPECIES_TYPE z1, SPECIES_TYPE z2) {
    if (maxn > this->maxn)
        throw invalid_argument("Given maxn couldn't be larger than global maxn");

    if (maxn > P.get_size())
        throw invalid_argument("Given maxn couldn't be larger than global length of P");

    DOUBLE_TYPE x, dx_dr; // dx -> dx/dr
    transform(r, x, dx_dr);
    // printf("r = %f, x = %f, fcut = %f \n", r, x, fcut(x));
    DOUBLE_TYPE f, df_dx;
    fcut(x, f, df_dx); // df -> df/dx

    //fill with zeros
    P.fill(0);
    dP_dr.fill(0);

    P(0) = A(0) * f;
    dP_dr(0) = A(0) * df_dx * dx_dr; // dP/dr;  chain rule: df_cut/dr = df_cut/dx * dx/dr
    if (maxn > 0) {
        P(1) = (A(1) * x + B(1)) * P(0);
        dP_dr(1) = A(1) * dx_dr * P(0) + (A(1) * x + B(1)) * dP_dr(0);
    }
    for (size_t n = 2; n < maxn; n++) {
        P(n) = (A(n) * x + B(n)) * P(n - 1) + C(n) * P(n - 2);
        dP_dr(n) = A(n) * dx_dr * P(n - 1) + (A(n) * x + B(n)) * dP_dr(n - 1) + C(n) * dP_dr(n - 2);
    }
}


// ====================================================================


bool SHIPsRadialFunctions::has_pair() {
    return this->haspair;
}

void SHIPsRadialFunctions::load(string fname) {
    FILE *fptr = fopen(fname.data(), "r");
    size_t res = fscanf(fptr, "radbasename=ACE.jl.Basic\n");
    if (res != 0)
        throw ("SHIPsRadialFunctions::load : couldnt read radbasename=ACE.jl.Basic");
    this->fread(fptr);
    fclose(fptr);
}

void SHIPsRadialFunctions::fread(FILE *fptr) {
    int res;
    size_t maxn;
    char haspair;
    DOUBLE_TYPE c;

    // check whether we have a pair potential 
    res = fscanf(fptr, "haspair: %c\n", &haspair);
    if (res != 1)
        throw ("SHIPsRadialFunctions::load : couldn't read haspair");

    if (this->radbasis.get_size() == 0) {
        this->radbasis.init(1, 1, "SHIPsRadialFunctions::radbasis");
    }
    // read the radial basis 
    this->radbasis(0, 0).fread(fptr);

    // read the pair potential 
    if (haspair == 't') {
        this->haspair = true;
        res = fscanf(fptr, "begin repulsive potential\n");
        res = fscanf(fptr, "begin polypairpot\n");
        // read the basis parameters
        pairbasis.fread(fptr);
        maxn = pairbasis.get_maxn();
        // read the coefficients 
        res = fscanf(fptr, "coefficients\n");
        paircoeffs.init(1, 1, maxn);
        for (size_t n = 0; n < maxn; n++) {
            res = fscanf(fptr, "%lf\n", &c);
            paircoeffs(0, 0, n) = c;
        }
        res = fscanf(fptr, "end polypairpot\n");
        // read the spline parameters
        ri.resize(1, 1);
        e0.resize(1, 1);
        A.resize(1, 1);
        B.resize(1, 1);
        res = fscanf(fptr, "spline parameters\n");
        res = fscanf(fptr, "   e_0 + B  exp(-A*(r/ri-1)) * (ri/r)\n");
        res = fscanf(fptr, "ri=%lf\n", &(this->ri(0, 0)));
        res = fscanf(fptr, "e0=%lf\n", &(this->e0(0, 0)));
        res = fscanf(fptr, "A=%lf\n", &(this->A(0, 0)));
        res = fscanf(fptr, "B=%lf\n", &(this->B(0, 0)));
        res = fscanf(fptr, "end repulsive potential\n");
    }
}

void SHIPsRadialFunctions::read_yaml(YAML_PACE::Node node) {
    // node - top-most YAML node
    if (node["polypairpot"]) {
        auto yaml_pairplot = node["polypairpot"];
        this->haspair = true;
        this->pairbasis.read_YAML(yaml_pairplot);
        //read paircoeffs
        auto maxn = pairbasis.get_maxn();
        paircoeffs.init(nelements, nelements, maxn, "SHIPsRadialFunctions::paircoeffs");
        if (!yaml_pairplot["coefficients"])
            throw runtime_error("`polypairpot::coefficients` not provided");
        auto yaml_coefficients = yaml_pairplot["coefficients"].as<map<vector<int>, vector<DOUBLE_TYPE>>>();

        for (const auto &p: yaml_coefficients) {
            SPECIES_TYPE mu_i = p.first[0];
            SPECIES_TYPE mu_j = p.first[1];
            if (mu_i > nelements - 1 || mu_j > nelements - 1)
                throw invalid_argument("yace::polypairpot::coefficients has species type key larger than nelements");
            auto coefficients = p.second;

            for (unsigned int i = 0; i < maxn; i++)
                this->paircoeffs(mu_i, mu_j, i) = coefficients.at(i);
        }

        if (node["reppot"] && node["reppot"]["coefficients"]) {
            auto reppot_coefficients = node["reppot"]["coefficients"].as<map<vector<int>, YAML_PACE::Node>>();;

            for (const auto &p: reppot_coefficients) {
                SPECIES_TYPE mu_i = p.first[0];
                SPECIES_TYPE mu_j = p.first[1];
                if (mu_i > nelements - 1 || mu_j > nelements - 1)
                    throw invalid_argument("yace::bonds has species type key larger than nelements");
                auto reppot_coef_yaml = p.second;

                this->ri(mu_i, mu_j) = reppot_coef_yaml["ri"].as<DOUBLE_TYPE>();
                this->e0(mu_i, mu_j) = reppot_coef_yaml["e0"].as<DOUBLE_TYPE>();
                this->A(mu_i, mu_j) = reppot_coef_yaml["A"].as<DOUBLE_TYPE>();
                this->B(mu_i, mu_j) = reppot_coef_yaml["B"].as<DOUBLE_TYPE>();
            }
        }
    }
    //reading bonds
    auto yaml_map_bond_specifications = node["bonds"].as<map<vector<int>, YAML_PACE::Node>>();
    for (const auto &p: yaml_map_bond_specifications) {
        SPECIES_TYPE mu_i = p.first[0];
        SPECIES_TYPE mu_j = p.first[1];
        if (mu_i > nelements - 1 || mu_j > nelements - 1)
            throw invalid_argument("yace::bonds has species type key larger than nelements");
        auto bond_yaml = p.second;
        this->radbasis(mu_i, mu_j).read_YAML(bond_yaml);
    }
}

size_t SHIPsRadialFunctions::get_maxn() {
    int maxn = 0;
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; mu_i++)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; mu_j++) {
            int cur_maxn = this->radbasis(mu_i, mu_j).get_maxn();
            if (cur_maxn > maxn)
                maxn = cur_maxn;
        }
    return maxn;
}

DOUBLE_TYPE SHIPsRadialFunctions::get_rcut() {
    DOUBLE_TYPE rcut = 0;

    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; mu_i++)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; mu_j++) {
            auto cur_rcut = this->radbasis(mu_i, mu_j).rcut;
            if (cur_rcut > rcut)
                rcut = cur_rcut;
        }
    return max(rcut, pairbasis.rcut);
}


void SHIPsRadialFunctions::fill_gk(DOUBLE_TYPE r, NS_TYPE maxn, SPECIES_TYPE z1, SPECIES_TYPE z2) {
    radbasis(z1, z2).calcP(r, maxn, z1, z2);
    for (NS_TYPE n = 0; n < maxn; n++) {
        gr(n) = radbasis(z1, z2).P(n);
        dgr(n) = radbasis(z1, z2).dP_dr(n);
    }
}


void SHIPsRadialFunctions::fill_Rnl(DOUBLE_TYPE r, NS_TYPE maxn, SPECIES_TYPE z1, SPECIES_TYPE z2) {
    radbasis(z1, z2).calcP(r, maxn, z1, z2);
    for (NS_TYPE n = 0; n < maxn; n++) {
        for (LS_TYPE l = 0; l <= lmax; l++) {
            fr(n, l) = radbasis(z1, z2).P(n);
            dfr(n, l) = radbasis(z1, z2).dP_dr(n);
        }
    }
}


void SHIPsRadialFunctions::setuplookupRadspline() {
}

void SHIPsRadialFunctions::init(SPECIES_TYPE nelements) {
    this->nelements = nelements;

    lambda.init(nelements, nelements, "lambda");
    lambda.fill(1.);

    cut.init(nelements, nelements, "cut");
    cut.fill(1.);

    dcut.init(nelements, nelements, "dcut");
    dcut.fill(1.);

    //hard-core repulsion
    prehc.init(nelements, nelements, "prehc");
    prehc.fill(0.);

    lambdahc.init(nelements, nelements, "lambdahc");
    lambdahc.fill(1.);

    radbasis.init(nelements, nelements, "SHIPsRadialFunctions::radbasis");
    ri.init(nelements, nelements, "SHIPsRadialFunctions::ri");
    e0.init(nelements, nelements, "SHIPsRadialFunctions::e0");
    A.init(nelements, nelements, "SHIPsRadialFunctions::A");
    B.init(nelements, nelements, "SHIPsRadialFunctions::B");
    ri.fill(0);
    e0.fill(0);
    A.fill(0);
    B.fill(0);
}

void SHIPsRadialFunctions::init(NS_TYPE nradb, LS_TYPE lmax, NS_TYPE nradial, DOUBLE_TYPE deltaSplineBins,
                                SPECIES_TYPE nelements,
                                vector<vector<string>> radbasename) {
    //mimic ACERadialFunctions::init
    this->nradbase = nradb;
    this->lmax = lmax;
    this->nradial = nradial;
    this->deltaSplineBins = deltaSplineBins;

    this->init(nelements);


    gr.init(nradbase, "gr");
    dgr.init(nradbase, "dgr");


    fr.init(nradial, lmax + 1, "fr");
    dfr.init(nradial, lmax + 1, "dfr");

    crad.init(nelements, nelements, nradial, (lmax + 1), nradbase, "crad");
    crad.fill(0.);

}


void SHIPsRadialFunctions::evaluate(DOUBLE_TYPE r, NS_TYPE nradbase_c, NS_TYPE nradial_c, SPECIES_TYPE mu_i,
                                    SPECIES_TYPE mu_j, bool calc_second_derivatives) {
    if (calc_second_derivatives)
        throw invalid_argument("SHIPsRadialFunctions has not `calc_second_derivatives` option");

    radbasis(mu_i, mu_j).calcP(r, nradbase_c, mu_i, mu_j);
    for (NS_TYPE nr = 0; nr < nradbase_c; nr++) {
        gr(nr) = radbasis(mu_i, mu_j).P(nr);
        dgr(nr) = radbasis(mu_i, mu_j).dP_dr(nr);
    }
    for (NS_TYPE nr = 0; nr < nradial_c; nr++) {
        for (LS_TYPE l = 0; l <= this->lmax; l++) {
            fr(nr, l) = radbasis(mu_i, mu_j).P(nr);
            dfr(nr, l) = radbasis(mu_i, mu_j).dP_dr(nr);
        }
    }

    if (this->has_pair())
        this->evaluate_pair(r, mu_i, mu_j);
    else {
        cr = 0;
        dcr = 0;
    }
}

void SHIPsRadialFunctions::evaluate_pair(DOUBLE_TYPE r,
                                         SPECIES_TYPE mu_i,
                                         SPECIES_TYPE mu_j,
                                         bool calc_second_derivatives) {
    // the outer polynomial potential
    if (r > ri(mu_i, mu_j)) {
        pairbasis.calcP(r, pairbasis.get_maxn(), mu_i, mu_j);
        cr = 0;
        dcr = 0;
        for (size_t n = 0; n < pairbasis.get_maxn(); n++) {
            cr += paircoeffs(mu_i, mu_j, n) * pairbasis.P(n);
            dcr += paircoeffs(mu_i, mu_j, n) * pairbasis.dP_dr(n);
        }
    } else { // the repulsive core part
        cr = e0(mu_i, mu_j) + B(mu_i, mu_j) * exp(-A(mu_i, mu_j) * (r / ri(mu_i, mu_j) - 1)) * (ri(mu_i, mu_j) / r);
        dcr = B(mu_i, mu_j) * exp(-A(mu_i, mu_j) * (r / ri(mu_i, mu_j) - 1)) * ri(mu_i, mu_j) *
              (-A(mu_i, mu_j) / ri(mu_i, mu_j) / r - 1 / (r * r));
    }
    // fix double-counting
    cr *= 0.5;
    dcr *= 0.5;
}


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

// Created by Yury Lysogorskiy on 1.04.20.
#include <fstream>

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ships_radial.h"

using namespace std;

inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

ACECTildeBasisSet::ACECTildeBasisSet(string filename) {
    if (ends_with(filename, ".ace"))
        load(filename);
    else if (ends_with(filename, ".yace"))
        load_yaml(filename);
    else
        throw invalid_argument("Unrecognized file type of '" + filename + "'. Accept only .ace or .yace");
}

ACECTildeBasisSet::ACECTildeBasisSet(const ACECTildeBasisSet &other) {
    ACECTildeBasisSet::_copy_scalar_memory(other);
    ACECTildeBasisSet::_copy_dynamic_memory(other);
    pack_flatten_basis();
}


ACECTildeBasisSet &ACECTildeBasisSet::operator=(const ACECTildeBasisSet &other) {
    if (this != &other) {
        _clean();
        _copy_scalar_memory(other);
        _copy_dynamic_memory(other);
        pack_flatten_basis();
    }
    return *this;
}


ACECTildeBasisSet::~ACECTildeBasisSet() {
    ACECTildeBasisSet::_clean();
}

void ACECTildeBasisSet::_clean() {
    // call parent method
    ACEFlattenBasisSet::_clean();
    _clean_contiguous_arrays();
    _clean_basis_arrays();
}

void ACECTildeBasisSet::_copy_scalar_memory(const ACECTildeBasisSet &src) {
    ACEFlattenBasisSet::_copy_scalar_memory(src);
    num_ctilde_max = src.num_ctilde_max;
}

void ACECTildeBasisSet::_copy_dynamic_memory(const ACECTildeBasisSet &src) {//allocate new memory
    ACEFlattenBasisSet::_copy_dynamic_memory(src);

    if (src.basis_rank1 == nullptr)
        throw runtime_error("Could not copy ACECTildeBasisSet::basis_rank1 - array not initialized");
    if (src.basis == nullptr) throw runtime_error("Could not copy ACECTildeBasisSet::basis - array not initialized");


    basis_rank1 = new ACECTildeBasisFunction *[src.nelements];
    basis = new ACECTildeBasisFunction *[src.nelements];

    //copy basis arrays
    for (SPECIES_TYPE mu = 0; mu < src.nelements; ++mu) {
        basis_rank1[mu] = new ACECTildeBasisFunction[src.total_basis_size_rank1[mu]];

        for (size_t i = 0; i < src.total_basis_size_rank1[mu]; i++) {
            basis_rank1[mu][i] = src.basis_rank1[mu][i];
        }


        basis[mu] = new ACECTildeBasisFunction[src.total_basis_size[mu]];
        for (size_t i = 0; i < src.total_basis_size[mu]; i++) {
            basis[mu][i] = src.basis[mu][i];
        }
    }
    //DON"T COPY CONTIGUOUS ARRAY, REBUILD THEM
}

void ACECTildeBasisSet::_clean_contiguous_arrays() {
    ACEFlattenBasisSet::_clean_contiguous_arrays();

    delete[] full_c_tildes_rank1;
    full_c_tildes_rank1 = nullptr;

    delete[] full_c_tildes;
    full_c_tildes = nullptr;
}

//re-pack the constituent dynamic arrays of all basis functions in contiguous arrays
void ACECTildeBasisSet::pack_flatten_basis() {
    compute_array_sizes(basis_rank1, basis);

    //1. clean contiguous arrays
    _clean_contiguous_arrays();
    //2. allocate contiguous arrays
    delete[] full_ns_rank1;
    full_ns_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    delete[] full_ls_rank1;
    full_ls_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    delete[] full_mus_rank1;
    full_mus_rank1 = new SPECIES_TYPE[rank_array_total_size_rank1];
    delete[] full_ms_rank1;
    full_ms_rank1 = new MS_TYPE[rank_array_total_size_rank1];

    delete[] full_c_tildes_rank1;
    full_c_tildes_rank1 = new DOUBLE_TYPE[coeff_array_total_size_rank1];


    delete[] full_ns;
    full_ns = new NS_TYPE[rank_array_total_size];
    delete[] full_ls;
    full_ls = new LS_TYPE[rank_array_total_size];
    delete[] full_mus;
    full_mus = new SPECIES_TYPE[rank_array_total_size];
    delete[] full_ms;
    full_ms = new MS_TYPE[ms_array_total_size];

    delete[] full_c_tildes;
    full_c_tildes = new DOUBLE_TYPE[coeff_array_total_size];


    //3. copy the values from private C_tilde_B_basis_function arrays to new contigous space
    //4. clean private memory
    //5. reassign private array pointers

    //r = 0, rank = 1
    size_t rank_array_ind_rank1 = 0;
    size_t coeff_array_ind_rank1 = 0;
    size_t ms_array_ind_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind_r1 = 0; func_ind_r1 < total_basis_size_rank1[mu]; ++func_ind_r1) {
            ACECTildeBasisFunction &func = basis_rank1[mu][func_ind_r1];

            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            full_ns_rank1[rank_array_ind_rank1] = func.ns[0];

            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            full_ls_rank1[rank_array_ind_rank1] = func.ls[0];

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            full_mus_rank1[rank_array_ind_rank1] = func.mus[0];

            //copy values ctildes from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_c_tildes_rank1[coeff_array_ind_rank1], func.ctildes,
                   func.ndensity * sizeof(DOUBLE_TYPE));


            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms_rank1[ms_array_ind_rank1], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.mus = &full_mus_rank1[rank_array_ind_rank1];
            func.ns = &full_ns_rank1[rank_array_ind_rank1];
            func.ls = &full_ls_rank1[rank_array_ind_rank1];
            func.ms_combs = &full_ms_rank1[ms_array_ind_rank1];
            func.ctildes = &full_c_tildes_rank1[coeff_array_ind_rank1];
            func.is_proxy = true;

            rank_array_ind_rank1 += func.rank;
            ms_array_ind_rank1 += func.rank *
                                  func.num_ms_combs;
            coeff_array_ind_rank1 += func.num_ms_combs * func.ndensity;

        }
    }


    //rank>1, r>0
    size_t rank_array_ind = 0;
    size_t coeff_array_ind = 0;
    size_t ms_array_ind = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            ACECTildeBasisFunction &func = basis[mu][func_ind];

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_mus[rank_array_ind], func.mus,
                   func.rank * sizeof(SPECIES_TYPE));

            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ns[rank_array_ind], func.ns,
                   func.rank * sizeof(NS_TYPE));
            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ls[rank_array_ind], func.ls,
                   func.rank * sizeof(LS_TYPE));
            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms[ms_array_ind], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //copy values ctildes from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_c_tildes[coeff_array_ind], func.ctildes,
                   func.num_ms_combs * func.ndensity * sizeof(DOUBLE_TYPE));


            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.ns = &full_ns[rank_array_ind];
            func.ls = &full_ls[rank_array_ind];
            func.mus = &full_mus[rank_array_ind];
            func.ctildes = &full_c_tildes[coeff_array_ind];
            func.ms_combs = &full_ms[ms_array_ind];
            func.is_proxy = true;

            rank_array_ind += func.rank;
            ms_array_ind += func.rank *
                            func.num_ms_combs;
            coeff_array_ind += func.num_ms_combs * func.ndensity;
        }
    }
}

void fwrite_c_tilde_b_basis_func(FILE *fptr, ACECTildeBasisFunction &func) {
    RANK_TYPE r;
    fprintf(fptr, "ctilde_basis_func: ");
    fprintf(fptr, "rank=%d ndens=%d mu0=%d ", func.rank, func.ndensity, func.mu0);

    fprintf(fptr, "mu=(");
    for (r = 0; r < func.rank; ++r)
        fprintf(fptr, " %d ", func.mus[r]);
    fprintf(fptr, ")\n");

    fprintf(fptr, "n=(");
    for (r = 0; r < func.rank; ++r)
        fprintf(fptr, " %d ", func.ns[r]);
    fprintf(fptr, ")\n");

    fprintf(fptr, "l=(");
    for (r = 0; r < func.rank; ++r)
        fprintf(fptr, " %d ", func.ls[r]);
    fprintf(fptr, ")\n");

    fprintf(fptr, "num_ms=%d\n", func.num_ms_combs);

    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++) {
        fprintf(fptr, "<");
        for (r = 0; r < func.rank; ++r)
            fprintf(fptr, " %d ", func.ms_combs[m_ind * func.rank + r]);
        fprintf(fptr, ">: ");
        for (DENSITY_TYPE p = 0; p < func.ndensity; p++)
            fprintf(fptr, " %.18f ", func.ctildes[m_ind * func.ndensity + p]);
        fprintf(fptr, "\n");
    }

}

void ACECTildeBasisSet::save(const string &filename) {
    FILE *fptr;
    fptr = fopen(filename.c_str(), "w");

    fprintf(fptr, "nelements=%d\n", nelements);

    //elements mapping
    fprintf(fptr, "elements:");
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu)
        fprintf(fptr, " %s", elements_name[mu].c_str());
    fprintf(fptr, "\n\n");

    fprintf(fptr, "lmax=%d\n\n", lmax);

//    fprintf(fptr, "embedding-function: %s\n", npoti.c_str());
    fprintf(fptr, "embedding-function: %s\n", map_embedding_specifications[0].npoti.c_str());

//    fprintf(fptr, "%ld FS parameters: ", FS_parameters.size());
//    for (int i = 0; i < FS_parameters.size(); ++i) {
//        fprintf(fptr, " %f", FS_parameters.at(i));
//    }

    //TODO: hardcoded for single species
    fprintf(fptr, "%ld FS parameters: ", map_embedding_specifications[0].FS_parameters.size());
    for (int i = 0; i < map_embedding_specifications[0].FS_parameters.size(); ++i) {
        fprintf(fptr, " %f", map_embedding_specifications[0].FS_parameters.at(i));
    }
    fprintf(fptr, "\n");

    //hard-core energy cutoff repulsion
    fprintf(fptr, "core energy-cutoff parameters: ");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i) {
        fprintf(fptr, "%.18f %.18f\n", map_embedding_specifications.at(mu_i).rho_core_cutoff,
                map_embedding_specifications.at(mu_i).drho_core_cutoff);
    }

    // save E0 values 
    fprintf(fptr, "E0:");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        fprintf(fptr, " %.18f", E0vals(mu_i));
    fprintf(fptr, "\n");


    fprintf(fptr, "\n");


    fprintf(fptr, "radbasename=%s\n", radial_functions->radbasenameij(0, 0).c_str());//TODO-single species
    fprintf(fptr, "nradbase=%d\n", nradbase);
    fprintf(fptr, "nradmax=%d\n", nradmax);


    fprintf(fptr, "cutoffmax=%f\n", cutoffmax);

    fprintf(fptr, "deltaSplineBins=%f\n", deltaSplineBins);

    //hard-core repulsion
    fprintf(fptr, "core repulsion parameters: ");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j)
            fprintf(fptr, "%.18f %.18f\n", radial_functions->prehc(mu_i, mu_j), radial_functions->lambdahc(mu_j, mu_j));





    //TODO: radial functions
    //radparameter
    fprintf(fptr, "radparameter=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j)
            fprintf(fptr, " %.18f", radial_functions->lambda(mu_i, mu_j));
    fprintf(fptr, "\n");

    fprintf(fptr, "cutoff=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j)
            fprintf(fptr, " %.18f", radial_functions->cut(mu_i, mu_j));
    fprintf(fptr, "\n");

    fprintf(fptr, "dcut=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j)
            fprintf(fptr, " %.18f", radial_functions->dcut(mu_i, mu_j));
    fprintf(fptr, "\n");

    fprintf(fptr, "crad=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            for (NS_TYPE k = 0; k < nradbase; k++) {
                for (NS_TYPE n = 0; n < nradmax; n++) {
                    for (LS_TYPE l = 0; l <= lmax; l++) {
                        fprintf(fptr, " %.18f", radial_functions->crad(mu_i, mu_j, n, l, k));
                    }
                    fprintf(fptr, "\n");
                }
            }
        }

    fprintf(fptr, "\n");

    fprintf(fptr, "rankmax=%d\n", rankmax);
    fprintf(fptr, "ndensitymax=%d\n", ndensitymax);
    fprintf(fptr, "\n");

    //num_c_tilde_max
    fprintf(fptr, "num_c_tilde_max=%d\n", num_ctilde_max);
    fprintf(fptr, "num_ms_combinations_max=%d\n", num_ms_combinations_max);


    //write total_basis_size and total_basis_size_rank1
    fprintf(fptr, "total_basis_size_rank1: ");
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        fprintf(fptr, "%d ", total_basis_size_rank1[mu]);
    }
    fprintf(fptr, "\n");

    for (SPECIES_TYPE mu = 0; mu < nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; ++func_ind)
            fwrite_c_tilde_b_basis_func(fptr, basis_rank1[mu][func_ind]);

    fprintf(fptr, "total_basis_size: ");
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        fprintf(fptr, "%d ", total_basis_size[mu]);
    }
    fprintf(fptr, "\n");

    for (SPECIES_TYPE mu = 0; mu < nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind)
            fwrite_c_tilde_b_basis_func(fptr, basis[mu][func_ind]);


    fclose(fptr);
}

void fread_c_tilde_b_basis_func(FILE *fptr, ACECTildeBasisFunction &func) {
    RANK_TYPE r;
    int res;
    char buf[3][128];

    res = fscanf(fptr, " ctilde_basis_func: ");

    res = fscanf(fptr, "rank=%s ndens=%s mu0=%s ", buf[0], buf[1], buf[2]);
    if (res != 3)
        throw invalid_argument("Could not read C-tilde basis function");

    func.rank = (RANK_TYPE) stol(buf[0]);
    func.ndensity = (DENSITY_TYPE) stol(buf[1]);
    func.mu0 = (SPECIES_TYPE) stol(buf[2]);

    func.mus = new SPECIES_TYPE[func.rank];
    func.ns = new NS_TYPE[func.rank];
    func.ls = new LS_TYPE[func.rank];

    res = fscanf(fptr, " mu=(");
    for (r = 0; r < func.rank; ++r) {
        res = fscanf(fptr, "%s", buf[0]);
        if (res != 1)
            throw invalid_argument("Could not read C-tilde basis function");
        func.mus[r] = (SPECIES_TYPE) stol(buf[0]);
    }
    res = fscanf(fptr, " )"); // ")"

    res = fscanf(fptr, " n=("); // "n="
    for (r = 0; r < func.rank; ++r) {
        res = fscanf(fptr, "%s", buf[0]);
        if (res != 1)
            throw invalid_argument("Could not read C-tilde basis function");

        func.ns[r] = (NS_TYPE) stol(buf[0]);
    }
    res = fscanf(fptr, " )");

    res = fscanf(fptr, " l=(");
    for (r = 0; r < func.rank; ++r) {
        res = fscanf(fptr, "%s", buf[0]);
        if (res != 1)
            throw invalid_argument("Could not read C-tilde basis function");
        func.ls[r] = (NS_TYPE) stol(buf[0]);
    }
    res = fscanf(fptr, " )");

    res = fscanf(fptr, " num_ms=%s\n", buf[0]);
    if (res != 1)
        throw invalid_argument("Could not read C-tilde basis function");

    func.num_ms_combs = (SHORT_INT_TYPE) stoi(buf[0]);

    func.ms_combs = new MS_TYPE[func.rank * func.num_ms_combs];
    func.ctildes = new DOUBLE_TYPE[func.ndensity * func.num_ms_combs];

    for (int m_ind = 0; m_ind < func.num_ms_combs; m_ind++) {
        res = fscanf(fptr, " <");
        for (r = 0; r < func.rank; ++r) {
            res = fscanf(fptr, "%s", buf[0]);
            if (res != 1)
                throw invalid_argument("Could not read C-tilde basis function");
            func.ms_combs[m_ind * func.rank + r] = stoi(buf[0]);
        }
        res = fscanf(fptr, " >:");
        for (DENSITY_TYPE p = 0; p < func.ndensity; p++) {
            res = fscanf(fptr, "%s", buf[0]);
            if (res != 1)
                throw invalid_argument("Could not read C-tilde basis function");
            func.ctildes[m_ind * func.ndensity + p] = stod(buf[0]);
        }
    }
}

string
format_error_message(const char *buffer, const string &filename, const string &var_name, const string &expected) {
    string err_message = "File '" + filename + "': couldn't read '" + var_name + "'";
    if (buffer)
        err_message = err_message + ", read:'" + buffer + "'";
    if (!expected.empty())
        err_message = err_message + ". Expected format: '" + expected + "'";
    return err_message;
}

void throw_error(const string &filename, const string &var_name, const string expected = "") {
    throw invalid_argument(format_error_message(nullptr, filename, var_name, expected));
}

DOUBLE_TYPE stod_err(const char *buffer, const string &filename, const string &var_name, const string expected = "") {
    try {
        return stod(buffer);
    } catch (invalid_argument &exc) {
        throw invalid_argument(format_error_message(buffer, filename, var_name, expected).c_str());
    }
}

int stoi_err(const char *buffer, const string &filename, const string &var_name, const string expected = "") {
    try {
        return stoi(buffer);
    } catch (invalid_argument &exc) {
        throw invalid_argument(format_error_message(buffer, filename, var_name, expected).c_str());
    }
}

long int stol_err(const char *buffer, const string &filename, const string &var_name, const string expected = "") {
    try {
        return stol(buffer);
    } catch (invalid_argument &exc) {
        throw invalid_argument(format_error_message(buffer, filename, var_name, expected).c_str());
    }
}

void ACECTildeBasisSet::load(const string filename) {
    int res;
    char buffer[1024], buffer2[1024];
    string radbasename = "ChebExpCos";

    FILE *fptr;
    fptr = fopen(filename.c_str(), "r");
    if (fptr == NULL)
        throw invalid_argument("Could not open file " + filename);

    //read number of elements
    res = fscanf(fptr, " nelements=%s\n", buffer);
    //try to read the buffer one more time
    char *rr;
    while (res != 1) {
        rr = fgets(buffer, 1024, fptr);
        if (!rr)
            break;
        res = fscanf(fptr, " nelements=%s\n", buffer);
    }

    if (res != 1)
        throw_error(filename, "nelements", "nelements=[number]");
    nelements = stoi_err(buffer, filename, "nelements", "nelements=[number]");

    //elements mapping
    elements_name = new string[nelements];
    res = fscanf(fptr, " elements:");
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        res = fscanf(fptr, "%s", buffer);
        if (res != 1)
            throw_error(filename, "elements", "elements: Ele1 Ele2 ...");
        elements_name[mu] = buffer;
    }
    // initialize embedding_specifications
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i) {
        map_embedding_specifications[mu_i] = ACEEmbeddingSpecification();
    }
    // load angular basis - only need spherical harmonics parameter 
    res = fscanf(fptr, " lmax=%s\n", buffer);
    if (res != 1)
        throw_error(filename, "lmax", "lmax=[number]");
    lmax = stoi_err(buffer, filename, "lmax", "lmax=[number]");
    spherical_harmonics.init(lmax);


    // reading "embedding-function:"
    bool is_embedding_function_specified = false;
    res = fscanf(fptr, " embedding-function: %s", buffer);
    if (res == 0) {
        //throw_error(filename, "E0", " E0: E0-species1 E0-species2 ...");
//        this->npoti = "FinnisSinclair"; // default values
        map_embedding_specifications[0].npoti = "FinnisSinclair"; // default values
        //printf("Warning! No embedding-function is specified, embedding-function: FinnisSinclair would be assumed\n");
        is_embedding_function_specified = false;
    } else {
//        this->npoti = buffer;
        map_embedding_specifications[0].npoti = buffer;
        is_embedding_function_specified = true;
    }
    //TODO: hard-coded for single-species case
//    this->map_embedding_specifications.at(0).npoti = this->npoti;


    int parameters_size;
    res = fscanf(fptr, "%s FS parameters:", buffer);
    if (res != 1)
        throw_error(filename, "FS parameters size", "[number] FS parameters: par1 par2 ...");
    parameters_size = stoi_err(buffer, filename, "FS parameters size", "[number] FS parameters");

    vector<DOUBLE_TYPE> FS_parameters;
    FS_parameters.resize(parameters_size);
    for (int i = 0; i < FS_parameters.size(); ++i) {
        res = fscanf(fptr, "%s", buffer);
        if (res != 1)
            throw_error(filename, "FS parameter", "[number] FS parameters: [par1] [par2] ...");
        FS_parameters[i] = stod_err(buffer, filename, "FS parameter", "[number] FS parameters: [par1] [par2] ...");
    }

    //TODO: hard-coded for single-species case
    this->map_embedding_specifications.at(0).FS_parameters = FS_parameters;
//    this->map_embedding_specifications.at(0).npoti = this->npoti;

    if (!is_embedding_function_specified) {
        // assuming non-linear potential, and embedding function type is important
        for (int j = 1; j < parameters_size; j += 2)
            if (FS_parameters[j] != 1.0) { //if not ensure linearity of embedding function parameters
                printf("ERROR! Your potential is non-linear\n");
                printf("Please specify 'embedding-function: FinnisSinclair' or 'embedding-function: FinnisSinclairShiftedScaled' before 'FS parameters size' line\n");
                throw_error(filename, "embedding-function", "FinnisSinclair or FinnisSinclairShiftedScaled");
            }
        printf("Notice! No embedding-function is specified, but potential has linear embedding, default embedding-function: FinnisSinclair would be assumed\n");
    }

    //hard-core energy cutoff repulsion
    res = fscanf(fptr, " core energy-cutoff parameters:");
    if (res != 0)
        throw_error(filename, "core energy-cutoff parameters", "core energy-cutoff parameters: [par1] [par2]");

//    rho_core_cutoffs.init(nelements, "rho_core_cutoffs");
//    drho_core_cutoffs.init(nelements, "drho_core_cutoffs");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i) {
        res = fscanf(fptr, "%s %s", buffer, buffer2);
        if (res != 2)
            throw_error(filename, "core energy-cutoff parameters",
                        "core energy-cutoff parameters: [rho_core_cut] [drho_core_cutoff] ...");
        DOUBLE_TYPE rho_core_cutoff = stod_err(buffer, filename, "rho core cutoff",
                                               "core energy-cutoff parameters: [rho_core_cut] [drho_core_cutoff] ...");
        DOUBLE_TYPE drho_core_cutoff = stod_err(buffer2, filename, "drho_core_cutoff",
                                                "core energy-cutoff parameters: [rho_core_cut] [drho_core_cutoff] ...");
        map_embedding_specifications.at(mu_i).rho_core_cutoff = rho_core_cutoff;
        map_embedding_specifications.at(mu_i).drho_core_cutoff = drho_core_cutoff;
    }

    // atom energy shift E0 (energy of isolated atom)
    E0vals.init(nelements);

    // reading "E0:"
    res = fscanf(fptr, " E0: %s", buffer);
    if (res == 0) {
        //throw_error(filename, "E0", " E0: E0-species1 E0-species2 ...");
        E0vals.fill(0.0);
    } else {
        double E0 = atof(buffer);
        E0vals(0) = E0;

        for (SPECIES_TYPE mu_i = 1; mu_i < nelements; ++mu_i) {
            res = fscanf(fptr, " %lf", &E0);
            if (res != 1)
                throw_error(filename, "E0", " couldn't read one of the E0 values");
            E0vals(mu_i) = E0;
        }
        res = fscanf(fptr, "\n");
        if (res != 0)
            printf("file %s : format seems broken near E0; trying to continue...\n", filename.c_str());
    }

    // check which radial basis we need to load 
    res = fscanf(fptr, " radbasename=%s\n", buffer);
    if (res != 1) {
        throw_error(filename, "radbasename", "rabbasename=ChebExpCos|ChebPow|ACE.jl.Basic");
    } else {
        radbasename = buffer;
    }

//    printf("radbasename = `%s`\n", radbasename.c_str());
    if (radbasename == "ChebExpCos" | radbasename == "ChebPow") {
        _load_radial_ACERadial(fptr, filename, radbasename);
    } else if (radbasename == "ACE.jl.Basic") {
        _load_radial_SHIPsBasic(fptr, filename, radbasename);
    } else {
        throw invalid_argument(
                ("File '" + filename + "': I don't know how to read radbasename = " + radbasename).c_str());
    }

    res = fscanf(fptr, " rankmax=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "rankmax", "rankmax=[number]");
    rankmax = stoi_err(buffer, filename, "rankmax", "rankmax=[number]");

    res = fscanf(fptr, " ndensitymax=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "ndensitymax", "ndensitymax=[number]");
    ndensitymax = stoi_err(buffer, filename, "ndensitymax", "ndensitymax=[number]");

    //TODO: hardcoded for single-species case
    this->map_embedding_specifications.at(0).ndensity = ndensitymax;

    // read the list of correlations to be put into the basis
    //num_c_tilde_max
    res = fscanf(fptr, " num_c_tilde_max=");
    res = fscanf(fptr, "%s\n", buffer);
    if (res != 1)
        throw_error(filename, "num_c_tilde_max", "num_c_tilde_max=[number]");
    num_ctilde_max = stol_err(buffer, filename, "num_c_tilde_max", "num_c_tilde_max=[number]");

    res = fscanf(fptr, " num_ms_combinations_max=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "num_ms_combinations_max", "num_ms_combinations_max=[number]");
    num_ms_combinations_max = stol_err(buffer, filename, "num_ms_combinations_max", "num_ms_combinations_max=[number]");

    //read total_basis_size_rank1
    total_basis_size_rank1 = new SHORT_INT_TYPE[nelements];
    basis_rank1 = new ACECTildeBasisFunction *[nelements];
    res = fscanf(fptr, " total_basis_size_rank1: ");


    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        res = fscanf(fptr, "%s", buffer);
        if (res != 1)
            throw_error(filename, "total_basis_size_rank1", "total_basis_size_rank1: [size_ele1] [size_ele2] ...");
//            throw invalid_argument(("File '" + filename + "': couldn't read total_basis_size_rank1").c_str());
        total_basis_size_rank1[mu] = stoi_err(buffer, filename, "total_basis_size_rank1",
                                              "total_basis_size_rank1: [size_ele1] [size_ele2] ...");
        basis_rank1[mu] = new ACECTildeBasisFunction[total_basis_size_rank1[mu]];
    }
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; ++func_ind) {
            fread_c_tilde_b_basis_func(fptr, basis_rank1[mu][func_ind]);
        }

    //read total_basis_size
    res = fscanf(fptr, " total_basis_size: ");
    total_basis_size = new SHORT_INT_TYPE[nelements];
    basis = new ACECTildeBasisFunction *[nelements];

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        res = fscanf(fptr, "%s", buffer);
        if (res != 1)
            throw_error(filename, "total_basis_size", "total_basis_size: [size_ele1] [size_ele2] ...");
        total_basis_size[mu] = stoi_err(buffer, filename, "total_basis_size",
                                        "total_basis_size: [size_ele1] [size_ele2] ...");
        basis[mu] = new ACECTildeBasisFunction[total_basis_size[mu]];
    }
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++)
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            fread_c_tilde_b_basis_func(fptr, basis[mu][func_ind]);
        }

    fclose(fptr);

//    radial_functions->radbasename = radbasename;
    radial_functions->setuplookupRadspline();
    pack_flatten_basis();
}

void ACECTildeBasisSet::compute_array_sizes(ACECTildeBasisFunction **basis_rank1, ACECTildeBasisFunction **basis) {
    //compute arrays sizes
    rank_array_total_size_rank1 = 0;
    //ms_array_total_size_rank1 = rank_array_total_size_rank1;
    coeff_array_total_size_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        if (total_basis_size_rank1[mu] > 0) {
            rank_array_total_size_rank1 += total_basis_size_rank1[mu];

            ACEAbstractBasisFunction &func = basis_rank1[mu][0];//TODO: get total density instead of density from first function
            coeff_array_total_size_rank1 += total_basis_size_rank1[mu] * func.ndensity;
        }
    }

    rank_array_total_size = 0;
    coeff_array_total_size = 0;

    ms_array_total_size = 0;
    max_dB_array_size = 0;


    max_B_array_size = 0;

    size_t cur_ms_size = 0;
    size_t cur_ms_rank_size = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {

        cur_ms_size = 0;
        cur_ms_rank_size = 0;
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            auto &func = basis[mu][func_ind];
            rank_array_total_size += func.rank;
            ms_array_total_size += func.rank * func.num_ms_combs;
            coeff_array_total_size += func.ndensity * func.num_ms_combs;

            cur_ms_size += func.num_ms_combs;
            cur_ms_rank_size += func.rank * func.num_ms_combs;
        }

        if (cur_ms_size > max_B_array_size)
            max_B_array_size = cur_ms_size;

        if (cur_ms_rank_size > max_dB_array_size)
            max_dB_array_size = cur_ms_rank_size;
    }
}

void ACECTildeBasisSet::_clean_basis_arrays() {
    if (basis_rank1 != nullptr)
        for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
            delete[] basis_rank1[mu];
            basis_rank1[mu] = nullptr;
        }

    if (basis != nullptr)
        for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
            delete[] basis[mu];
            basis[mu] = nullptr;
        }
    delete[] basis;
    basis = nullptr;

    delete[] basis_rank1;
    basis_rank1 = nullptr;
}

//pack into 1D array with all basis functions
void ACECTildeBasisSet::flatten_basis(C_tilde_full_basis_vector2d &mu0_ctilde_basis_vector) {

    _clean_basis_arrays();
    basis_rank1 = new ACECTildeBasisFunction *[nelements];
    basis = new ACECTildeBasisFunction *[nelements];

    delete[] total_basis_size_rank1;
    delete[] total_basis_size;
    total_basis_size_rank1 = new SHORT_INT_TYPE[nelements];
    total_basis_size = new SHORT_INT_TYPE[nelements];


    size_t tot_size_rank1 = 0;
    size_t tot_size = 0;

    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        tot_size = 0;
        tot_size_rank1 = 0;

        for (auto &func: mu0_ctilde_basis_vector[mu]) {
            if (func.rank == 1) tot_size_rank1 += 1;
            else tot_size += 1;
        }

        total_basis_size_rank1[mu] = tot_size_rank1;
        basis_rank1[mu] = new ACECTildeBasisFunction[tot_size_rank1];

        total_basis_size[mu] = tot_size;
        basis[mu] = new ACECTildeBasisFunction[tot_size];
    }


    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        size_t ind_rank1 = 0;
        size_t ind = 0;

        for (auto &func: mu0_ctilde_basis_vector[mu]) {
            if (func.rank == 1) { //r=0, rank=1
                basis_rank1[mu][ind_rank1] = func;
                ind_rank1 += 1;
            } else {  //r>0, rank>1
                basis[mu][ind] = func;
                ind += 1;
            }
        }

    }
}


void ACECTildeBasisSet::_load_radial_ACERadial(FILE *fptr,
                                               const string filename,
                                               const string radbasename) {
    //initialize map_bond_specifications
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; mu_i++)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; mu_j++) {
            ACEBondSpecification bondSpecification;
            map_bond_specifications[make_pair(mu_i, mu_j)] = bondSpecification;
        }

    int res;
    char buffer[1024], buffer2[1024];

    res = fscanf(fptr, " nradbase=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "nradbase", "nradbase=[number]");
    nradbase = stoi_err(buffer, filename, "nradbase", "nradbase=[number]");
    //WARNING! hardcoded for single specie
    map_bond_specifications[make_pair(0, 0)].nradbasemax = nradbase;

    res = fscanf(fptr, " nradmax=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "nradmax", "nradmax=[number]");
    nradmax = stoi_err(buffer, filename, "nradmax", "nradmax=[number]");
    //WARNING! hardcoded for single specie
    map_bond_specifications[make_pair(0, 0)].nradmax = nradmax;

    res = fscanf(fptr, " cutoffmax=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "cutoffmax", "cutoffmax=[number]");
    cutoffmax = stod_err(buffer, filename, "cutoffmax", "cutoffmax=[number]");


    res = fscanf(fptr, " deltaSplineBins=");
    res = fscanf(fptr, "%s", buffer);
    if (res != 1)
        throw_error(filename, "deltaSplineBins", "deltaSplineBins=[spline density, Angstroms]");
    deltaSplineBins = stod_err(buffer, filename, "deltaSplineBins", "deltaSplineBins=[spline density, Angstroms]");

    //WARNING! hardcoded for single specie
    map_bond_specifications[make_pair(0, 0)].lmax = lmax;
    map_bond_specifications[make_pair(0, 0)].radbasename = radbasename;
    map_bond_specifications[make_pair(0, 0)].inner_cutoff_type = "density"; // default, for backward compatibility

    if (radial_functions == nullptr)
        radial_functions = new ACERadialFunctions(nradbase, lmax, nradmax,
                                                  deltaSplineBins,
                                                  nelements,
                                                  {{radbasename}});
    else
        radial_functions->init(nradbase, lmax, nradmax,
                               deltaSplineBins,
                               nelements,
                               {{radbasename}});


    //hard-core repulsion
    res = fscanf(fptr, " core repulsion parameters:");
    if (res != 0)
        throw_error(filename, "core repulsion parameters", "core repulsion parameters: [prehc lambdahc] ...");

    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            res = fscanf(fptr, "%s %s", buffer, buffer2);
            if (res != 2)
                throw_error(filename, "core repulsion parameters", "core repulsion parameters: [prehc lambdahc] ...");
            radial_functions->prehc(mu_i, mu_j) = stod_err(buffer, filename, "core repulsion parameters",
                                                           "core repulsion parameters: [prehc lambdahc] ...");
            radial_functions->lambdahc(mu_i, mu_j) = stod_err(buffer2, filename, "core repulsion parameters",
                                                              "core repulsion parameters: [prehc lambdahc] ...");

            map_bond_specifications[make_pair(mu_i, mu_j)].prehc = radial_functions->prehc(mu_i, mu_j);
            map_bond_specifications[make_pair(mu_i, mu_j)].lambdahc = radial_functions->lambdahc(mu_i, mu_j);
        }



    //read radial functions parameter
    res = fscanf(fptr, " radparameter=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            res = fscanf(fptr, "%s", buffer);
            if (res != 1)
                throw_error(filename, "radparameter", "radparameter=[param_ele1] [param_ele2]");
            radial_functions->lambda(mu_i, mu_j) = stod_err(buffer, filename, "radparameter",
                                                            "radparameter=[param_ele1] [param_ele2]");
            map_bond_specifications[make_pair(mu_i, mu_j)].radparameters = {radial_functions->lambda(mu_i, mu_j)};
        }


    res = fscanf(fptr, " cutoff=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            res = fscanf(fptr, "%s", buffer);
            if (res != 1)
                throw_error(filename, "cutoff", "cutoff=[param_ele1] [param_ele2]");
            radial_functions->cut(mu_i, mu_j) = stod_err(buffer, filename, "cutoff",
                                                         "cutoff=[param_ele1] [param_ele2]");

            map_bond_specifications[make_pair(mu_i, mu_j)].rcut = radial_functions->cut(mu_i, mu_j);
        }


    res = fscanf(fptr, " dcut=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            res = fscanf(fptr, " %s", buffer);
            if (res != 1)
                throw_error(filename, "dcut", "dcut=[param_ele1] [param_ele2]");
            radial_functions->dcut(mu_i, mu_j) = stod_err(buffer, filename, "dcut", "dcut=[param_ele1] [param_ele2]");
            map_bond_specifications[make_pair(mu_i, mu_j)].dcut = radial_functions->dcut(mu_i, mu_j);
        }


    res = fscanf(fptr, " crad=");
    for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i)
        for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
            Array3D<DOUBLE_TYPE> radcoefficients(nradmax, lmax + 1, nradbase);
            for (NS_TYPE k = 0; k < nradbase; k++)
                for (NS_TYPE n = 0; n < nradmax; n++)
                    for (LS_TYPE l = 0; l <= lmax; l++) {
                        res = fscanf(fptr, "%s", buffer);
                        if (res != 1)
                            throw_error(filename, "crad", "crad=[crad_]...[crad_knl]: nradbase*nrad*(l+1) times");
                        radial_functions->crad(mu_i, mu_j, n, l, k) = stod_err(buffer, filename, "crad",
                                                                               "crad=[crad_]...[crad_knl]: nradbase*nrad*(l+1) times");
                        radcoefficients(n, l, k) = radial_functions->crad(mu_i, mu_j, n, l, k);
                    }

            map_bond_specifications[make_pair(mu_i, mu_j)].radcoefficients = radcoefficients.to_vector();
        }
}

void ACECTildeBasisSet::_load_radial_SHIPsBasic(FILE *fptr,
                                                const string filename,
                                                const string radbasename) {
    // create a radial basis object, and read it from the file pointer
    SHIPsRadialFunctions *ships_radial_functions = new SHIPsRadialFunctions();

    ships_radial_functions->nelements = nelements;
    ships_radial_functions->radbasis.init(nelements, nelements, "SHIPsRadialFunctions::radbasis");
    ships_radial_functions->fread(fptr);

    _post_load_radial_SHIPsBasic(ships_radial_functions);
}

void ACECTildeBasisSet::_post_load_radial_SHIPsBasic(
        SHIPsRadialFunctions *ships_radial_functions) {//mimic ships_radial_functions to ACERadialFunctions
    ships_radial_functions->nradial = ships_radial_functions->get_maxn();
    ships_radial_functions->nradbase = ships_radial_functions->get_maxn();

    nradbase = ships_radial_functions->get_maxn();
    nradmax = ships_radial_functions->get_maxn();
    cutoffmax = ships_radial_functions->get_rcut();
    deltaSplineBins = 0.001;

    ships_radial_functions->nradbase = nradbase;
    ships_radial_functions->lmax = lmax;
    ships_radial_functions->nradial = nradmax;
    ships_radial_functions->gr.init(nradbase, "gr");
    ships_radial_functions->dgr.init(nradbase, "dgr");

    ships_radial_functions->fr.init(nradmax, lmax + 1, "fr");
    ships_radial_functions->dfr.init(nradmax, lmax + 1, "dfr");
    ships_radial_functions->crad.init(nelements, nelements, nradmax, (lmax + 1), nradbase, "crad");
    ships_radial_functions->crad.fill(0.);


    if (radial_functions) delete radial_functions;
    radial_functions = ships_radial_functions;
    radial_functions->prehc.fill(0);
    radial_functions->lambdahc.fill(1);
    radial_functions->lambda.fill(0);

    radial_functions->cut.init(nelements, nelements, "cut");
    radial_functions->dcut.init(nelements, nelements, "dcut");

    radial_functions->cut.fill(ships_radial_functions->get_rcut());
    radial_functions->dcut.fill(0);

    radial_functions->crad.fill(0);
}

vector<vector<SPECIES_TYPE>> ACECTildeBasisSet::get_all_coeffs_mask() const {
    //TODO: implement
    vector<vector<SPECIES_TYPE>> all_coeffs_mask;
    return all_coeffs_mask;
}

vector<DOUBLE_TYPE> ACECTildeBasisSet::get_all_coeffs() const {
    auto coeffs = radial_functions->crad.to_flatten_vector();

    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++) {
            auto ndens = basis_rank1[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis_rank1[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++)
                    coeffs.emplace_back(basis_rank1[mu][func_ind].ctildes[ms_ind * ndens + p]);
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; func_ind++) {
            auto ndens = basis[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++)
                    coeffs.emplace_back(basis[mu][func_ind].ctildes[ms_ind * ndens + p]);
            }
        }
    }

    return coeffs;
}

void ACECTildeBasisSet::set_all_coeffs(const vector<DOUBLE_TYPE> &coeffs) {
    size_t crad_size = radial_functions->crad.get_size();
    vector<DOUBLE_TYPE> crad_flatten_vector(coeffs.begin(), coeffs.begin() + crad_size);
    vector<DOUBLE_TYPE> basis_coeffs_vector(coeffs.begin() + crad_size, coeffs.end());

    radial_functions->crad = crad_flatten_vector;
    radial_functions->setuplookupRadspline();

    size_t coeffs_ind = 0;
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++) {
            auto ndens = basis_rank1[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis_rank1[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++, coeffs_ind++) {
                    basis_rank1[mu][func_ind].ctildes[ms_ind * ndens + p] = basis_coeffs_vector[coeffs_ind];
                }
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; func_ind++) {
            auto ndens = basis[mu][func_ind].ndensity;
            for (SHORT_INT_TYPE ms_ind = 0; ms_ind < basis[mu][func_ind].num_ms_combs; ms_ind++) {
                for (DENSITY_TYPE p = 0; p < ndens; p++, coeffs_ind++) {
                    basis[mu][func_ind].ctildes[ms_ind * ndens + p] = basis_coeffs_vector[coeffs_ind];
                }
            }
        }
    }
}

void ACECTildeBasisSet::save_yaml(const string &yaml_file_name) const {
    YAML_PACE::Node ctilde_basis_yaml;

    vector<string> elements_name_vec;
    elements_name_vec.assign(this->elements_name, this->elements_name + this->nelements);
    ctilde_basis_yaml["elements"] = elements_name_vec;
    ctilde_basis_yaml["elements"].SetStyle(YAML_PACE::EmitterStyle::Flow);

    ctilde_basis_yaml["E0"] = E0vals.to_vector();
    ctilde_basis_yaml["E0"].SetStyle(YAML_PACE::EmitterStyle::Flow);

    ctilde_basis_yaml["deltaSplineBins"] = this->deltaSplineBins;

    map<int, YAML_PACE::Node> yaml_map_embedding_specifications;
    for (const auto &p: this->map_embedding_specifications)
        yaml_map_embedding_specifications[p.first] = p.second.to_YAML();
    ctilde_basis_yaml["embeddings"] = yaml_map_embedding_specifications;

    map<vector<int>, YAML_PACE::Node> yaml_map_bond_specifications;
    for (const auto &p: this->map_bond_specifications) {
        vector<int> bond_pair = {(int) p.first.first, (int) p.first.second};
        YAML_PACE::Node bond_yaml;
        bond_yaml = bond_pair;
        bond_yaml.SetStyle(YAML_PACE::EmitterStyle::Flow);
        yaml_map_bond_specifications[bond_pair] = p.second.to_YAML();
    }
    ctilde_basis_yaml["bonds"] = yaml_map_bond_specifications;

    //iterate over keys and make them in Flow style
    for (YAML_PACE::detail::iterator_value p: ctilde_basis_yaml["bonds"]) {
        p.first.SetStyle(YAML_PACE::EmitterStyle::Flow);
    }

    map<int, vector<YAML_PACE::Node>> acebbasisfunc_map;
    for (SPECIES_TYPE mu = 0; mu < this->nelements; mu++) {
        vector<YAML_PACE::Node> acebbasisfunc_vec;
        for (size_t ind = 0; ind < this->total_basis_size_rank1[mu]; ind++)
            acebbasisfunc_vec.emplace_back(this->basis_rank1[mu][ind].to_YAML());

        for (size_t ind = 0; ind < this->total_basis_size[mu]; ind++)
            acebbasisfunc_vec.emplace_back(this->basis[mu][ind].to_YAML());

        acebbasisfunc_map[mu] = acebbasisfunc_vec;
    }

    ctilde_basis_yaml["functions"] = acebbasisfunc_map;

    YAML_PACE::Emitter yaml_emitter;
    yaml_emitter << ctilde_basis_yaml;

    std::ofstream fout(yaml_file_name);
    fout << yaml_emitter.c_str() << endl;
}

void ACECTildeBasisSet::load_yaml(const string &yaml_file_name) {
    //set the input file - first thing to do
    ifstream f(yaml_file_name.c_str());
    if (!f.good()) {
        stringstream s;
        s << "Potential file " << yaml_file_name << " doesn't exists";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }

    //load the file with yaml
    YAML_PACE::Node ctilde_basis_yaml = YAML_PACE::LoadFile(yaml_file_name);

    //reading elements and mapping
    auto elements_yaml = ctilde_basis_yaml["elements"];
    auto elements_name_vec = elements_yaml.as<vector<string>>();
    this->nelements = elements_name_vec.size();
    if (this->elements_name != nullptr)
        delete[] this->elements_name;
    this->elements_name = new string[nelements];
    for (int mu = 0; mu < nelements; mu++) {
        this->elements_name[mu] = elements_name_vec.at(mu);
        this->elements_to_index_map[elements_name_vec.at(mu)] = mu;
    }

    //reading E0vals
    auto e0_vec = ctilde_basis_yaml["E0"].as<vector<DOUBLE_TYPE >>();
    E0vals.init(nelements);
    E0vals.fill(0);
    E0vals = e0_vec;

    //reading embeddings
    auto yaml_map_embedding_specifications = ctilde_basis_yaml["embeddings"].as<map<int, YAML_PACE::Node>>();
    this->ndensitymax = 0;
    for (auto p: yaml_map_embedding_specifications) {
        SPECIES_TYPE mu_i = p.first;
        if (mu_i > nelements - 1)
            throw invalid_argument("yace::embeddings has species type key larger than nelements");

        auto &emb_yaml = p.second;
        ACEEmbeddingSpecification embeddingSpecification;

        embeddingSpecification.ndensity = emb_yaml["ndensity"].as<DENSITY_TYPE>();
        embeddingSpecification.FS_parameters = emb_yaml["FS_parameters"].as<vector<DOUBLE_TYPE>>();
        embeddingSpecification.npoti = emb_yaml["npoti"].as<string>();
        embeddingSpecification.rho_core_cutoff = emb_yaml["rho_core_cutoff"].as<DOUBLE_TYPE>();
        embeddingSpecification.drho_core_cutoff = emb_yaml["drho_core_cutoff"].as<DOUBLE_TYPE>();

        map_embedding_specifications[mu_i] = embeddingSpecification;

        if (embeddingSpecification.ndensity > this->ndensitymax)
            this->ndensitymax = embeddingSpecification.ndensity;
    }

    //reading bonds
    auto yaml_map_bond_specifications = ctilde_basis_yaml["bonds"].as<map<vector<int>, YAML_PACE::Node>>();
    this->lmax = 0;
    this->nradmax = 0;
    this->nradbase = 0;
    this->cutoffmax = 0;

    // check, if bonds::[]::radbasename=="ACE.jl.radbase"
    bool ACE_jl_radbase = false;
    bool PACE_radbase = false;
    for (const auto &p: yaml_map_bond_specifications) {
        auto bond_yaml = p.second;
        string radbasename = bond_yaml["radbasename"].as<string>();
        if (radbasename.rfind("ACE.jl", 0) == 0)
            ACE_jl_radbase = true;
        else
            PACE_radbase = true;
    }
    // check if both type of radbase -> inconsistency
    if (ACE_jl_radbase & PACE_radbase) {
        throw invalid_argument(
                "Only ACE.jl.* or PACE's radial basis are possible, but both types are used simultaneously.");
    }

    if (PACE_radbase) {
        vector<vector<string>> radbasename_ij(nelements, vector<string>(nelements));
        for (const auto &p: yaml_map_bond_specifications) {
            pair<SPECIES_TYPE, SPECIES_TYPE> bond_pair = make_pair(p.first[0], p.first[1]);
            if (bond_pair.first > nelements - 1 || bond_pair.second > nelements - 1)
                throw invalid_argument("yace::bonds has species type key larger than nelements");

            auto bond_yaml = p.second;
            ACEBondSpecification bondSpec;
            bondSpec.from_YAML(bond_yaml);

            map_bond_specifications[bond_pair] = bondSpec;

            radbasename_ij.at(bond_pair.first).at(bond_pair.second) = bondSpec.radbasename;

            //update lmax, nradbase max, ...
            if (bondSpec.nradmax > this->nradmax)
                this->nradmax = bondSpec.nradmax;

            if (bondSpec.lmax > this->lmax)
                this->lmax = bondSpec.lmax;

            if (bondSpec.nradbasemax > this->nradbase)
                this->nradbase = bondSpec.nradbasemax;

            if (bondSpec.rcut > this->cutoffmax)
                this->cutoffmax = bondSpec.rcut;

        }
        this->deltaSplineBins = ctilde_basis_yaml["deltaSplineBins"].as<DOUBLE_TYPE>();


        if (radial_functions == nullptr)
            radial_functions = new ACERadialFunctions(nradbase, lmax, nradmax,
                                                      deltaSplineBins,
                                                      nelements,
                                                      radbasename_ij);
        else
            radial_functions->init(nradbase, lmax, nradmax,
                                   deltaSplineBins,
                                   nelements,
                                   radbasename_ij);

        for (SPECIES_TYPE mu_i = 0; mu_i < nelements; ++mu_i) {
            for (SPECIES_TYPE mu_j = 0; mu_j < nelements; ++mu_j) {
                auto bond = make_pair(mu_i, mu_j);
                const auto &bondSpec = map_bond_specifications[bond];
                radial_functions->cut(mu_i, mu_j) = bondSpec.rcut;
                radial_functions->dcut(mu_i, mu_j) = bondSpec.dcut;
                radial_functions->prehc(mu_i, mu_j) = bondSpec.prehc;
                radial_functions->lambdahc(mu_i, mu_j) = bondSpec.lambdahc;
                radial_functions->lambda(mu_i, mu_j) = bondSpec.radparameters.at(0);

                radial_functions->cut_in(mu_i, mu_j) = bondSpec.rcut_in;
                radial_functions->dcut_in(mu_i, mu_j) = bondSpec.dcut_in;
                radial_functions->inner_cutoff_type = bondSpec.inner_cutoff_type;

                //setup crad
                for (NS_TYPE n = 0; n < bondSpec.nradmax; n++)
                    for (LS_TYPE l = 0; l <= bondSpec.lmax; l++)
                        for (NS_TYPE k = 0; k < bondSpec.nradbasemax; k++) {
                            radial_functions->crad(mu_i, mu_j, n, l, k) = bondSpec.radcoefficients.at(n).at(l).at(k);
                        }
            }
        }
        ///////////////////////////////////////////////////////////////////
    } else if (ACE_jl_radbase) {
        ///////////////////////////////////////////////////////////////////
        //read  lmax from YACE
        if (ctilde_basis_yaml["lmax"])
            this->lmax = ctilde_basis_yaml["lmax"].as<LS_TYPE>();
        else
            throw invalid_argument(
                    "For `ACE.jl.*` radbase functions, `lmax` should be provided in the YACE separately.");
        // no need to store map_bond_specifications, only SHIPsRadialFunctions
        SHIPsRadialFunctions *ships_radial_functions = new SHIPsRadialFunctions();
        ships_radial_functions->init(nelements);
        ships_radial_functions->read_yaml(ctilde_basis_yaml);
        _post_load_radial_SHIPsBasic(ships_radial_functions);
    }
    ///////////////////////////////////////////////////////////////////

    //setup spherical_harmonics and  radialBasis
    spherical_harmonics.init(lmax);
    radial_functions->setuplookupRadspline();

    //reading ACECTildeBasisFunctions
    //TODO:setup rankmax
    map<int, vector<YAML_PACE::Node>> acebbasisfunc_map = ctilde_basis_yaml["functions"].as<map<int, vector<YAML_PACE::Node> >>();

    vector<int> int_vec;
    vector<DOUBLE_TYPE> double_vec;

    total_basis_size_rank1 = new SHORT_INT_TYPE[nelements];
    basis_rank1 = new ACECTildeBasisFunction *[nelements];

    total_basis_size = new SHORT_INT_TYPE[nelements];
    basis = new ACECTildeBasisFunction *[nelements];

    this->rankmax = 0;

    for (const auto &p: acebbasisfunc_map) {
        SPECIES_TYPE mu = p.first;
        if (mu > nelements - 1)
            throw invalid_argument("yace::functions has species type key larger than nelements");

        total_basis_size_rank1[mu] = 0;
        total_basis_size[mu] = 0;

        auto ctildefunc_vec_yaml = p.second;

        vector<ACECTildeBasisFunction> ctildefunc_vec;//TODO: read write is_half_ms_basis
        for (const auto &ctildefunc_yaml: ctildefunc_vec_yaml) {

            ACECTildeBasisFunction ctildefunc;

            ctildefunc.mu0 = ctildefunc_yaml["mu0"].as<SHORT_INT_TYPE>();
            ctildefunc.rank = ctildefunc_yaml["rank"].as<SHORT_INT_TYPE>();
            ctildefunc.ndensity = ctildefunc_yaml["ndensity"].as<SHORT_INT_TYPE>();
            ctildefunc.num_ms_combs = ctildefunc_yaml["num_ms_combs"].as<SHORT_INT_TYPE>();

            int_vec = ctildefunc_yaml["mus"].as<vector<int>>();
            if (int_vec.size()!=ctildefunc.rank)
                throw invalid_argument("mus:: not sufficient number of values");
            ctildefunc.mus = new SPECIES_TYPE[ctildefunc.rank];
            for (int r = 0; r < ctildefunc.rank; r++)
                ctildefunc.mus[r] = int_vec.at(r);

            int_vec = ctildefunc_yaml["ns"].as<vector<int>>();
            if (int_vec.size()!=ctildefunc.rank)
                throw invalid_argument("ns:: not sufficient number of values");
            ctildefunc.ns = new NS_TYPE[ctildefunc.rank];
            for (int r = 0; r < ctildefunc.rank; r++)
                ctildefunc.ns[r] = int_vec.at(r);


            int_vec = ctildefunc_yaml["ls"].as<vector<int>>();
            if (int_vec.size()!=ctildefunc.rank)
                throw invalid_argument("ls:: not sufficient number of values");
            ctildefunc.ls = new LS_TYPE[ctildefunc.rank];
            for (int r = 0; r < ctildefunc.rank; r++)
                ctildefunc.ls[r] = int_vec.at(r);

            //this->ms_combs; //[num_ms_combs * rank]
            int_vec = ctildefunc_yaml["ms_combs"].as<vector<int>>();
            if (int_vec.size()!=ctildefunc.rank * ctildefunc.num_ms_combs)
                throw invalid_argument("ms_combs:: not sufficient number of values");
            ctildefunc.ms_combs = new MS_TYPE[ctildefunc.rank * ctildefunc.num_ms_combs];
            for (int r = 0; r < ctildefunc.rank * ctildefunc.num_ms_combs; r++)
                ctildefunc.ms_combs[r] = int_vec.at(r);


            // this->ctildes; //[num_of_ms_combs * ndensity]
            double_vec = ctildefunc_yaml["ctildes"].as<vector<DOUBLE_TYPE >>();
            if (double_vec.size()!=ctildefunc.ndensity * ctildefunc.num_ms_combs)
                throw invalid_argument("ctildes:: not sufficient number of values");
            ctildefunc.ctildes = new DOUBLE_TYPE[ctildefunc.ndensity * ctildefunc.num_ms_combs];
            for (int r = 0; r < ctildefunc.ndensity * ctildefunc.num_ms_combs; r++)
                ctildefunc.ctildes[r] = double_vec.at(r);

            ctildefunc_vec.emplace_back(ctildefunc);

            if (ctildefunc.rank == 1)
                total_basis_size_rank1[mu]++;
            else
                total_basis_size[mu]++;

        } // end for over ctildefunc_vec_yaml

//        cout << "total_basis_size_rank1[mu]=" << total_basis_size_rank1[mu] << endl;
//        cout << "total_basis_size[mu]=" << total_basis_size[mu] << endl;

        basis_rank1[mu] = new ACECTildeBasisFunction[total_basis_size_rank1[mu]];
        basis[mu] = new ACECTildeBasisFunction[total_basis_size[mu]];
        size_t func_ind_rank1 = 0, func_ind = 0;
        for (const ACECTildeBasisFunction ctildefunc: ctildefunc_vec) {
            if (ctildefunc.rank == 1) {
                basis_rank1[mu][func_ind_rank1] = ctildefunc;
                func_ind_rank1++;
            } else {
                basis[mu][func_ind] = ctildefunc;
                func_ind++;
            }
            //aggregate rankmax
            if (this->rankmax < ctildefunc.rank)
                this->rankmax = ctildefunc.rank;
        }
    }


    pack_flatten_basis();
}


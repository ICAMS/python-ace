//
// Created by Yury Lysogorskiy on 16.03.2020.
//
#include "ace_b_basis.h"

#include <algorithm>
#include <sstream>


#include "ace_yaml_input.h"
#include "ace_couplings.h"
#include "ace_c_basis.h"
#include "ace_utils.h"

void group_basis_functions_by_index(const vector<ACEBBasisFunction> &basis,
                                    Basis_functions_map &basis_functions_map) {
    for (const auto &cur_basfunc : basis) {
        auto *current_basis_function = const_cast<ACEBBasisFunction *>(&cur_basfunc);
        SPECIES_TYPE X0 = current_basis_function->mu0;
        RANK_TYPE r = cur_basfunc.rank;
        Vector_ns vector_ns(current_basis_function->ns, current_basis_function->ns + r - 1 + 1);
        Vector_ls vector_ls(current_basis_function->ls, current_basis_function->ls + r - 1 + 1);
        Vector_Xs vector_Xs(current_basis_function->mus, current_basis_function->mus + r - 1 + 1);
        Basis_index_key key(X0, vector_Xs, vector_ns, vector_ls);
        auto search = basis_functions_map.find(key);
        if (search == basis_functions_map.end()) { // not in dict
            basis_functions_map[key] = Basis_function_ptr_list();
        }
        basis_functions_map[key].push_back(current_basis_function);
    }
}


void summation_over_LS(Basis_functions_map &basis_functions_map,
                       vector<ACECTildeBasisFunction> &ctilde_basis) {

#ifdef DEBUG_C_TILDE
    cout << "rankmax=" << (int) r << "\t";
    cout << "number of basis functions (len dict)= " << basis_functions_map.size() << endl;
#endif
    // loop over dictionary of grouped basis functions
    ctilde_basis.resize(basis_functions_map.size());
    int new_b_tilde_index = 0;
    for (auto it = basis_functions_map.begin(); basis_functions_map.end() != it; ++it, ++new_b_tilde_index) {

        ACECTildeBasisFunction &new_func = ctilde_basis[new_b_tilde_index];
        const SPECIES_TYPE &X0 = get<0>(it->first);
        const Vector_Xs &XS = get<1>(it->first);
        const Vector_ns &ns = get<2>(it->first);
        const Vector_ls &ls = get<3>(it->first);


        Basis_function_ptr_list b_basis_list = it->second;
        ACEBBasisFunction *first_basis_function = b_basis_list.front();

        //RANK_TYPE cur_rank = first_basis_function->rank;
        RANK_TYPE cur_rank = ns.size();
        new_func.rank = cur_rank;
        new_func.ndensity = first_basis_function->ndensity;

        new_func.mu0 = X0;
        delete[] new_func.mus;
        new_func.mus = new SPECIES_TYPE[cur_rank];

        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.mus[i] = XS[i];

        delete[] new_func.ns;
        new_func.ns = new NS_TYPE[cur_rank];
        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.ns[i] = ns[i];

        delete[] new_func.ls;
        new_func.ls = new LS_TYPE[cur_rank];
        for (RANK_TYPE i = 0; i < cur_rank; ++i) new_func.ls[i] = ls[i];


        //TODO: join the ms combinations, not only take the first one
        map<Vector_ms, vector<DOUBLE_TYPE>> ms_combinations_coefficients_map;
        for (const auto &bas_func: b_basis_list) {
            for (int ms_ind = 0; ms_ind < bas_func->num_ms_combs; ++ms_ind) {
                Vector_ms ms_vec(cur_rank);
                for (RANK_TYPE rr = 0; rr < cur_rank; ++rr)
                    ms_vec[rr] = bas_func->ms_combs[ms_ind * cur_rank + rr];
                if (ms_combinations_coefficients_map.find(ms_vec) == ms_combinations_coefficients_map.end())
                    ms_combinations_coefficients_map[ms_vec].resize(new_func.ndensity);
                //sum-up vector
                for (DENSITY_TYPE p = 0; p < new_func.ndensity; ++p)
                    ms_combinations_coefficients_map[ms_vec][p] += bas_func->coeff[p] * bas_func->gen_cgs[ms_ind];
            }
        }

        new_func.num_ms_combs = ms_combinations_coefficients_map.size();

        delete[] new_func.ms_combs;
        new_func.ms_combs = new MS_TYPE[cur_rank * new_func.num_ms_combs];
        delete[] new_func.ctildes;
        new_func.ctildes = new DOUBLE_TYPE[new_func.num_ms_combs * new_func.ndensity];

        int ms_ind = 0;
        for (const auto &ms_coeff_pair: ms_combinations_coefficients_map) {
            Vector_ms ms_vec = ms_coeff_pair.first;
            vector<DOUBLE_TYPE> coeff = ms_coeff_pair.second;

            //copy ms combination
            for (RANK_TYPE rr = 0; rr < cur_rank; rr++)
                new_func.ms_combs[ms_ind * cur_rank + rr] = ms_vec[rr];
            //copy corresponding c_tilde coefficient
            for (DENSITY_TYPE p = 0; p < new_func.ndensity; ++p)
                new_func.ctildes[ms_ind * new_func.ndensity + p] = coeff[p];

            SHORT_INT_TYPE sign = 0;
            for (RANK_TYPE t = 0; t < cur_rank; ++t)
                if (ms_vec[t] < 0) {
                    sign = -1;
                    break;
                } else if (ms_vec[t] > 0) {
                    sign = +1;
                    break;
                }


            ms_ind++;
        }

        new_func.is_half_ms_basis = first_basis_function->is_half_ms_basis;

#ifdef DEBUG_C_TILDE
        cout << "new_func=" << endl;
        print_C_tilde_B_basis_function(ctilde_basis[r - 1][new_b_tilde_index]);
#endif
    }
}

template<typename T>
T max_of(vector<T> vec) {
    return *max_element(vec.begin(), vec.end());
}

LS_TYPE get_lmax(vector<LS_TYPE> ls, vector<LS_TYPE> LS) {
    LS_TYPE func_lmax = max_of(ls);
    LS_TYPE func_Lmax = 0;
    if (LS.size() > 0)
        func_Lmax = max_of(LS);
    return (func_lmax > func_Lmax ? func_lmax : func_Lmax);
}


//constructor from BBasisConfiguration
ACEBBasisSet::ACEBBasisSet(BBasisConfiguration &bBasisConfiguration) {
    initialize_basis(bBasisConfiguration);
}

//constructor by loading from YAML file
ACEBBasisSet::ACEBBasisSet(string yaml_file_name) {
    ACEBBasisSet::load(yaml_file_name);
}

//copy constructor
ACEBBasisSet::ACEBBasisSet(const ACEBBasisSet &other) {
    ACEBBasisSet::_copy_scalar_memory(other);
    ACEBBasisSet::_copy_dynamic_memory(other);
    ACEBBasisSet::pack_flatten_basis();
}

//operator=
ACEBBasisSet &ACEBBasisSet::operator=(const ACEBBasisSet &other) {
    if (this != &other) {
        ACEBBasisSet::_clean();
        ACEBBasisSet::_copy_scalar_memory(other);
        ACEBBasisSet::_copy_dynamic_memory(other);
        ACEBBasisSet::pack_flatten_basis();
    }
    return *this;
}

ACEBBasisSet::~ACEBBasisSet() {
    ACEBBasisSet::_clean();
}


//pack into 1D array with all basis functions
void ACEBBasisSet::flatten_basis() {
    _clean_basis_arrays();

    if (total_basis_size_rank1 != nullptr) delete[] total_basis_size_rank1;
    if (total_basis_size != nullptr) delete[] total_basis_size;

    total_basis_size_rank1 = new SHORT_INT_TYPE[nelements];
    total_basis_size = new SHORT_INT_TYPE[nelements];


    basis_rank1 = new ACEBBasisFunction *[nelements];
    basis = new ACEBBasisFunction *[nelements];

    size_t tot_size_rank1 = 0;
    size_t tot_size = 0;

    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        tot_size = 0;
        tot_size_rank1 = 0;

        for (auto &func: this->mu0_bbasis_vector[mu]) {
            if (func.rank == 1) tot_size_rank1 += 1;
            else tot_size += 1;
        }


        total_basis_size_rank1[mu] = tot_size_rank1;
        basis_rank1[mu] = new ACEBBasisFunction[tot_size_rank1];

        total_basis_size[mu] = tot_size;
        basis[mu] = new ACEBBasisFunction[tot_size];
    }


    for (SPECIES_TYPE mu = 0; mu < this->nelements; ++mu) {
        size_t ind_rank1 = 0;
        size_t ind = 0;

        for (auto &func: this->mu0_bbasis_vector[mu]) {
            if (func.rank == 1) { //r=0, rank=1
                basis_rank1[mu][ind_rank1] = func;
                ind_rank1 += 1;
            } else { //r>0, rank>1
                basis[mu][ind] = func;
                ind += 1;
            }
        }

    }
}

void ACEBBasisSet::_clean() {
    // call parent method
    ACEFlattenBasisSet::_clean();
    _clean_contiguous_arrays();
    _clean_basis_arrays();
}

void ACEBBasisSet::_clean_contiguous_arrays() {
    if (full_gencg_rank1 != nullptr) delete[] full_gencg_rank1;
    full_gencg_rank1 = nullptr;

    if (full_gencg != nullptr) delete[] full_gencg;
    full_gencg = nullptr;

    if (full_coeff_rank1 != nullptr) delete[] full_coeff_rank1;
    full_coeff_rank1 = nullptr;

    if (full_coeff != nullptr) delete[] full_coeff;
    full_coeff = nullptr;

    if (full_LS != nullptr) delete[] full_LS;
    full_LS = nullptr;
}

void ACEBBasisSet::_clean_basis_arrays() {
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

void ACEBBasisSet::_copy_scalar_memory(const ACEBBasisSet &src) {
    ACEFlattenBasisSet::_copy_scalar_memory(src);
    mu0_bbasis_vector = src.mu0_bbasis_vector;

    total_num_of_ms_comb_rank1 = src.total_num_of_ms_comb_rank1;
    total_num_of_ms_comb = src.total_num_of_ms_comb;

    total_LS_size = src.total_LS_size;
}

void ACEBBasisSet::_copy_dynamic_memory(const ACEBBasisSet &src) {//allocate new memory
    ACEFlattenBasisSet::_copy_dynamic_memory(src);

    if (src.basis_rank1 == nullptr)
        throw runtime_error("Could not copy ACEBBasisSet::basis_rank1 - array not initialized");
    if (src.basis == nullptr)
        throw runtime_error("Could not copy ACEBBasisSet::basis - array not initialized");

    basis_rank1 = new ACEBBasisFunction *[nelements];
    basis = new ACEBBasisFunction *[nelements];

    //copy basis arrays
    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        basis_rank1[mu] = new ACEBBasisFunction[total_basis_size_rank1[mu]];

        for (size_t i = 0; i < total_basis_size_rank1[mu]; i++) {
            this->basis_rank1[mu][i] = src.basis_rank1[mu][i];
        }

        basis[mu] = new ACEBBasisFunction[total_basis_size[mu]];
        for (size_t i = 0; i < total_basis_size[mu]; i++) {
            basis[mu][i] = src.basis[mu][i];
        }
    }

    //DON"T COPY CONTIGUOUS ARRAY, REBUILD THEM
}

void ACEBBasisSet::pack_flatten_basis() {
    compute_array_sizes(basis_rank1, basis);

    //2. allocate contiguous arrays
    full_ns_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    full_ls_rank1 = new NS_TYPE[rank_array_total_size_rank1];
    full_mus_rank1 = new SPECIES_TYPE[rank_array_total_size_rank1];
    full_ms_rank1 = new MS_TYPE[rank_array_total_size_rank1];

    full_gencg_rank1 = new DOUBLE_TYPE[total_num_of_ms_comb_rank1];
    full_coeff_rank1 = new DOUBLE_TYPE[coeff_array_total_size_rank1];


    full_ns = new NS_TYPE[rank_array_total_size];
    full_ls = new LS_TYPE[rank_array_total_size];
    full_LS = new LS_TYPE[total_LS_size];

    full_mus = new SPECIES_TYPE[rank_array_total_size];
    full_ms = new MS_TYPE[ms_array_total_size];

    full_gencg = new DOUBLE_TYPE[total_num_of_ms_comb];
    full_coeff = new DOUBLE_TYPE[coeff_array_total_size];

    //3. copy the values from private C_tilde_B_basis_function arrays to new contigous space
    //4. clean private memory
    //5. reassign private array pointers

    //r = 0, rank = 1
    size_t rank_array_ind_rank1 = 0;
    size_t coeff_array_ind_rank1 = 0;
    size_t ms_array_ind_rank1 = 0;
    size_t gen_cg_ind_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind_r1 = 0; func_ind_r1 < total_basis_size_rank1[mu]; ++func_ind_r1) {
            auto &func = basis_rank1[mu][func_ind_r1];
            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            full_ns_rank1[rank_array_ind_rank1] = func.ns[0];

            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            full_ls_rank1[rank_array_ind_rank1] = func.ls[0];

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            full_mus_rank1[rank_array_ind_rank1] = func.mus[0];

            //copy values full_coeff_rank1 from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_coeff_rank1[coeff_array_ind_rank1], func.coeff,
                   func.ndensity * sizeof(DOUBLE_TYPE));

            memcpy(&full_gencg_rank1[gen_cg_ind_rank1], func.gen_cgs,
                   func.num_ms_combs * sizeof(DOUBLE_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms_rank1[ms_array_ind_rank1], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.ns = &full_ns_rank1[rank_array_ind_rank1];
            func.ls = &full_ls_rank1[rank_array_ind_rank1];
            func.mus = &full_mus_rank1[rank_array_ind_rank1];
            func.coeff = &full_coeff_rank1[coeff_array_ind_rank1];
            func.gen_cgs = &full_gencg_rank1[gen_cg_ind_rank1];
            func.ms_combs = &full_ms_rank1[ms_array_ind_rank1];
            func.is_proxy = true;

            rank_array_ind_rank1 += func.rank;
            ms_array_ind_rank1 += func.rank *
                                  func.num_ms_combs;

            coeff_array_ind_rank1 += func.ndensity;
            gen_cg_ind_rank1 += func.num_ms_combs;
        }
    }


    //rank>1, r>0
    size_t rank_array_ind = 0;
    size_t coeff_array_ind = 0;
    size_t ms_array_ind = 0;
    size_t gen_cg_ind = 0;
    size_t LS_array_ind = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            ACEBBasisFunction &func = basis[mu][func_ind];

            //copy values ns from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ns[rank_array_ind], func.ns,
                   func.rank * sizeof(NS_TYPE));
            //copy values ls from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ls[rank_array_ind], func.ls,
                   func.rank * sizeof(LS_TYPE));
            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_mus[rank_array_ind], func.mus,
                   func.rank * sizeof(SPECIES_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_LS[LS_array_ind], func.LS,
                   func.rankL * sizeof(LS_TYPE));


            //copy values full_coeff_rank1 from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_coeff[coeff_array_ind], func.coeff,
                   func.ndensity * sizeof(DOUBLE_TYPE));

            memcpy(&full_gencg[gen_cg_ind], func.gen_cgs,
                   func.num_ms_combs * sizeof(DOUBLE_TYPE));

            //copy values mus from c_tilde_basis_function private memory to contigous memory part
            memcpy(&full_ms[ms_array_ind], func.ms_combs,
                   func.num_ms_combs *
                   func.rank * sizeof(MS_TYPE));

            //release memory of each ACECTildeBasisFunction if it is not proxy
            func._clean();

            func.ns = &full_ns[rank_array_ind];
            func.ls = &full_ls[rank_array_ind];
            func.mus = &full_mus[rank_array_ind];
            func.LS = &full_LS[LS_array_ind];

            func.coeff = &full_coeff[coeff_array_ind];
            func.gen_cgs = &full_gencg[gen_cg_ind];

            func.ms_combs = &full_ms[ms_array_ind];
            func.is_proxy = true;

            rank_array_ind += func.rank;
            ms_array_ind += func.rank *
                            func.num_ms_combs;
            coeff_array_ind += func.ndensity;
            gen_cg_ind += func.num_ms_combs;
            LS_array_ind += func.rankL;

        }
    }

}


vector<string> split_key(string mainkey) {

    vector<string> splitted;
    istringstream stream(mainkey);

    for (string mainkey; stream >> mainkey;)
        splitted.emplace_back(mainkey);

    return splitted;
}

void ACEBBasisSet::load(string filename) {
    BBasisConfiguration basisConfiguration;
    basisConfiguration.load(filename);
    initialize_basis(basisConfiguration);
}

void order_and_compress_b_basis_function(ACEBBasisFunction &func) {
    vector<tuple<SPECIES_TYPE, NS_TYPE, LS_TYPE, MS_TYPE, int> > v;

    vector<SPECIES_TYPE> new_XS(func.rank);
    vector<NS_TYPE> new_NS(func.rank);
    vector<LS_TYPE> new_LS(func.rank);
    vector<int> sort_order(func.rank);

    map<vector<MS_TYPE>, DOUBLE_TYPE> ms_map;
    int new_ms_ind = 0;
    vector<MS_TYPE> new_ms;
    DOUBLE_TYPE new_gen_cg;

    for (SHORT_INT_TYPE ms_comb_ind = 0; ms_comb_ind < func.num_ms_combs; ms_comb_ind++) {

        v.clear();
        for (RANK_TYPE r = 0; r < func.rank; r++) {
            v.emplace_back(
                    make_tuple(func.mus[r], func.ns[r], func.ls[r], func.ms_combs[ms_comb_ind * func.rank + r], r));
        }

        sort(v.begin(), v.end());

        //check if (tup(0..2) always the same
        if (ms_comb_ind == 0) {
            for (RANK_TYPE r = 0; r < func.rank; r++) {
                new_XS[r] = get<0>(v[r]);
                new_NS[r] = get<1>(v[r]);
                new_LS[r] = get<2>(v[r]);
                sort_order[r] = get<4>(v[r]);
            }
        }

        for (RANK_TYPE r = 0; r < func.rank; r++) {
            if (new_XS[r] != get<0>(v[r]) ||
                new_NS[r] != get<1>(v[r]) ||
                new_LS[r] != get<2>(v[r])) {
                stringstream s;
                s << "INCONSISTENT SORTED BLOCK!\n";
                s << "->>sorted XS-ns-ls-ms combinations: {\n";
                char buf[1024];
                for (const auto &tup: v) {
                    sprintf(buf, "(%d, %d, %d, %d)\n", get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
                    s << buf;
                }
                s << "}";
                throw logic_error(s.str());
            }
        }

        vector<MS_TYPE> new_ms(func.rank);
        for (RANK_TYPE r = 0; r < func.rank; r++)
            new_ms[r] = get<3>(v[r]);

        auto search = ms_map.find(new_ms);
        if (search == ms_map.end()) { // not in dict
            ms_map[new_ms] = 0;
        }

        ms_map[new_ms] += func.gen_cgs[ms_comb_ind];
    }
    //  drop-out the k,v pairs from ms_map when value is zero
    for (auto it = ms_map.begin(); it != ms_map.end();) {
        auto key_ms = it->first;
        auto val_gen_cg = it->second;
        if (abs(val_gen_cg) < 1e-15) {
            ms_map.erase(it++);
        } else {
            ++it;
        }
    }


    int gain = func.num_ms_combs - ms_map.size();

    if (gain > 0) {
        for (RANK_TYPE r = 0; r < func.rank; r++) {
            func.mus[r] = new_XS[r];
            func.ns[r] = new_NS[r];
            func.ls[r] = new_LS[r];
        }
        func.sort_order = sort_order;
        SHORT_INT_TYPE new_num_of_ms_combinations = ms_map.size();


        delete[] func.gen_cgs;
        delete[] func.ms_combs;

        func.gen_cgs = new DOUBLE_TYPE[new_num_of_ms_combinations];
        func.ms_combs = new MS_TYPE[new_num_of_ms_combinations * func.rank];


        for (auto it = ms_map.begin(); it != ms_map.end(); ++it, ++new_ms_ind) {
            new_ms = it->first;
            new_gen_cg = it->second;

            for (RANK_TYPE r = 0; r < func.rank; r++)
                func.ms_combs[new_ms_ind * func.rank + r] = new_ms[r];


            func.gen_cgs[new_ms_ind] = new_gen_cg;
        }

        func.num_ms_combs = new_num_of_ms_combinations;

    }

}

// compress each basis function by considering A*A*..*A symmetry wrt. permutations
void ACEBBasisSet::compress_basis_functions() {
    SHORT_INT_TYPE tot_ms_combs = 0, num_ms_combs = 0;
    SHORT_INT_TYPE tot_new_ms_combs = 0, new_ms_combs = 0;
    for (SPECIES_TYPE elei = 0; elei < this->nelements; ++elei) {

        num_ms_combs = 0;
        new_ms_combs = 0;

        //order bbasis functions
        if (is_sort_functions)
            std::sort(this->mu0_bbasis_vector[elei].begin(), this->mu0_bbasis_vector[elei].end(), compare);

        vector<ACEBBasisFunction> &sub_basis = this->mu0_bbasis_vector[elei];

        for (ACEBBasisFunction &func: sub_basis) {
            tot_ms_combs += func.num_ms_combs;
            num_ms_combs += func.num_ms_combs;

            order_and_compress_b_basis_function(func);
            tot_new_ms_combs += func.num_ms_combs;
            new_ms_combs += func.num_ms_combs;
        }
        if (new_ms_combs < num_ms_combs) {
//            printf("element: %d - basis compression from %d to %d by %d ms-combinations (%.2f%%) \n",
//                   (int) elei, num_ms_combs, new_ms_combs,
//                   num_ms_combs - new_ms_combs, 1. * (num_ms_combs - new_ms_combs) / num_ms_combs * 100.);
        }

    }

    if (new_ms_combs < num_ms_combs) {
//        printf("Total basis compression from %d to %d by %d ms-combinations\n",
//               tot_ms_combs, tot_new_ms_combs,
//               tot_ms_combs - tot_new_ms_combs);
    }
}


void ACEBBasisSet::save(const string &filename) {
    BBasisConfiguration config = this->to_BBasisConfiguration();
    config.save(filename);
}

vector<SPECIES_TYPE> get_unique_species(const ACEBBasisFunction &func) {
    vector<SPECIES_TYPE> species;
    species.emplace_back(func.mu0);
    SPECIES_TYPE mu;
    for (int i = 0; i < func.rank; i++) {
        mu = func.mus[i];
        if (std::find(species.begin(), species.end(), mu) == species.end())
            species.emplace_back(mu);
    }
    return species;
}


BBasisConfiguration ACEBBasisSet::to_BBasisConfiguration() const {
    BBasisConfiguration config;

    config.metadata = this->metadata;
    config.auxdata = this->auxdata;

    config.deltaSplineBins = radial_functions->deltaSplineBins;
    vector<BBasisFunctionsSpecificationBlock> blocks;
    vector<string> elements_mapping(nelements);

    for (SPECIES_TYPE s = 0; s < nelements; s++)
        elements_mapping[s] = elements_name[s];

    //run over all func_specs and collect them into dictionary by unique elements
    map<vector<SPECIES_TYPE>, vector<BBasisFunctionSpecification>> map_elements_vec_basisfuncspec;
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        auto basis_r1 = basis_rank1[mu];
        auto n_basis_r1 = total_basis_size_rank1[mu];
        for (int func_ind = 0; func_ind < n_basis_r1; func_ind++) {
            ACEBBasisFunction func = basis_r1[func_ind];
            const auto uniq_species = get_unique_species(func);

            BBasisFunctionSpecification spec(elements_mapping, func);
            map_elements_vec_basisfuncspec[uniq_species].emplace_back(spec);
        }

        auto basis_high_r = basis[mu];
        auto n_basis_high_r = total_basis_size[mu];
        for (int func_ind = 0; func_ind < n_basis_high_r; func_ind++) {
            ACEBBasisFunction func = basis_high_r[func_ind];
            const auto uniq_species = get_unique_species(func);
            BBasisFunctionSpecification spec(elements_mapping, func);
            map_elements_vec_basisfuncspec[uniq_species].emplace_back(spec);
        }
    }

    //loop over collected dictionary map_elements_vec_basisfuncspec
    // and assign block's parameters depending on the number of species:
    for (auto const &pair: map_elements_vec_basisfuncspec) {
        const auto unique_species = pair.first;
        const auto vec_bfuncspecs = pair.second;

        int number_species = unique_species.size();

        //create new block
        BBasisFunctionsSpecificationBlock block;
        block.number_of_species = number_species;

        block.elements_vec = {};
        for (SPECIES_TYPE mu: unique_species)
            block.elements_vec.emplace_back(elements_name[mu]);

        block.block_name = join(block.elements_vec, " ");

        //push one more elemtn to make pair : Al -> Al Al
        if (number_species == 1) {
            block.elements_vec.emplace_back(block.elements_vec[0]);
        }


        SPECIES_TYPE mu0 = unique_species.at(0);


        block.mu0 = elements_name[mu0];

        // 1-specie parameters:
        // - embedding function
        // - rhocut, drhocut
        if (number_species == 1) {
            //make single atom interaction block
            const auto &embeddingSpecification = this->map_embedding_specifications.at(mu0);

            block.ndensityi = embeddingSpecification.ndensity;
            block.npoti = embeddingSpecification.npoti;
            block.fs_parameters = embeddingSpecification.FS_parameters;
            block.rho_cut = embeddingSpecification.rho_core_cutoff;
            block.drho_cut = embeddingSpecification.drho_core_cutoff;
        }

        // 2-specie parameters - pair interaction:
        // - cutoff, dcutoff
        // - nradbase, nradmax, lmax, type of rad-base
        // - crad[n][l][k]
        // - core repulsion parameters
        if (number_species <= 2) {
            SPECIES_TYPE mu1 = (number_species == 1) ? unique_species.at(0) : unique_species.at(1);
            const auto &bondSpecification = map_bond_specifications.at(make_pair(mu0, mu1));

            block.rcutij = bondSpecification.rcut;
            block.dcutij = bondSpecification.dcut;
            block.NameOfCutoffFunctionij = "cos";
            block.radbase = bondSpecification.radbasename;

            block.nradmaxi = bondSpecification.nradmax;
            block.lmaxi = bondSpecification.lmax;
            block.nradbaseij = bondSpecification.nradbasemax;
            block.radparameters = bondSpecification.radparameters;
            block.radcoefficients = bondSpecification.radcoefficients;
            block.core_rep_parameters = {bondSpecification.prehc, bondSpecification.lambdahc};

            block.r_in = bondSpecification.rcut_in;
            block.delta_in = bondSpecification.dcut_in;
            block.inner_cutoff_type = bondSpecification.inner_cutoff_type;
        }

        //for 3 and more species interaction - no extra information

        block.funcspecs = vec_bfuncspecs;

        // update rankmax
        block.rankmax = 0;
        for (const auto &spec: block.funcspecs) {
            if (spec.rank > block.rankmax)
                block.rankmax = spec.rank;
        }

        blocks.emplace_back(block);
    }

    config.funcspecs_blocks = blocks;

    return config;
}


void ACEBBasisSet::compute_array_sizes(ACEBBasisFunction **basis_rank1, ACEBBasisFunction **basis) {
    //compute arrays sizes
    rank_array_total_size_rank1 = 0;
    //ms_array_total_size_rank1 = rank_array_total_size_rank1;
    coeff_array_total_size_rank1 = 0;

    total_num_of_ms_comb_rank1 = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        if (total_basis_size_rank1[mu] > 0) {
            rank_array_total_size_rank1 += total_basis_size_rank1[mu];
            //only one ms-comb per rank-1 basis func
            total_num_of_ms_comb_rank1 += total_basis_size_rank1[mu]; // compute size for full_gencg_rank1
            ACEAbstractBasisFunction &func = basis_rank1[mu][0];
            coeff_array_total_size_rank1 += total_basis_size_rank1[mu] * func.ndensity;// *size of full_coeff_rank1
        }
    }

    //rank>1
    rank_array_total_size = 0;
    coeff_array_total_size = 0;

    ms_array_total_size = 0;
    max_dB_array_size = 0;

    total_num_of_ms_comb = 0;

    max_B_array_size = 0;

    total_LS_size = 0;

    size_t cur_ms_size = 0;
    size_t cur_ms_rank_size = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        cur_ms_size = 0;
        cur_ms_rank_size = 0;
        if (total_basis_size[mu] == 0) continue;
        ACEAbstractBasisFunction &func = basis[mu][0];
        coeff_array_total_size += total_basis_size[mu] * func.ndensity; // size of full_coeff
        for (int func_ind = 0; func_ind < total_basis_size[mu]; ++func_ind) {
            auto &func = basis[mu][func_ind];
            rank_array_total_size += func.rank;
            ms_array_total_size += func.rank * func.num_ms_combs;
            total_num_of_ms_comb += func.num_ms_combs; // compute size for full_gencg
            cur_ms_size += func.num_ms_combs;
            cur_ms_rank_size += func.rank * func.num_ms_combs;
            total_LS_size += func.rankL;
        }

        if (cur_ms_size > max_B_array_size)
            max_B_array_size = cur_ms_size;

        if (cur_ms_rank_size > max_dB_array_size)
            max_dB_array_size = cur_ms_rank_size;
    }
}


ACECTildeBasisSet ACEBBasisSet::to_ACECTildeBasisSet() const {
    C_tilde_full_basis_vector2d mu0_ctilde_basis_vector(nelements);
    SHORT_INT_TYPE num_ctilde_max = 0;
    for (SPECIES_TYPE mu0 = 0; mu0 < this->nelements; mu0++) {
        auto const &b_basis_vector = this->mu0_bbasis_vector.at(mu0);
        auto &ctilde_basis_vectors = mu0_ctilde_basis_vector[mu0];
        convert_B_to_Ctilde_basis_functions(b_basis_vector, ctilde_basis_vectors);

        //sort ctilde basis vectors
        if (is_sort_functions)
            std::sort(ctilde_basis_vectors.begin(), ctilde_basis_vectors.end(), compare);

        if (num_ctilde_max < ctilde_basis_vectors.size())
            num_ctilde_max = ctilde_basis_vectors.size();
    }

    ACECTildeBasisSet dest;
    //imitate the copy constructor of ACECTildeBasisSet:

    // ACECTildeBasisSet::_copy_scalar_memory(const ACECTildeBasisSet
    dest.ACEFlattenBasisSet::_copy_scalar_memory(*this);
    dest.num_ctilde_max = num_ctilde_max;
    dest.is_sort_functions = is_sort_functions;

    // ACECTildeBasisSet::_copy_dynamic_memory(const ACECTildeBasisSet &src)

    // call copy_dynamic memory of ACEAbstractBasisSet but not for ACEFlattenbasisSet
    dest.ACEFlattenBasisSet::_copy_dynamic_memory(*this);

    //could not copied, should be recomputed !!!
//    this->basis = new ACECTildeBasisFunction *[src.nelements];
//    this->basis_rank1 = new ACECTildeBasisFunction *[src.nelements];
//    this->full_c_tildes_rank1 = new DOUBLE_TYPE[src.coeff_array_total_size_rank1];
//    this->full_c_tildes = new DOUBLE_TYPE[src.coeff_array_total_size];

    //pack into 1D array with all basis functions
    dest.flatten_basis(mu0_ctilde_basis_vector);
    dest.pack_flatten_basis();

    return dest;
};


void ACEBBasisSet::initialize_basis(BBasisConfiguration &basisSetup) {

    ACEClebschGordan clebsch_gordan;

    set<string> elements_set;

    //compute the maximum of rank, ndensity, lmax, nradmax, nradbase and cutoff
    for (auto &func_spec_block: basisSetup.funcspecs_blocks) {
        func_spec_block.update_params(); // elements_vec: Al -> Al Al and another checks
        func_spec_block.validate_individual_functions();
        func_spec_block.validate_radcoefficients();

        if (func_spec_block.rankmax > this->rankmax)
            this->rankmax = func_spec_block.rankmax;
        if (func_spec_block.ndensityi > this->ndensitymax)
            this->ndensitymax = func_spec_block.ndensityi;
        if (func_spec_block.lmaxi > this->lmax)
            this->lmax = func_spec_block.lmaxi;
        if (func_spec_block.nradbaseij > this->nradbase)
            this->nradbase = func_spec_block.nradbaseij;

        if (func_spec_block.nradmaxi > this->nradmax)
            this->nradmax = func_spec_block.nradmaxi;

        if (func_spec_block.rcutij > this->cutoffmax)
            this->cutoffmax = func_spec_block.rcutij;

        for (const auto &el: func_spec_block.elements_vec)
            elements_set.insert(el);
    }

    //order elements lexicographically
    vector<string> element_names;
    std::copy(elements_set.begin(), elements_set.end(), std::back_inserter(element_names));
    std::sort(element_names.begin(), element_names.end());
    int number_of_elements = elements_set.size();

    //update elements_to_index_map
    for (const auto &el: element_names)
        if (!is_key_in_map(el, elements_to_index_map)) {
            int current_map_size = elements_to_index_map.size();
            elements_to_index_map[el] = static_cast<SPECIES_TYPE>(current_map_size);
        }

    //default initialize with ChebExpCos
    vector<vector<string>> radbasename_ij(number_of_elements, vector<string>(number_of_elements, "ChebExpCos"));


    //TODO: read all inner_cutoff_func  (check consistency) and r_in, delta_in
    string inner_cutoff_func;
    vector<vector<DOUBLE_TYPE>> r_in_ij(number_of_elements, vector<DOUBLE_TYPE>(number_of_elements, 0));
    vector<vector<DOUBLE_TYPE>> delta_in_ij(number_of_elements, vector<DOUBLE_TYPE>(number_of_elements, 0));
    for (auto &func_spec_block: basisSetup.funcspecs_blocks) {
        SPECIES_TYPE ele_i = elements_to_index_map[func_spec_block.elements_vec[0]];
        SPECIES_TYPE ele_j = ele_i;
        if (func_spec_block.number_of_species == 2)
            ele_j = elements_to_index_map[func_spec_block.elements_vec[1]];

        if (func_spec_block.number_of_species <= 2) {
            radbasename_ij.at(ele_i).at(ele_j) = func_spec_block.radbase;
            if (inner_cutoff_func.empty())
                inner_cutoff_func = func_spec_block.inner_cutoff_type;
            else if (inner_cutoff_func != func_spec_block.inner_cutoff_type) {
                stringstream ss;
                ss << "`inner_cutoff_type` should be identical for all one- and two-species blocks, but `"
                   << func_spec_block.inner_cutoff_type << "` for "
                   << func_spec_block.block_name << " is different from `" << inner_cutoff_func << "`";
                throw invalid_argument(ss.str());
            }
            r_in_ij.at(ele_i).at(ele_j) = func_spec_block.r_in;
            delta_in_ij.at(ele_i).at(ele_j) = func_spec_block.delta_in;
        }
    }

    if (basisSetup.deltaSplineBins > 0)
        deltaSplineBins = basisSetup.deltaSplineBins;
    else
        throw invalid_argument("ACEBBasisSet:deltaSplineBins should be positive");

    nelements = elements_to_index_map.size();

    clebsch_gordan.init(2 * lmax);
    spherical_harmonics.init(lmax);
    if (!radial_functions)
        radial_functions = new ACERadialFunctions();


    radial_functions->init(nradbase, lmax, nradmax,
                           basisSetup.deltaSplineBins,
                           nelements,
                           radbasename_ij);
    radial_functions->inner_cutoff_type = inner_cutoff_func;

    radial_functions->cut_in = r_in_ij;
    radial_functions->dcut_in = delta_in_ij;

    is_sort_functions = basisSetup.is_sort_functions;
    E0vals.init(nelements, "E0 values");
    E0vals.fill(0.0);
    //setting up the basis functions, from file or like that
    num_ms_combinations_max = 0;

    // loop over functions specifications blocks
    // fill-in rank, max rank, ndensity max
    // from PAIR (A-A or A-B) species blocks:
    //  - set radial_functions.lambda, cut, dcut, crad and prehc and lambdahc,
    //  - rho_core_cutoffs, drho_core_cutoffs and FS_parameters
    for (auto &func_spec_block: basisSetup.funcspecs_blocks) {
        //below, only pair_species blocks (i.e. A-A or A-B) are considered
        if (func_spec_block.number_of_species > 2) continue;

        //common part for 1- and 2-species blocks:


        SPECIES_TYPE ele_i = elements_to_index_map[func_spec_block.elements_vec[0]];
        SPECIES_TYPE ele_j;
        if (func_spec_block.elements_vec.size() == 1)
            ele_j = ele_i;
        else
            ele_j = elements_to_index_map[func_spec_block.elements_vec[1]];
#ifdef DEBUG_READ_YAML
        cout << "Update radial function parameters for " << ele_i << "-" << ele_j << " pair" << endl;
#endif
        // create and save ACEBondSpecification into map_bond_specifications
        ACEBondSpecification bondSpecification;
        bondSpecification.lmax = func_spec_block.lmaxi;
        bondSpecification.nradbasemax = func_spec_block.nradbaseij;
        bondSpecification.nradmax = func_spec_block.nradmaxi;
        bondSpecification.radbasename = func_spec_block.radbase;
        bondSpecification.radparameters = func_spec_block.radparameters;
        bondSpecification.radcoefficients = func_spec_block.radcoefficients;

        bondSpecification.prehc = func_spec_block.core_rep_parameters.at(0); //prehc
        bondSpecification.lambdahc = func_spec_block.core_rep_parameters.at(1); //lambdahc

        bondSpecification.rcut = func_spec_block.rcutij;
        bondSpecification.dcut = func_spec_block.dcutij;

        bondSpecification.inner_cutoff_type = func_spec_block.inner_cutoff_type;
        bondSpecification.rcut_in = func_spec_block.r_in;
        bondSpecification.dcut_in = func_spec_block.delta_in;

        map_bond_specifications[make_pair(ele_i, ele_j)] = bondSpecification;
        //compare with symmetric bond A-B: B-A
        auto symm_bond = make_pair(ele_j, ele_i);
        if (map_bond_specifications.find(symm_bond) != map_bond_specifications.end()) {
            //symmetric bond already exists
            auto symm_bond_spec = map_bond_specifications.at(symm_bond);
            if (symm_bond_spec != bondSpecification) {
                stringstream ss;
                ss << "Bonds specifications for pair (" << (int) ele_i << "," << (int) ele_j << ") are inconsistent";
                throw invalid_argument(ss.str());
            }
        }

        // use ACEBondSpecification to partially fill radial_functions
        radial_functions->lambda(ele_i, ele_j) = bondSpecification.radparameters[0];
        radial_functions->cut(ele_i, ele_j) = bondSpecification.rcut;
        radial_functions->dcut(ele_i, ele_j) = bondSpecification.dcut;
        for (NS_TYPE n = 0; n < bondSpecification.nradmax; n++)
            for (LS_TYPE l = 0; l <= bondSpecification.lmax; l++)
                for (NS_TYPE k = 0; k < bondSpecification.nradbasemax; k++) {
                    radial_functions->crad(ele_i, ele_j, n, l, k) = bondSpecification.radcoefficients.at(n).at(l).at(k);
                }
        //set hard-core repulsion core-repulsion parameters:
        radial_functions->prehc(ele_i, ele_j) = func_spec_block.core_rep_parameters.at(0); //prehc
        radial_functions->lambdahc(ele_i, ele_j) = func_spec_block.core_rep_parameters.at(1); //lambdahc

        // create and save ACEEmbeddingSpecification into map_embedding_specifications
        if (func_spec_block.number_of_species == 1) {
            ACEEmbeddingSpecification embedding_specification;
            embedding_specification.npoti = func_spec_block.npoti;
            embedding_specification.FS_parameters = func_spec_block.fs_parameters;
            embedding_specification.ndensity = func_spec_block.ndensityi;

            if (embedding_specification.FS_parameters.size() != 2 * embedding_specification.ndensity) {
                stringstream ss;
                ss << "Number of fs_parameters = " << embedding_specification.FS_parameters.size() <<
                   " is not equal to ndensity*2 = " << 2 * embedding_specification.ndensity << " for ele_i = " << ele_i;
                throw invalid_argument(ss.str());
            }

            embedding_specification.rho_core_cutoff = func_spec_block.rho_cut;
            embedding_specification.drho_core_cutoff = func_spec_block.drho_cut;

            //create empty embedding_specifications
            map_embedding_specifications[ele_i] = embedding_specification;
        }
    } // end loop over pairs_species_blocks

    radial_functions->setuplookupRadspline();

    //invert "elements_to_index_map" to index->element array "elements_name"
    elements_name = new string[nelements];
    for (auto const &elem_ind : elements_to_index_map) {
        elements_name[elem_ind.second] = elem_ind.first;
    }

    //0 dim - X_0: ele_0; central element type(0..nelements-1)
    //1 dim - vector<C_tilde_B_basis_function> for different [X_, n_, l_]
    //mu0_ctilde_basis_vector.resize(nelements);
    mu0_bbasis_vector.resize(nelements);

    // loop over all B-basis functions specification blocks,
    // construction of actual ACEBBasisFunction
    for (auto species_block: basisSetup.funcspecs_blocks) { // n
        SPECIES_TYPE mu0 = elements_to_index_map[species_block.mu0];
        NS_TYPE *nr;
        LS_TYPE *ls;
        LS_TYPE *LS;
        DOUBLE_TYPE *cs;


        if (species_block.funcspecs.empty()) continue;

        //[basis_ind]
        vector<ACEBBasisFunction> &basis = mu0_bbasis_vector[mu0];

        for (auto &curr_bFuncSpec: species_block.funcspecs) {
            ACEBBasisFunction new_basis_func;
            if (curr_bFuncSpec.skip)
                continue;
            //BBasisFunctionSpecification &curr_bFuncSpec = species_block.bfuncspec_vector[rank - 1][basis_ind];
            RANK_TYPE rank = curr_bFuncSpec.rank;
            nr = &curr_bFuncSpec.ns[0];
            ls = &curr_bFuncSpec.ls[0];
            cs = &curr_bFuncSpec.coeffs[0]; // len = ndensity
            if (rank > 2)
                LS = &curr_bFuncSpec.LS[0];
            else
                LS = nullptr;

            try {
                generate_basis_function_n_body(rank, nr, ls, LS, new_basis_func,
                                               clebsch_gordan, true);
            } catch (const invalid_argument &exc) {
                stringstream s;
                s << curr_bFuncSpec.to_string() << " could not be constructed: " << endl << exc.what();
                throw invalid_argument(s.str());
            }

            new_basis_func.mu0 = elements_to_index_map[curr_bFuncSpec.elements.at(0)];
            //TODO: move new mus here
            for (RANK_TYPE r = 1; r <= rank; r++)
                new_basis_func.mus[r - 1] = elements_to_index_map[curr_bFuncSpec.elements.at(r)];

            new_basis_func.ndensity = species_block.ndensityi;
            new_basis_func.coeff = new DOUBLE_TYPE[new_basis_func.ndensity];
            for (DENSITY_TYPE p = 0; p < species_block.ndensityi; ++p)
                new_basis_func.coeff[p] = cs[p];

            if (num_ms_combinations_max < new_basis_func.num_ms_combs)
                num_ms_combinations_max = new_basis_func.num_ms_combs;

            basis.emplace_back(new_basis_func);

            //check that new_basis_func::mus, ns, ls are valid wrt. bond_spec
            for (RANK_TYPE r = 0; r < new_basis_func.rank; r++) {
                auto bond_pair = make_pair(new_basis_func.mu0, new_basis_func.mus[r]);
                auto &bond_spec = map_bond_specifications.at(bond_pair);
                if (new_basis_func.rank == 1) { //check nradbase
                    if (new_basis_func.ns[r] > bond_spec.nradbasemax) {
                        stringstream ss;
                        ss << "BASIS INCONSISTENCY: B-function " << endl;
                        ss << curr_bFuncSpec.to_string() << endl;
                        ss << " has larger ns = " << new_basis_func.ns[r]
                           << " than the nradbasemax=" << bond_spec.nradbasemax << " for corresponding bond ("
                           << (int) new_basis_func.mu0 << ", " << (int) new_basis_func.mus[r] << ")";
                        throw invalid_argument(ss.str());
                    }
                } else { //check nradmax
                    if (new_basis_func.ns[r] > bond_spec.nradmax) {
                        stringstream ss;
                        ss << "BASIS INCONSISTENCY: B-function " << endl;
                        ss << curr_bFuncSpec.to_string() << endl;
                        ss << " has larger ns = " << new_basis_func.ns[r]
                           << " than the nradmax=" << bond_spec.nradmax << " for corresponding bond ("
                           << (int) new_basis_func.mu0 << ", " << (int) new_basis_func.mus[r] << ")";
                        throw invalid_argument(ss.str());
                    }
                }

                //check lmax
                if (new_basis_func.ls[r] > bond_spec.lmax) {
                    stringstream ss;
                    ss << "BASIS INCONSISTENCY: B-function " << endl;
                    ss << curr_bFuncSpec.to_string() << endl;
                    ss << " has larger ls = " << new_basis_func.ls[r]
                       << " than the lmax=" << bond_spec.lmax << " for corresponding bond ("
                       << (int) new_basis_func.mu0 << ", " << (int) new_basis_func.mus[r] << ")";
                    throw invalid_argument(ss.str());
                }
            }


        } //end loop func specs
        //optinally sort the bbasis
        if (is_sort_functions)
            std::sort(basis.begin(), basis.end(), compare);
    }//end loop over func_spec_blocks

    metadata = basisSetup.metadata;
    auxdata = basisSetup.auxdata;

    compress_basis_functions();
    flatten_basis();
    pack_flatten_basis();
}

void convert_B_to_Ctilde_basis_functions(const vector<ACEBBasisFunction> &b_basis_vector,
                                         vector<ACECTildeBasisFunction> &ctilde_basis_vector) {
    Basis_functions_map basis_functions_map;
    group_basis_functions_by_index(b_basis_vector, basis_functions_map);
#ifdef DEBUG_C_TILDE
    for(int r=0; r<rankmax; ++r) {
        auto basis_functions_map = basis_functions_map[r];
        if(basis_functions_map.empty()) continue;
        cout<<"rankmax="<<(int)r<<"\t";
        cout<<"number of b_basis_vector functions = "<<basis_functions_map.size()<<endl;

        for (auto & it : basis_functions_map) {
            const Vector_ns &num_ms_combs = get<1>(it.first);
            const Vector_ls &ls = get<2>(it.first);
            Basis_function_ptr_list bas_list = it.second;

            cout << "size=" << bas_list.size() << endl;

            for (auto &it : bas_list) {
                print_B_basis_function(*it);
            }
            cout << endl;

        }

    }
#endif

    summation_over_LS(basis_functions_map, ctilde_basis_vector);
}


void BBasisConfiguration::save(const string &yaml_file_name) {
    YAML_PACE::Node out_yaml;

    YAML_PACE::Node global_yaml;
    global_yaml["DeltaSplineBins"] = deltaSplineBins;


    vector<YAML_PACE::Node> species;
    for (auto &block: funcspecs_blocks) {
        YAML_PACE::Node block_yaml = block.to_YAML();
        species.emplace_back(block_yaml);
    }

    if (metadata.size() > 0)
        out_yaml["metadata"] = metadata;

    if (!auxdata.empty())
        out_yaml["auxdata"] = auxdata.to_YAML();

    out_yaml["global"] = global_yaml;
    out_yaml["species"] = species;

    YAML_PACE::Emitter yaml_emitter;
    yaml_emitter << out_yaml;

    std::ofstream fout(yaml_file_name);
    fout << yaml_emitter.c_str() << endl;
}

void BBasisConfiguration::load(const string &yaml_file_name, bool validate) {
    Input input;
    input.parse_input(yaml_file_name);
    this->metadata = input.global.metadata;
    this->auxdata = input.global.auxdata;
    this->deltaSplineBins = input.global.DeltaSplineBins;
    this->funcspecs_blocks = input.bbasis_func_spec_blocks_vector; // TODO: re-order funcspecs_blocks
    if (validate)
        this->validate(true);
}

vector<DOUBLE_TYPE> BBasisConfiguration::get_all_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (auto &block: this->funcspecs_blocks) {
        auto coeffs = block.get_all_coeffs();
        res.insert(end(res), begin(coeffs), end(coeffs));
    }
    return res;
}

vector<DOUBLE_TYPE> BBasisConfiguration::get_radial_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (auto &block: this->funcspecs_blocks) {
        auto coeffs = block.get_radial_coeffs();
        res.insert(end(res), begin(coeffs), end(coeffs));
    }
    return res;
}

vector<DOUBLE_TYPE> BBasisConfiguration::get_func_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (auto &block: this->funcspecs_blocks) {
        auto coeffs = block.get_func_coeffs();
        res.insert(end(res), begin(coeffs), end(coeffs));
    }
    return res;
}

void BBasisConfiguration::set_all_coeffs(const vector<DOUBLE_TYPE> &new_all_coeffs) {
    size_t ind = 0;
    for (auto &block: this->funcspecs_blocks) {
        size_t expected_num_of_coeffs = block.get_number_of_coeffs();
        vector<DOUBLE_TYPE> block_new_coeffs = vector<DOUBLE_TYPE>(new_all_coeffs.begin() + ind,
                                                                   new_all_coeffs.begin() + ind +
                                                                   expected_num_of_coeffs);
        block.set_all_coeffs(block_new_coeffs);
        ind += expected_num_of_coeffs;
    }
}

void BBasisConfiguration::set_radial_coeffs(const vector<DOUBLE_TYPE> &new_all_coeffs) {
    size_t ind = 0;
    for (auto &block: this->funcspecs_blocks) {
        size_t expected_num_of_coeffs = block.get_number_of_radial_coeffs();
        vector<DOUBLE_TYPE> block_new_coeffs = vector<DOUBLE_TYPE>(new_all_coeffs.begin() + ind,
                                                                   new_all_coeffs.begin() + ind +
                                                                   expected_num_of_coeffs);
        block.set_radial_coeffs(block_new_coeffs);
        ind += expected_num_of_coeffs;
    }
}

void BBasisConfiguration::set_func_coeffs(const vector<DOUBLE_TYPE> &new_all_coeffs) {
    size_t ind = 0;
    for (auto &block: this->funcspecs_blocks) {
        size_t expected_num_of_coeffs = block.get_number_of_func_coeffs();
        vector<DOUBLE_TYPE> block_new_coeffs = vector<DOUBLE_TYPE>(new_all_coeffs.begin() + ind,
                                                                   new_all_coeffs.begin() + ind +
                                                                   expected_num_of_coeffs);
        block.set_func_coeffs(block_new_coeffs);
        ind += expected_num_of_coeffs;
    }
}

bool BBasisConfiguration::validate(bool raise_exception) {
    // validate by trying to create ACEBBasisSet
    try {
        for (auto &func_spec_block: funcspecs_blocks) {
            func_spec_block.update_params();
            func_spec_block.validate_individual_functions();
            func_spec_block.validate_radcoefficients();
        }

        //consistency with higher n-ary blocks elsewhere are checked inside ACEBBasisSet c'tor
        ACEBBasisSet bbasis(*this);
        return true;
    } catch (std::invalid_argument const &err) {
        if (raise_exception)
            throw err;
        else
            return false;
    }
}


YAML_PACE::Node BBasisFunctionsSpecificationBlock::to_YAML() const {
    YAML_PACE::Node block_node;
    block_node["speciesblock"] = this->block_name; //join(this->elements_vec, " ");
    block_node["speciesblock"].SetStyle(YAML_PACE::EmitterStyle::Flow);

    //single species
    if (number_of_species == 1) {
        block_node["ndensityi"] = ndensityi;
        block_node["npoti"] = npoti;
        block_node["parameters"] = fs_parameters;
        block_node["parameters"].SetStyle(YAML_PACE::EmitterStyle::Flow);
        block_node["rho_core_cut"] = rho_cut;
        block_node["drho_core_cut"] = drho_cut;
    }

    //pairs
    if (number_of_species <= 2) {
        block_node["nradmaxi"] = nradmaxi;
        block_node["lmaxi"] = lmaxi;
        block_node["rcutij"] = rcutij;
        block_node["dcutij"] = dcutij;
        block_node["NameOfCutoffFunctionij"] = NameOfCutoffFunctionij;

        block_node["r_in"] = r_in;
        block_node["delta_in"] = delta_in;
        block_node["inner_cutoff_type"] = inner_cutoff_type;

        block_node["nradbaseij"] = nradbaseij;

        block_node["radbase"] = radbase;
        block_node["radparameters"] = radparameters;
        block_node["radparameters"].SetStyle(YAML_PACE::EmitterStyle::Flow);

        block_node["radcoefficients"] = radcoefficients;
        block_node["radcoefficients"].SetStyle(YAML_PACE::EmitterStyle::Flow);

        block_node["core-repulsion"] = core_rep_parameters;
        block_node["core-repulsion"].SetStyle(YAML_PACE::EmitterStyle::Flow);
    }

    vector<YAML_PACE::Node> nbody;

    for (auto &funcspec: funcspecs) {
        nbody.emplace_back(funcspec.to_YAML());
    }

    block_node["nbody"] = nbody;

    return block_node;
}

vector<DOUBLE_TYPE> BBasisFunctionsSpecificationBlock::get_radial_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++) {
                res.emplace_back(this->radcoefficients.at(n).at(l).at(k));
            }

    return res;
}

vector<DOUBLE_TYPE> BBasisFunctionsSpecificationBlock::get_func_coeffs() const {
    vector<DOUBLE_TYPE> res;

    for (auto &f: this->funcspecs) {
        for (auto c: f.coeffs)
            res.emplace_back(c);
    }

    return res;
}

vector<DOUBLE_TYPE> BBasisFunctionsSpecificationBlock::get_all_coeffs() const {
    vector<DOUBLE_TYPE> res;
    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++) {
                res.emplace_back(this->radcoefficients.at(n).at(l).at(k));
            }

    for (auto &f: this->funcspecs) {
        for (auto c: f.coeffs)
            res.emplace_back(c);
    }

    return res;
}

void BBasisFunctionsSpecificationBlock::set_all_coeffs(const vector<DOUBLE_TYPE> &new_coeffs) {
    size_t total_size = this->get_number_of_coeffs();
    if (total_size != new_coeffs.size())
        throw invalid_argument("Number of new coefficients " + to_string(new_coeffs.size()) +
                               " differs from expected number of coefficients: " + to_string(total_size));
    size_t ind = 0;

    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++, ind++) {
                this->radcoefficients.at(n).at(l).at(k) = new_coeffs[ind];
            }


    for (auto &spec: this->funcspecs) {
        for (DENSITY_TYPE p = 0; p < spec.coeffs.size(); p++, ind++)
            spec.coeffs[p] = new_coeffs[ind];
    }
}

void BBasisFunctionsSpecificationBlock::set_radial_coeffs(const vector<DOUBLE_TYPE> &new_coeffs) {
    size_t total_size = this->get_number_of_radial_coeffs();
    if (total_size != new_coeffs.size())
        throw invalid_argument("Number of new coefficients radial " + to_string(new_coeffs.size()) +
                               " differs from expected number of radial coefficients: " + to_string(total_size));
    size_t ind = 0;

    for (NS_TYPE n = 0; n < this->nradmaxi; n++)
        for (LS_TYPE l = 0; l <= this->lmaxi; l++)
            for (NS_TYPE k = 0; k < this->nradbaseij; k++, ind++) {
                this->radcoefficients.at(n).at(l).at(k) = new_coeffs[ind];
            }
}

void BBasisFunctionsSpecificationBlock::set_func_coeffs(const vector<DOUBLE_TYPE> &new_coeffs) {
    size_t total_size = this->get_number_of_func_coeffs();
    if (total_size != new_coeffs.size())
        throw invalid_argument("Number of new func coefficients " + to_string(new_coeffs.size()) +
                               " differs from expected number of func coefficients: " + to_string(total_size));
    size_t ind = 0;

    for (auto &spec: this->funcspecs) {
        for (DENSITY_TYPE p = 0; p < spec.coeffs.size(); p++, ind++)
            spec.coeffs[p] = new_coeffs[ind];
    }
}

int BBasisFunctionsSpecificationBlock::get_number_of_radial_coeffs() const {
    size_t num = this->nradmaxi * (this->lmaxi + 1) * this->nradbaseij;
    return num;
}

int BBasisFunctionsSpecificationBlock::get_number_of_func_coeffs() const {
    size_t num = 0;
    for (auto &func: funcspecs) {
        num += func.coeffs.size();
    }
    return num;
}


int BBasisFunctionsSpecificationBlock::get_number_of_coeffs() const {
    size_t num = this->nradmaxi * (this->lmaxi + 1) * this->nradbaseij; //this->radcoefficients.size();
    for (auto &func: funcspecs) {
        num += func.coeffs.size();
    }
    return num;
}

void BBasisFunctionsSpecificationBlock::validate_radcoefficients() {
    if (radcoefficients.size() < nradmaxi) {
        stringstream s;
        s << "Error: species block " << block_name
          << " has insufficient number of radcoefficients (shape=("
          << radcoefficients.size()
          << ",...)) whereas nradmaxi = " << nradmaxi << "";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    } else {
        for (NS_TYPE n = 0; n < nradmaxi; n++)
            if (radcoefficients.at(n).size() < lmaxi + 1) {
                stringstream s;
                s << "Error: species block " << block_name
                  << " has insufficient number in radcoefficients[" << n + 1 << "] (shape=("
                  << radcoefficients.at(n).size()
                  << ",...)) whereas lmaxi+1 = " << lmaxi + 1 << "";
                cerr << "Exception: " << s.str();
                throw invalid_argument(s.str());
            } else {
                for (LS_TYPE l = 0; l <= lmaxi; l++)
                    if (radcoefficients.at(n).at(l).size() < nradbaseij) {
                        stringstream s;
                        s << "Error: species block " << block_name
                          << " has insufficient number radcoefficients[" << n + 1 << "][" << l << "].size="
                          << radcoefficients.at(n).at(l).size()
                          << " whereas it should be nradbase = " << nradbaseij << "";
                        cerr << "Exception: " << s.str();
                        throw invalid_argument(s.str());
                    }
            }
    }
}


void BBasisFunctionsSpecificationBlock::validate_individual_functions() {
    int func_n_density = 1; // initialize with 1

    LS_TYPE block_lmax = 0;
    NS_TYPE block_nradmax = 0;
    NS_TYPE block_nradbasemax = 0;

    if (funcspecs.size() > 0) {
        func_n_density = funcspecs.at(0).coeffs.size();
    }
    for (auto &funcSpec: funcspecs) {
        funcSpec.validate();

        if (func_n_density != funcSpec.coeffs.size()) {
            stringstream s;
            s << funcSpec.to_string() << ":" << endl
              << "Number of function 'coeffs'(" << funcSpec.coeffs.size()
              << ") is inconsistent with the expected density(" << func_n_density << ")";
            throw invalid_argument(s.str());
        }

        if (funcSpec.rank == 1) {
            if (block_nradbasemax < funcSpec.ns[0])
                block_nradbasemax = funcSpec.ns[0];
        } else {
            NS_TYPE ns_max = *max_element(funcSpec.ns.begin(), funcSpec.ns.end());
            if (block_nradmax < ns_max)
                block_nradmax = ns_max;
        }

        LS_TYPE ls_max = *max_element(funcSpec.ls.begin(), funcSpec.ls.end());
    }

    //check for consitency only for 1-, 2-species blocks (for pair interaction only!)
    //check consistency with higher n-ary blocks is checked elsewhere (in ACEBBasisSet)
    if (this->number_of_species <= 2) {
        if (this->lmaxi < block_lmax)
            throw invalid_argument("Function specifications max 'ls' is larger than block lmaxi");

        if (this->nradbaseij < block_nradbasemax)
            throw invalid_argument("Function specifications max 'ns' is larger than block nradbaseij");

        if (this->nradmaxi < block_nradmax)
            throw invalid_argument("Function specifications max 'ns' is larger than block nradmaxi");
    }

}

void BBasisFunctionsSpecificationBlock::update_params() {
    int block_rankmax = 0; // initialize with zero

    int func_n_density = this->ndensityi > 0 ? this->ndensityi : 1; // initialize with 1 or more

    if (funcspecs.size() > 0) {
        func_n_density = funcspecs.at(0).coeffs.size();
    }
    this->ndensityi = func_n_density;

    for (auto &funcSpec: funcspecs) {
        if (funcSpec.rank > block_rankmax) block_rankmax = funcSpec.rank;
    }

    this->elements_vec = split_key(this->block_name);

    //duplicate elements_vec if single-species case
    if (this->elements_vec.size() == 1)
        this->elements_vec.emplace_back(this->elements_vec[0]);

    set<string> elements_set;
    for (auto const &el: this->elements_vec)
        if (elements_set.count(el) == 0)
            elements_set.insert(el);

    this->rankmax = block_rankmax;
    this->number_of_species = elements_set.size();
    this->mu0 = this->elements_vec[0];
}

BBasisFunctionsSpecificationBlock BBasisFunctionsSpecificationBlock::copy() const {
    BBasisFunctionsSpecificationBlock new_block = *this;
    return new_block;
}

vector<vector<SPECIES_TYPE>> ACEBBasisSet::get_crad_coeffs_mask() const {
    SPECIES_TYPE mu_i, mu_j;
    vector<vector<SPECIES_TYPE>> crad_mask;

    //(0,0),(0,1),(1,1)  NO(1,0)
    for (mu_i = 0; mu_i < nelements; mu_i++)
        for (mu_j = mu_i; mu_j < nelements; mu_j++) { // collect only upper-diag coefficients
            auto bond_pair = make_pair(mu_i, mu_j);
            if (map_bond_specifications.find(bond_pair) == map_bond_specifications.end())
                continue;
            const auto &bondSpec = map_bond_specifications.at(bond_pair);
            for (NS_TYPE n = 0; n < bondSpec.nradmax; n++)
                for (LS_TYPE l = 0; l <= bondSpec.lmax; l++)
                    for (NS_TYPE k = 0; k < bondSpec.nradbasemax; k++) {
                        vector<SPECIES_TYPE> v;
                        if (mu_i != mu_j)
                            v = {mu_i, mu_j};
                        else
                            v = {mu_i};

                        crad_mask.emplace_back(v);
                    }
        }
    return crad_mask;
}


vector<DOUBLE_TYPE> ACEBBasisSet::get_crad_coeffs() const {
    SPECIES_TYPE mu_i, mu_j;
    vector<DOUBLE_TYPE> crad_coeffs;

    //(0,0),(0,1),(1,1)  NO(1,0)
    for (mu_i = 0; mu_i < nelements; mu_i++)
        for (mu_j = mu_i; mu_j < nelements; mu_j++) { // collect only upper-diag coefficients
            // NOTE: not all pairs of elements could be presented, necessary to check the bond presence
            auto bond_pair = make_pair(mu_i, mu_j);
            // if no given key pair found, skip
            if (map_bond_specifications.find(bond_pair) == map_bond_specifications.end())
                continue;
            const auto &bondSpec = map_bond_specifications.at(bond_pair);
            for (NS_TYPE n = 0; n < bondSpec.nradmax; n++)
                for (LS_TYPE l = 0; l <= bondSpec.lmax; l++)
                    for (NS_TYPE k = 0; k < bondSpec.nradbasemax; k++) {
                        auto v = radial_functions->crad(mu_i, mu_j, n, l, k);
                        crad_coeffs.emplace_back(v);
                    }
        }
    return crad_coeffs;
}

void ACEBBasisSet::set_crad_coeffs(const vector<DOUBLE_TYPE> &crad_flatten_coeffs) {

    SPECIES_TYPE mu_i, mu_j;
    size_t crad_ind = 0;

    for (mu_i = 0; mu_i < nelements; mu_i++)
        for (mu_j = mu_i; mu_j < nelements; mu_j++) {
            // NOTE: not all pairs of elements could be presented, necessary to check the bond presence
            auto bond_pair = make_pair(mu_i, mu_j);
            // if no given key pair found, skip
            if (map_bond_specifications.find(bond_pair) == map_bond_specifications.end())
                continue;

            auto &bondSpec = map_bond_specifications.at(bond_pair);
            auto sym_bond_pair = make_pair(mu_j, mu_i);
            auto &sym_bondSpec = map_bond_specifications.at(sym_bond_pair);

            for (NS_TYPE n = 0; n < bondSpec.nradmax; n++)
                for (LS_TYPE l = 0; l <= bondSpec.lmax; l++)
                    for (NS_TYPE k = 0; k < bondSpec.nradbasemax; k++) {
                        auto v = crad_flatten_coeffs.at(crad_ind);
                        radial_functions->crad(mu_i, mu_j, n, l, k) = v;
                        bondSpec.radcoefficients[n][l][k] = v;
                        //set symmetrized coefficients
                        if (mu_i != mu_j) {
                            radial_functions->crad(mu_j, mu_i, n, l, k) = v;
                            sym_bondSpec.radcoefficients[n][l][k] = v;
                        }
                        crad_ind++;
                    }
        }

    radial_functions->setuplookupRadspline();
}


vector<vector<SPECIES_TYPE>> ACEBBasisSet::get_basis_coeffs_mask() const {
    vector<vector<SPECIES_TYPE>> coeffs_mask;
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis_rank1[mu][func_ind].ndensity; p++) {
                coeffs_mask.emplace_back(basis_rank1[mu][func_ind].get_unique_species());
            }
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis[mu][func_ind].ndensity; p++) {
                coeffs_mask.emplace_back(basis[mu][func_ind].get_unique_species());
            }
        }
    }
    return coeffs_mask;
}


vector<DOUBLE_TYPE> ACEBBasisSet::get_basis_coeffs() const {
    vector<DOUBLE_TYPE> coeffs;
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis_rank1[mu][func_ind].ndensity; p++)
                coeffs.emplace_back(basis_rank1[mu][func_ind].coeff[p]);
        }

        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis[mu][func_ind].ndensity; p++)
                coeffs.emplace_back(basis[mu][func_ind].coeff[p]);
        }
    }
    return coeffs;
}

vector<tuple<SPECIES_TYPE, int, vector<SPECIES_TYPE>, DENSITY_TYPE>>
ACEBBasisSet::get_basis_coeffs_markup() const {
    //  coeffs_id: tuple(species_type, rank, (mu_0 + mus ), dens_id)
    //  the order is the same as in get_basis_coeffs

    vector<tuple<SPECIES_TYPE, int, vector<SPECIES_TYPE>, DENSITY_TYPE >> markup;

    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {

        //rank 1
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++) {
            auto &func = basis_rank1[mu][func_ind];
            vector<SPECIES_TYPE> total_mus(func.rank + 1);
            total_mus[0] = func.mu0;
            for (RANK_TYPE r = 0; r < func.rank; r++)
                total_mus[r + 1] = func.mus[r];

            for (DENSITY_TYPE p = 0; p < func.ndensity; p++)
                markup.emplace_back(make_tuple(mu, (int) func.rank, total_mus, p));
        }

        // rank > 1
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size[mu]; func_ind++) {

            auto &func = basis[mu][func_ind];

            vector<SPECIES_TYPE> total_mus(func.rank + 1);
            total_mus[0] = func.mu0;
            for (RANK_TYPE r = 0; r < func.rank; r++)
                total_mus[r + 1] = func.mus[r];

            for (DENSITY_TYPE p = 0; p < func.ndensity; p++)
                markup.emplace_back(make_tuple(mu, (int) func.rank, total_mus, p));
        }
    }
    return markup;
}

vector<DOUBLE_TYPE> ACEBBasisSet::get_all_coeffs() const {
    auto cradCoeffs = this->get_crad_coeffs();
    auto basisCoeffs = this->get_basis_coeffs();

    vector<DOUBLE_TYPE> coeffs;
    coeffs.reserve(cradCoeffs.size() + basisCoeffs.size());
    coeffs.insert(coeffs.end(), cradCoeffs.begin(), cradCoeffs.end());
    coeffs.insert(coeffs.end(), basisCoeffs.begin(), basisCoeffs.end());

    return coeffs;
}

vector<vector<SPECIES_TYPE>> ACEBBasisSet::get_all_coeffs_mask() const {
    auto cradCoeffs_mask = this->get_crad_coeffs_mask();
    auto basisCoeffs_mask = this->get_basis_coeffs_mask();

    vector<vector<SPECIES_TYPE>> coeffs_mask;
    coeffs_mask.reserve(cradCoeffs_mask.size() + basisCoeffs_mask.size());
    coeffs_mask.insert(coeffs_mask.end(), cradCoeffs_mask.begin(), cradCoeffs_mask.end());
    coeffs_mask.insert(coeffs_mask.end(), basisCoeffs_mask.begin(), basisCoeffs_mask.end());

    return coeffs_mask;
}


void ACEBBasisSet::set_basis_coeffs(const vector<DOUBLE_TYPE> &basis_coeffs_vector) {
    size_t coeffs_ind = 0;
    size_t sequential_func_ind = 0;
//    cout << "set_basis_coeffs basis_coeffs_vector.size=" << basis_coeffs_vector.size() << endl;
//    cout << "mu0_bbasis_vector.size=" << mu0_bbasis_vector.size() << endl;
//    cout << "LAB 1" << endl;
    for (SPECIES_TYPE mu = 0; mu < nelements; mu++) {
        sequential_func_ind = 0;
//        cout << "lab2.1: mu=" << mu << endl;
//        cout << " mu0_bbasis_vector.at(mu).size=" << mu0_bbasis_vector.at(mu).size() << endl;
        for (SHORT_INT_TYPE func_ind = 0; func_ind < total_basis_size_rank1[mu]; func_ind++, sequential_func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis_rank1[mu][func_ind].ndensity; p++, coeffs_ind++) {
                basis_rank1[mu][func_ind].coeff[p] = basis_coeffs_vector.at(coeffs_ind);
                //update also mu0_bbasis_vector for consistency
                mu0_bbasis_vector.at(mu).at(sequential_func_ind).coeff[p] = basis_coeffs_vector.at(coeffs_ind);
            }
        }
//        cout << "lab2.2: mu=" << mu << endl;
        for (SHORT_INT_TYPE func_ind = 0;
             func_ind < total_basis_size[mu]; func_ind++, sequential_func_ind++) {
            for (DENSITY_TYPE p = 0; p < basis[mu][func_ind].ndensity; p++, coeffs_ind++) {
                basis[mu][func_ind].coeff[p] = basis_coeffs_vector.at(coeffs_ind);
                //update also mu0_bbasis_vector for consistency
                mu0_bbasis_vector.at(mu).at(sequential_func_ind).coeff[p] = basis_coeffs_vector.at(coeffs_ind);
            }
        }
    }
}

void ACEBBasisSet::set_all_coeffs(const vector<DOUBLE_TYPE> &coeffs) {

    SPECIES_TYPE mu_i, mu_j;
    vector<DOUBLE_TYPE> crad_coeffs;

    size_t crad_size = 0;
    for (mu_i = 0; mu_i < nelements; mu_i++)
        for (mu_j = mu_i; mu_j < nelements; mu_j++) { // only upper-diag coefficients
            auto bond_pair = make_pair(mu_i, mu_j);
            if (map_bond_specifications.find(bond_pair) == map_bond_specifications.end())
                continue;
            const auto &bondSpec = map_bond_specifications.at(bond_pair);
            crad_size += bondSpec.nradbasemax * (bondSpec.lmax + 1) * bondSpec.nradmax;
        }

    if (crad_size > coeffs.size())
        throw invalid_argument("Number of expected radial function's parameters is more than given coefficients");
    vector<DOUBLE_TYPE> crad_flatten_vector(coeffs.begin(), coeffs.begin() + crad_size);
    set_crad_coeffs(crad_flatten_vector);

    vector<DOUBLE_TYPE> basis_coeffs_vector(coeffs.begin() + crad_size, coeffs.end());
    set_basis_coeffs(basis_coeffs_vector);
}

bool AuxiliaryData::empty() const {
    return int_data.empty() && int_arr_data.empty() &&
           double_data.empty() && double_arr_data.empty() &&
           string_data.empty() && string_arr_data.empty();
}

YAML_PACE::Node AuxiliaryData::to_YAML() const {
    YAML_PACE::Node node;

    node["_int"] = int_data;
    node["_int"].SetStyle(YAML_PACE::EmitterStyle::Flow);
    for (auto const &pair : int_arr_data) {
        node["_int_arr"][pair.first] = pair.second;
        node["_int_arr"][pair.first].SetStyle(YAML_PACE::EmitterStyle::Flow);
    }

    node["_double"] = double_data;
    node["_double"].SetStyle(YAML_PACE::EmitterStyle::Flow);
    for (auto const &pair : double_arr_data) {
        node["_double_arr"][pair.first] = pair.second;
        node["_double_arr"][pair.first].SetStyle(YAML_PACE::EmitterStyle::Flow);
    }

    node["_string"] = string_data;
    node["_string"].SetStyle(YAML_PACE::EmitterStyle::Flow);
    for (auto const &pair : string_arr_data) {
        node["_string_arr"][pair.first] = pair.second;
        node["_string_arr"][pair.first].SetStyle(YAML_PACE::EmitterStyle::Flow);
    }
    return node;
}

void AuxiliaryData::from_YAML(YAML_PACE::Node &node) {
    if (node["_int"])
        int_data = node["_int"].as<map<string, int>>();
    if (node["_int_arr"])
        int_arr_data = node["_int_arr"].as<map<string, vector<int>>>();

    if (node["_double"])
        double_data = node["_double"].as<map<string, double>>();
    if (node["_double_arr"])
        double_arr_data = node["_double_arr"].as<map<string, vector<double>>>();

    if (node["_string"])
        string_data = node["_string"].as<map<string, string>>();
    if (node["_string_arr"])
        string_arr_data = node["_string_arr"].as<map<string, vector<string>>>();
}
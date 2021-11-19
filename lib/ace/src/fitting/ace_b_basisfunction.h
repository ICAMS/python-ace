//
// Created by Yury Lysogorskiy on 28.02.20.
//

#ifndef ACE_B_BASISFUNCTION_H
#define ACE_B_BASISFUNCTION_H

#include <iostream>
#include <iomanip>
#include <cstring>

#include <vector>
#include <sstream>

#include "yaml-cpp/yaml.h"

#include "ace_c_basisfunction.h"
#include "ace_types.h"


using namespace std;

struct ACEBBasisFunction;

/**
A class for holding single B-basis function specification data.
*/
class BBasisFunctionSpecification {
public:
    RANK_TYPE rank;
    vector<string> elements;
    vector<NS_TYPE> ns;
    vector<LS_TYPE> ls;
    vector<LS_TYPE> LS;
    vector<DOUBLE_TYPE> coeffs; //coefficients for different densities
    bool skip = false; //skip this function in BBasis construction

    BBasisFunctionSpecification() = default;

    /**
     *
     * @param elements_mapping: [0->"Al", 1->"Cu",...]
     * @param func
     */
    BBasisFunctionSpecification(const vector<string> &elements_mapping, const ACEBBasisFunction &func);

    BBasisFunctionSpecification(const vector<string> &elements, const vector<NS_TYPE> &ns,
                                const vector<LS_TYPE> &ls,
                                const vector<LS_TYPE> &LS,
                                const vector<DOUBLE_TYPE> &coeffs);

    BBasisFunctionSpecification(const vector<string> &elements, const vector<NS_TYPE> &ns,
                                const vector<LS_TYPE> &ls,
                                const vector<DOUBLE_TYPE> &coeffs) : BBasisFunctionSpecification(elements, ns, ls, {},
                                                                                                 coeffs) {};

    BBasisFunctionSpecification(const vector<string> &elements, const vector<NS_TYPE> &ns,
                                const vector<DOUBLE_TYPE> &coeffs) : BBasisFunctionSpecification(elements, ns, {0}, {},
                                                                                                 coeffs) {};

    ~BBasisFunctionSpecification() = default;

    bool less_specification_than(const BBasisFunctionSpecification &another) const {
        if (rank < another.rank) return true;
        else if (elements < another.elements) return true;
        else if (ns < another.ns) return true;
        else if (ls < another.ls) return true;
        else if (LS < another.LS) return true;
        else return false;
    }

    bool great_specification_than(const BBasisFunctionSpecification &another) const {
        if (rank > another.rank) return true;
        else if (elements > another.elements) return true;
        else if (ns > another.ns) return true;
        else if (ls > another.ls) return true;
        else if (LS > another.LS) return true;
        else return false;
    }

    bool specification_equal_to(const BBasisFunctionSpecification &another) const {
        return (elements == another.elements) and (rank == another.rank) and (ns == another.ns) and
               (ls == another.ls) and (LS == another.LS);
    }

    bool operator<(const BBasisFunctionSpecification &another) const {
        return less_specification_than(another) and coeffs < another.coeffs;
    }

    bool operator>(const BBasisFunctionSpecification &another) const {
        return great_specification_than(another) and coeffs > another.coeffs;
    }

    bool operator==(const BBasisFunctionSpecification &another) const {
        return specification_equal_to(another) and (coeffs == another.coeffs);
    }


    string to_string() const;

    void validate();

    YAML_PACE::Node to_YAML() const;

    BBasisFunctionSpecification copy() const {
        BBasisFunctionSpecification new_func;
        new_func.rank = this->rank;
        new_func.elements = this->elements;
        new_func.ns = this->ns;
        new_func.ls = this->ls;
        new_func.LS = this->LS;
        new_func.coeffs = this->coeffs;
        new_func.skip = this->skip;
        return new_func;
    }
};


struct ACEBBasisFunction : public ACEAbstractBasisFunction {

// array of corresponding general Clebsch-Gordan coefficients for (m1, m2, ..., m_rank)_i ms-combination
    // size =  num_ms_combs
    // effective shape [num_ms_combs]
    DOUBLE_TYPE *gen_cgs = nullptr;

    // "LS" - indexes for intermediate coupling "L"
    // size  = rankL
    // effective shape [rankL]
    LS_TYPE *LS = nullptr;

    // coefficients of the atomic cluster expansion, size = ndensity, shape [ndensity]
    DOUBLE_TYPE *coeff = nullptr;

    RANK_TYPE rankL = 0; ///< number of intermediate coupling  \f$ L\f$, rankL=rank-2

    vector<int> sort_order;///< argsort, to preserve the original order before order_and_compress

    ACEBBasisFunction() = default;

    // Because the ACEBBasisFunction contains dynamically allocated arrays, the Rule of Three should be
    // fullfilled, i.e. copy constructor (copy the dynamic arrays), operator= (release previous arrays and
    // copy the new dynamic arrays) and destructor (release all dynamically allocated memory)

    //copy constructor
    ACEBBasisFunction(const ACEBBasisFunction &other) {
        _copy_from(other);
    }

    //operator=
    ACEBBasisFunction &operator=(const ACEBBasisFunction &other) {
        if (this != &other) {
            _clean();
            _copy_from(other);
        }
        return *this;
    }

    //destructor
    ~ACEBBasisFunction() {
        _clean();
    }

    explicit ACEBBasisFunction(BBasisFunctionSpecification &bBasisSpecification, bool is_half_basis = true,
                               bool compress = true);

    explicit ACEBBasisFunction(BBasisFunctionSpecification &bBasisSpecification,
                               const map<string, SPECIES_TYPE> &elements_to_index_map, bool is_half_basis = true,
                               bool compress = true);

    void _copy_from(const ACEBBasisFunction &other) {
        ACEAbstractBasisFunction::_copy_from(other);
        is_proxy = false;

        rankL = other.rankL;
        sort_order = other.sort_order;
        basis_mem_copy(other, LS, rankL, LS_TYPE)
        basis_mem_copy(other, gen_cgs, num_ms_combs, DOUBLE_TYPE)
        basis_mem_copy(other, coeff, ndensity, DOUBLE_TYPE)
    }

    void _clean() override {
        ACEAbstractBasisFunction::_clean();

        //release memory if the structure is not proxy
        if (!is_proxy) {
            delete[] LS;
            delete[] gen_cgs;
            delete[] coeff;
        }

        LS = nullptr;
        gen_cgs = nullptr;
        coeff = nullptr;
    }

    void print() const {
        cout << "ACEBBasisFunction: ndensity= " << (int) this->ndensity << ", mu0 = " << (int) this->mu0 << " mus = (";

        for (RANK_TYPE r = 0; r < this->rank; r++)
            cout << (int) this->mus[r] << " ";
        cout << "), ns=(";
        for (RANK_TYPE r = 0; r < this->rank; r++)
            cout << (int) this->ns[r] << " ";
        cout << "), ls=(";
        for (RANK_TYPE r = 0; r < this->rank; r++)
            cout << this->ls[r] << " ";
        cout << "), LS=(";
        for (RANK_TYPE r = 0; r < this->rankL; r++)
            cout << this->LS[r] << " ";
        cout << "), c=(";
        DENSITY_TYPE p;
        for (p = 0; p < this->ndensity - 1; ++p)
            cout << std::setprecision(20) << this->coeff[p] << ", ";
        cout << std::setprecision(20) << this->coeff[p] << ")";
        cout << " " << this->num_ms_combs << " m_s combinations: {" << endl;

        for (int i = 0; i < this->num_ms_combs; i++) {
            cout << "\t<";
            for (RANK_TYPE r = 0; r < this->rank; r++)
                cout << this->ms_combs[i * this->rank + r] << " ";
            cout << " >: " << this->gen_cgs[i] << endl;
        }
        if (this->is_proxy)
            cout << "proxy ";
        if (this->is_half_ms_basis)
            cout << "half_ms_basis";
        cout << " argsort=";
        for (auto x: sort_order)
            cout << " " << x;
        cout << "}" << endl;
    }

    void create_from_spec(BBasisFunctionSpecification &bBasisSpecification,
                          const map<string, SPECIES_TYPE> &elements_to_index_map,
                          bool is_half_basis, bool compress);
};


string B_basis_function_to_string(const ACEBBasisFunction &func);


#endif //ACE_B_BASISFUNCTION_H

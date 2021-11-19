//
// Created by Yury Lysogorskiy on 28.02.20.
//

#include "ace_b_basisfunction.h"

#include <algorithm>
#include <cstdio>
#include <sstream>


#include "ace_b_basis.h"
#include "ace_clebsch_gordan.h"
#include "ace_couplings.h"
#include "ace_utils.h"

ACEClebschGordan clebsch_gordan(10);

string B_basis_function_to_string(const ACEBBasisFunction &func) {
    stringstream sstream;
    char s[1024];
    sprintf(s, "ACEBBasisFunction: ndensity= %d, mu0 = %d mus = (", func.ndensity, func.mu0);
    sstream << s;
    cout << s;

    for (RANK_TYPE r = 0; r < func.rank; r++)
        sstream << func.mus[r] << " ";
    sstream << "), ns=(";
    cout << "), ns=(";

    for (RANK_TYPE r = 0; r < func.rank; r++)
        sstream << func.ns[r] << " ";
    sstream << "), ls=(";
    cout << "), ls=(";

    for (RANK_TYPE r = 0; r < func.rank; r++)
        sstream << func.ls[r] << " ";
    sstream << "), LS = (";
    cout << "), LS = (";

    for (RANK_TYPE r = 0; r < func.rankL; r++)
        sstream << func.LS[r] << " ";
    sstream << "), c=(";
    cout << "), c=(";

    DENSITY_TYPE p;
    for (p = 0; p < func.ndensity - 1; ++p)
        sstream << func.coeff[p] << ", ";
    sstream << func.coeff[p] << ")";
    sprintf(s, " %d m_s combinations: {\n", func.num_ms_combs);
    sstream << s;
    cout << s;

    for (int i = 0; i < func.num_ms_combs; i++) {
        sstream << "\t<";
        for (RANK_TYPE r = 0; r < func.rank; r++)
            sstream << func.ms_combs[i * func.rank + r] << " ";
        sstream << " >: " << func.gen_cgs[i] << "\n";
    }
    sstream << "}\n";
    return sstream.str();
}

ACEBBasisFunction::ACEBBasisFunction(BBasisFunctionSpecification &bBasisSpecification, bool is_half_basis,
                                     bool compress) {

    // from https://stackoverflow.com/questions/28100712/better-way-of-counting-unique-item
    auto elements = bBasisSpecification.elements;
    std::sort(elements.begin(), elements.end());
    int uniqueElements = std::unique(elements.begin(), elements.end()) - elements.begin();

    if (uniqueElements > 1)
        throw invalid_argument(
                "Couldn't create ACEBBasisFunction from BBasisFunctionSpecification with"
                "multiple elements without elements_to_index_map, please use another constructor");
    map<string, SPECIES_TYPE> elements_to_index_map;
    elements_to_index_map[elements[0]] = 0;
    create_from_spec(bBasisSpecification, elements_to_index_map, is_half_basis, compress);


}

void ACEBBasisFunction::create_from_spec(BBasisFunctionSpecification &bBasisSpecification,
                                         const map<string, SPECIES_TYPE> &elements_to_index_map,
                                         bool is_half_basis,
                                         bool compress) {
    RANK_TYPE rank = bBasisSpecification.rank;
    RANK_TYPE rankL = 0;
    const NS_TYPE *nr = &bBasisSpecification.ns[0];
    const LS_TYPE *ls = &bBasisSpecification.ls[0];
    const DOUBLE_TYPE *cs = &bBasisSpecification.coeffs[0];;
    LS_TYPE *LS;

    if (rank > 2) {
        LS = &(bBasisSpecification.LS[0]);
        rankL = rank - 2;
    } else
        LS = nullptr;

    LS_TYPE lmax = 0;
    for (RANK_TYPE r = 0; r < rank; r++)
        if (ls[r] > lmax)
            lmax = ls[r];
    for (RANK_TYPE r = 0; r < rank; r++)
        if (ls[r] > lmax)
            lmax = ls[r];

    clebsch_gordan.init(2 * lmax);

    try {
        generate_basis_function_n_body(rank, nr, ls, LS, *this,
                                       clebsch_gordan, is_half_basis);
    } catch (const invalid_argument &exc) {
        stringstream s;
        s << bBasisSpecification.to_string() << " could not be constructed: " << endl << exc.what();
        throw invalid_argument(s.str());
    }

    mu0 = elements_to_index_map.at(bBasisSpecification.elements[0]);
    for (RANK_TYPE r = 1; r <= rank; r++)
        mus[r - 1] = elements_to_index_map.at(bBasisSpecification.elements[r]);

    ndensity = bBasisSpecification.coeffs.size();
    coeff = new DOUBLE_TYPE[ndensity];
    for (DENSITY_TYPE p = 0; p < ndensity; ++p)
        coeff[p] = cs[p];
    if (compress)
        order_and_compress_b_basis_function(*this);


    if (num_ms_combs == 0) {
        stringstream ss;
        ss << "ls=[" << join(bBasisSpecification.ls, ",") << "], LS=[" << join(bBasisSpecification.LS, ",") << "]";
        throw invalid_argument("B-basis function specification is invalid: no valid ms-combinations for " + ss.str());
    }
}


ACEBBasisFunction::ACEBBasisFunction(BBasisFunctionSpecification &bBasisSpecification,
                                     const map<string, SPECIES_TYPE> &elements_to_index_map, bool is_half_basis,
                                     bool compress) {
    create_from_spec(bBasisSpecification, elements_to_index_map, is_half_basis, compress);

}

BBasisFunctionSpecification::BBasisFunctionSpecification(const vector<string> &elements, const vector<NS_TYPE> &ns,
                                                         const vector<LS_TYPE> &ls,
                                                         const vector<LS_TYPE> &LS,
                                                         const vector<DOUBLE_TYPE> &coeffs) : rank(elements.size() - 1),
                                                                                              elements(elements),
                                                                                              ns(ns),
                                                                                              ls(ls),
                                                                                              LS(LS),
                                                                                              coeffs(coeffs) {

    if (elements.size() - 1 != ns.size())
        throw invalid_argument("size of 'ns' should be by one less than size of 'elements'");
    validate();
}


string BBasisFunctionSpecification::to_string() const {
    stringstream s;
    s << "BBasisFunctionSpecification(elements=[" << join(elements, ",") << "],  " \
 << "ns=[" << join(ns, ",") << "],  "\
 << "ls=[" << join(ls, ",") << "],  ";
    if (!LS.empty())
        s << "LS=[" << join(LS, ",") << "],  ";
    s << "coeffs=[" << join(coeffs, ",") << "]";
    if(skip)
        s << " skipped";
    s << ")";
    return s.str();
}

void BBasisFunctionSpecification::validate() {
    this->rank = this->elements.size() - 1;
    expand_ls_LS(this->rank, this->ls, this->LS);

    if (ns.size() != ls.size()) {
        throw invalid_argument("'ls' should have the same length as 'ns'");
    }
    //min of ns >0
    NS_TYPE ns_min = ns.at(0);
    for (auto n:ns) if (n < ns_min) ns_min = n;
    if (ns_min < 1) {
        stringstream s;
        s << this->to_string() << ":" << endl;
        s << "minimum value of 'ns'(" << ns_min << ") should be not less than 1";
        throw invalid_argument(s.str());
    }

    //min of ls >0
    LS_TYPE ls_min = ls.at(0);
    for (auto l:ls) if (l < ls_min) ls_min = l;
    if (ls_min < 0) {
        stringstream s;
        s << this->to_string() << ":" << endl;
        s << "minimum value of 'ls'(" << ls_min << ") should be not less than 0";
        throw invalid_argument(s.str());
    }


    //rank
    if (this->rank != this->ns.size()) {
        stringstream s;
        s << this->to_string() << ":" << endl;
        s << "size of 'ns'(" << ns.size() << ") is inconsistent with the rank(" << rank
          << ") and size of 'elements' - 1 (" << elements.size() - 1 << ")";
        throw invalid_argument(s.str());
    }
    if (this->ls.size() != this->rank) {
        stringstream s;
        s << this->to_string() << ":" << endl;
        s << "size of 'ls'(" << ls.size() << ") is inconsistent with the rank(" << rank
          << ") and size of 'elements' - 1 (" << elements.size() - 1 << ")";
        throw invalid_argument(s.str());
    }
    try {
        validate_ls_LS(this->ls, this->LS);
    } catch (const invalid_argument &exc) {
        stringstream s;
        s << this->to_string() << ":" << endl;
        s << exc.what();
        throw invalid_argument(s.str());
    }
}

BBasisFunctionSpecification::BBasisFunctionSpecification(const vector<string> &elements_mapping,
                                                         const ACEBBasisFunction &func) {
    this->rank = func.rank;
    vector<string> elements(rank + 1);
    elements[0] = elements_mapping[func.mu0];
    for (RANK_TYPE r = 0; r < func.rank; r++) {
        if (func.sort_order.empty())
            elements[r + 1] = elements_mapping[func.mus[r]];
        else
            elements[func.sort_order[r] + 1] = elements_mapping[func.mus[func.sort_order[r]]];
    }
    this->elements = elements;

    this->ns = vector<NS_TYPE>(func.rank);
    this->ls = vector<LS_TYPE>(func.rank);
    for (RANK_TYPE r = 0; r < func.rank; r++) {
        if (func.sort_order.empty()) {
            this->ns[r] = func.ns[r];
            this->ls[r] = func.ls[r];
        } else {
            this->ns[func.sort_order[r]] = func.ns[r];
            this->ls[func.sort_order[r]] = func.ls[r];
        }
    }

    this->LS = vector<LS_TYPE>(func.rankL);
    for (RANK_TYPE r = 0; r < func.rankL; r++) {
        this->LS[r] = func.LS[r];
    }

    this->coeffs = vector<DOUBLE_TYPE>(func.ndensity);
    for (DENSITY_TYPE p = 0; p < func.ndensity; p++)
        this->coeffs[p] = func.coeff[p];


}

YAML_PACE::Node BBasisFunctionSpecification::to_YAML() const {
    YAML_PACE::Node spec_yaml;
    spec_yaml.SetStyle(YAML_PACE::EmitterStyle::Flow);
    spec_yaml["type"] = join(this->elements, " ");
    spec_yaml["nr"] = ns;
    spec_yaml["nl"] = ls;
    if (!LS.empty())
        spec_yaml["lint"] = LS;
    spec_yaml["c"] = coeffs;
    if (skip)
        spec_yaml["skip"] = true;

    return spec_yaml;
}


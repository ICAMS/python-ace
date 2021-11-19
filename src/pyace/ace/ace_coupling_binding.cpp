#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iterator>
#include "ace_c_basisfunction.h"
#include "ace_couplings.h"
#include "ace_utils.h"

namespace py = pybind11;
using namespace std;

ACEClebschGordan cs(12);
vector<LS_TYPE> emptyLS(0);

list<ms_cg_pair> generate_ms_cg_list_wrapper(const vector<LS_TYPE>& ls,
                                            const vector<LS_TYPE>& LS,
                                            const bool half_basis=true) {
    list<ms_cg_pair> empty_res;
    list<ms_cg_pair> ms_cs_pairs_list;

    const RANK_TYPE rank = ls.size();

    RANK_TYPE rankL = 0;
    RANK_TYPE rankLind = 0;
    if (rank > 2) {
        rankL = rank - 2;
        rankLind = rankL-1;
    }

    vector<LS_TYPE> newLS;
    if(rankL!=LS.size()) {
        if (rank==1 || rank==2 ) //LS is empty
            newLS.resize(0);
        else if (rank==3) { // LS has 1 element, 0 independent: LS[-1]==ls[-1]
            newLS.resize(rankL);
            newLS[0]=ls[rank-1];
        } else if (rank==4) { // LS has 2 element, 1 independent: LS[-1]==LS[-2]
            if(LS.size()==rankLind) { // independent value is given
                newLS.resize(rankL);
                newLS[0]=newLS[1]=LS[0];
            } else
                return empty_res;
        } else if (rank==5) { // LS has 3 elements, 2 independent: LS[-1] = ls[-1]
            if(LS.size()==rankLind) {// independent value is given
                newLS.resize(rankL);
                for(RANK_TYPE r = 0; r<rankL-1; r++)
                    newLS[r]=LS[r];
                newLS[2] = ls[rank-1];
            } else
                return empty_res;
        } else if (rank>=6) {// LS[-1]==LS[-2]
            if(LS.size()==rankLind) {// independent value is given ( r=6, rL=rank-2=4, rLind=rank-3=3
                newLS.resize(rankL);
                for(RANK_TYPE r = 0; r<rankLind; r++)
                    newLS[r]=LS[r];
                //newLS
                newLS[rankL-1] = newLS[rankL-2]; // two last elements of `LS` are equal
            } else
                return empty_res;
        } else
            return empty_res;
    } else
        newLS = LS;

    LS_TYPE Lmax=0;
    for(int i=0;i<rank;i++) if(ls[i]>Lmax)  Lmax = ls[i];
    for(int i=0;i<rankL;i++) if(newLS[i]>Lmax)  Lmax = newLS[i];

    if(cs.lmax<Lmax)
        cs.init(Lmax);

    int res = generate_ms_cg_list(rank, ls.data(), newLS.data(), half_basis,
            cs, ms_cs_pairs_list);

    return ms_cs_pairs_list;
}

string ms_cg_pair_repr(ms_cg_pair& a) {
    stringstream s;
    s <<"<ms=[" << join(a.ms, ",") << "] : gen_cg=" << a.c <<">";
    return s.str();
}

bool ms_cg_pair_eq(ms_cg_pair& a,ms_cg_pair& b) {
    return (a.ms==b.ms) & (a.c==b.c);
}

py::tuple expand_ls_LS_wrapper(int rank, vector<LS_TYPE> ls, vector<LS_TYPE> LS){
    expand_ls_LS(rank, ls, LS);
    return py::make_tuple(ls, LS);
}

bool is_valid_ls_LS(vector<LS_TYPE> ls, vector<LS_TYPE> LS){
    try {
        validate_ls_LS(ls, LS);
        return true;
    } catch (std::invalid_argument) {}
    return false;
}

bool true_func(int rank, vector<LS_TYPE> ls) {
    return true;
}

PYBIND11_MODULE(coupling, m) {
    py::options options;
//    options.disable_function_signatures();

    m.def("clebsch_gordan", [](LS_TYPE j1, MS_TYPE m1, LS_TYPE j2, MS_TYPE m2, LS_TYPE J, MS_TYPE M) {
        return cs.clebsch_gordan(j1, m1, j2, m2, J, M);
    }, "clebsch_gordan(j1, m1, j2, m2,  J, M):float ");

    m.def("anotherClebschGordan", &anotherClebschGordan, "anotherClebschGordan(j1, m1, j2, m2,  J, M):float ");

    m.def("expand_ls_LS", &expand_ls_LS_wrapper);
//    m.def("expand_ls_LS_wrapper", &true_func);

    m.def("validate_ls_LS", &validate_ls_LS, "validate_ls_LS(ls:List[int], LS:List[int]) ");
    m.def("is_valid_ls_LS", &is_valid_ls_LS, "is_valid_ls_LS(ls:List[int], LS:List[int]): boolean ");

    py::class_<ACECouplingTree>(m, "ACECouplingTree", R"mydelimiter(
        Wrapper class for M-coupling tree construction
)mydelimiter")
            .def(py::init<int>(), py::arg("rank") = 2)
            .def_readonly("tree_indices_array", &ACECouplingTree::tree_indices_array);

py::class_<ms_cg_pair>(m, "MsCgPair", R"mydelimiter(
        pair of ms-combination and corresponding generalized Clebsch-Gordan coefficient
)mydelimiter")
            .def(py::init <>())
            .def_readonly("ms", &ms_cg_pair::ms)
            .def_readonly("gen_cg",&ms_cg_pair::c)
            .def("__repr__", &ms_cg_pair_repr)
            .def("__eq__", &ms_cg_pair_eq)
            ;

m.def("generate_ms_cg_list",&generate_ms_cg_list_wrapper, R"mydelimiter(

        Generate list of ms with corresponding generalized Clebsch-Gordan coefficients

        Parameters
        ----------
        ls : List[int]
            list of ls
        LS : List[Int] (default = [])
            list of LS (or lint)
        half_basis: boolean (default =True)
            whether generate only non-negative
            combinations of ms-vector, i.e. those
            that has first positive non-zero value
        Returns
        -------
            List[ms_cg_pair]
)mydelimiter", py::arg("ls"), py::arg("LS")=emptyLS, py::arg("half_basis")=true);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

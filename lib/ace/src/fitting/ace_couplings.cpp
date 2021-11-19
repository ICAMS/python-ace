#include "ace_utils.h"
#include "ace_b_basisfunction.h"
#include <algorithm>
#include <cstdio>
#include "ace_couplings.h"

#include <cmath>
#include <iostream>
#include <list>
#include <queue>
#include <sstream>

void expand_ls_LS(RANK_TYPE rank, vector<LS_TYPE> &ls, vector<LS_TYPE> &LS);

using namespace std;

//generate coupling tree as a
vector<SHORT_INT_TYPE> generate_coupling_tree(RANK_TYPE r) {
    vector<SHORT_INT_TYPE> tree_indices_array;

    if (r < 2) return tree_indices_array;
    tree_indices_array.resize(3 * (r - 1));

    SHORT_INT_TYPE i;
    SHORT_INT_TYPE ms[r];

    queue<SHORT_INT_TYPE> Tree_indices_queue;
    queue<SHORT_INT_TYPE> ms_queue;
    queue<SHORT_INT_TYPE> MS_queue;

    for (i = 0; i < r; i++) {
        ms[i] = i;
        ms_queue.push(ms[i]);
    }


    RANK_TYPE MS[r - 2];
    for (i = 0; i < r - 2; i++) {
        MS[i] = i + r;
        MS_queue.push(MS[i]);
    }

    queue<RANK_TYPE> interm_queue;
    short int i1, i2, i_int = 0;

    for (i = 0; i < r / 2; i++) {
        i1 = ms_queue.front();
        ms_queue.pop();

        i2 = ms_queue.front();
        ms_queue.pop();


        if (!MS_queue.empty()) {
            i_int = MS_queue.front();
            MS_queue.pop();
        } else {
            i_int = -1;
        }
#ifdef DEBUG_COUPLING
        cout<<"("<<i1<<","<<i2<<")"<<i_int<<endl;
#endif
        Tree_indices_queue.push(i1);
        Tree_indices_queue.push(i2);
        Tree_indices_queue.push(i_int);
        interm_queue.push(i_int);
    }


    while (interm_queue.size() > 1 & i_int != -1) {
        i1 = interm_queue.front();
        interm_queue.pop();

        i2 = interm_queue.front();
        interm_queue.pop();
#ifdef DEBUG_COUPLING
        cout<<"interm_queue.size()="<<interm_queue.size()<<endl;
#endif
        if (!MS_queue.empty()) {
            i_int = MS_queue.front();
            MS_queue.pop();
        } else {
            i_int = -1;
        }

#ifdef DEBUG_COUPLING
        cout<<"MS_queue.size()="<<MS_queue.size()<<endl;
        cout<<"MS_queue.empty()="<<MS_queue.empty()<<endl;

        cout<<"("<<i1<<","<<i2<<")"<<i_int<<endl;
#endif
        Tree_indices_queue.push(i1);
        Tree_indices_queue.push(i2);
        Tree_indices_queue.push(i_int);

        interm_queue.push(i_int);

    }

    if (i_int != -1) {
        //last node of tree
#ifdef DEBUG_COUPLING
        cout<<"last node"<<endl;
#endif
        i1 = interm_queue.front();
        interm_queue.pop();

        i2 = ms_queue.front();
        ms_queue.pop();

        i_int = -1;
        //MS_queue.pop();
#ifdef DEBUG_COUPLING
        cout<<"("<<i1<<","<<i2<<")"<<i_int<<endl;
#endif
        Tree_indices_queue.push(i1);
        Tree_indices_queue.push(i2);
        Tree_indices_queue.push(i_int);

        interm_queue.push(i_int);

    }
#ifdef DEBUG_COUPLING
    cout<<"main tree indcies, size="<<Tree_indices_queue.size()<<": ";
#endif
    if (Tree_indices_queue.size() != 3 * (r - 1)) {
        stringstream s;
        s << "Unable to build coupling tree for rank = " << (int) r << ", expected size = " << 3 * (r - 1)
          << " actual size = " << Tree_indices_queue.size();
        cerr << "Exception:" << s.str() << endl;
        throw invalid_argument(s.str());
//        exit(EXIT_FAILURE);
    }

    i = 0;
    while (!Tree_indices_queue.empty()) {
        tree_indices_array[i] = Tree_indices_queue.front();
#ifdef DEBUG_COUPLING
        cout<<tree_indices_array[i]<<" ";
#endif
        Tree_indices_queue.pop();
        i++;
    }
#ifdef DEBUG_COUPLING
    cout << endl;
#endif
    return tree_indices_array;
}

//compute the sign of the given ms combination as the sign of the first non-zero element
// if all elements are zero, sign = 0
int get_ms_sign(const RANK_TYPE rank, const MS_TYPE *ms) {
    int sign = 0;
    for (SHORT_INT_TYPE j = 0; j < rank; ++j)
        if (ms[j] < 0) {
            sign = -1;
            break;
        } else if (ms[j] > 0) {
            sign = +1;
            break;
        }
    return sign;
}

// by the given rank and corresponding array of ls,
// generate the ms_strides array (size=rank) and the total size of the ms-space
unsigned long get_ms_space_size_and_strides(const RANK_TYPE rank, const LS_TYPE *ls, unsigned long *ms_strides) {
    ms_strides[0] = 1;
    unsigned long int m_space_size = (2 * ls[0] + 1);
    for (RANK_TYPE i = 1; i < rank; i++) {
        ms_strides[i] = ms_strides[i - 1] * (2 * ls[i - 1] + 1);
        m_space_size *= (2 * ls[i] + 1);
    }
    return m_space_size;
}

// in:   rank, current enumeration index in ms-space (ms_space_ind), ls-array and ms_strides array
// out:  reconstructed array of ms (from the ms-space)
void unpack_ms_space_point(const RANK_TYPE rank, unsigned long ms_space_ind, const LS_TYPE *ls,
                           const unsigned long *ms_strides, MS_TYPE *ms) {
    unsigned long int ms_space_point = ms_space_ind;
    for (SHORT_INT_TYPE j = rank - 1; j >= 0; j--) {
        ms[j] = ms_space_point / ms_strides[j] - ls[j];
        ms_space_point = ms_space_point % ms_strides[j];
#ifdef DEBUG_COUPLING
        cout<<"ms["<<j<<"]="<<ms[j]<<" ";
#endif
    }
}

int generate_ms_cg_list(const RANK_TYPE rank, const LS_TYPE *ls, const LS_TYPE *LS, const bool half_basis,
                        const ACEClebschGordan &cs, list<ms_cg_pair> &ms_cs_pairs_list) noexcept(false) {
    RANK_TYPE rankL = 0;
    if (rank > 2)
        rankL = rank - 2;
    LS_TYPE lsLS[2 * rank - 1];
    LS_TYPE sum_of_ls = 0;
    for (RANK_TYPE i = 0; i < rank; i++) {
        lsLS[i] = ls[i];
        sum_of_ls += ls[i];
    }
    if (sum_of_ls % 2 != 0) {
        stringstream s;
        s << "sum of ls is not even ";
        s << "ls = (";
        for (RANK_TYPE i = 0; i < rank; i++) s << ls[i] << " ";
        s << ")";
        throw invalid_argument(s.str());
    }

    for (RANK_TYPE i = rank; i < 2 * rank - 2; i++) {
        lsLS[i] = LS[i - rank];
        sum_of_ls += lsLS[i];
    }
    lsLS[2 * rank - 2] = 0; // last L = 0

    auto tree_indices_array = generate_coupling_tree(rank);

#ifdef DEBUG_COUPLING
    for (int i = 0; i < 2 * rank - 1; i++) {
        cout << "lsLS[" << i << "]=" << lsLS[i] << endl;
    }
    cout<<"Loop over {m}_space"<<endl;
#endif
    // get the ms-space strides and ms-space size
    unsigned long int ms_strides[rank];
    unsigned long ms_space_size = get_ms_space_size_and_strides(rank, ls, ms_strides);

    int sum_of_ms;
    int sign;
    MS_TYPE ms[rank];
    bool invalid_ms_combination = false; // flag to mark the current ms-combination as invalid
    //  LOOP OVER M-SPACE (all (m1,m2,..mr) combinations, accumulate the correct ms-combinations and corresponding
    // generalized Clebsh-Gordan coefficients into "ms_cs_pairs_list"
    for (unsigned long int ms_space_ind = 0; ms_space_ind < ms_space_size; ms_space_ind++) {
        invalid_ms_combination = false;
        // "unpack" the enumeration index of ms-point into vector of ms in  ms-space
        unpack_ms_space_point(rank, ms_space_ind, ls, ms_strides, ms);

        //compute the sum of the ms
        sum_of_ms = 0;
        for (SHORT_INT_TYPE j = 0; j < rank; j++)
            sum_of_ms += ms[j];

        // if sum not equal to 0, then the combination is not rotational-invariant, skip it
        if (sum_of_ms != 0) {
#ifdef DEBUG_COUPLING
            cout<<"sum_of_ms!=0, skip"<<endl;
#endif
            continue;
        }
        sign = get_ms_sign(rank, ms);

        if (half_basis & (sign < 0)) {
            //cout<<"Negative m[0]"<<endl;
            continue;
        }
#ifdef DEBUG_COUPLING
        cout<<endl;
#endif
        MS_TYPE msMS[2 * rank - 1]; //2*rank-2 + 1 =2*rank-1;  +1 for M_last==0
        //copy the ms vector to the first part of joint msMS vector
        for (RANK_TYPE i_mM = 0; i_mM < rank; i_mM++)
            msMS[i_mM] = ms[i_mM];

        // product accumulator for gen. Clebsh-Gordan
        DOUBLE_TYPE gen_Clebsch_Gordan_coef = 1.0;
        MS_TYPE i1, i2, i_coupled;

#ifdef DEBUG_COUPLING
        cout<<"loop over tree"<<endl;
#endif
        //Loop over M-tree: reconstruct the the MS-part values by using the rules from M-tree and ms-values
        for (RANK_TYPE triple_ind = 0; triple_ind < rank - 1; triple_ind++) {
            i1 = tree_indices_array[triple_ind * 3 + 0];
            i2 = tree_indices_array[triple_ind * 3 + 1];
            i_coupled = tree_indices_array[triple_ind * 3 + 2];
            //index "-1" stands for the last node of the tree
            if (i_coupled == -1) i_coupled = 2 * rank - 2;

            //apply tree coupling node summation rule in order to compute next (coupled) M value
            msMS[i_coupled] = msMS[i1] + msMS[i2];
#ifdef DEBUG_COUPLING
            cout<<"Tree triplet:"<<endl<<" m("<<i_coupled<<") = m("<<i1<<")+m("<<i2<<")"<<endl;
            cout<<msMS[i_coupled]<<" = "<<msMS[i1]<<"+"<<msMS[i2]<<endl;
            cout<<"i_coupled ="<<i_coupled <<endl;
#endif
            //check if the current node of M-tree and its parents have valid m-values
            if (abs(msMS[i1]) > lsLS[i1] || abs(msMS[i2]) > lsLS[i2] || abs(msMS[i_coupled]) > lsLS[i_coupled]) {
                invalid_ms_combination = true; // mark combination as invalid
                break; // stop traversing the M-tree
            }

            // get the Clebsh-Gordan coefficient for new coupling (m[i1], m[i2] | m[i_coupled])
            DOUBLE_TYPE cg_value = cs.clebsch_gordan(lsLS[i1], msMS[i1], lsLS[i2], msMS[i2], lsLS[i_coupled],
                                                     msMS[i_coupled]);
#ifdef DEBUG_COUPLING
            printf("C_MS(%d, %d|%d)_LS(%d, %d|%d) = %f\n",
                    msMS[i1], msMS[i2], msMS[i_coupled],lsLS[i1],lsLS[i2], lsLS[i_coupled],cg_value);
#endif
            // accumulate the Clebsh-Gordan coefficient into joint product over the tree's nodes.
            gen_Clebsch_Gordan_coef *= cg_value;

            // if coefficient became zero, no need to continue
            if (gen_Clebsch_Gordan_coef == 0.0) {
#ifdef DEBUG_COUPLING
                printf("gen_Clebsch_Gordan_coef = 0, could stop this loop\n");
#endif
                break; // stop traversing the M-tree
            }

        } // end loop over M-tree
        if (invalid_ms_combination) continue; // if msMS-combination is invalid, go to next one
        if (gen_Clebsch_Gordan_coef == 0.0) continue; // if Clebsh-Gordan coefficient is zero, go to next one

        // now, the ms-MS combination is built and the associated generalized Clebsh-Gordan coefficient is accumulated
#ifdef DEBUG_COUPLING
        cout << "msMS= ";
        for (int i_mM = 0; i_mM < 2 * rank - 1; i_mM++)
            cout << msMS[i_mM] << " ";
        cout << "gen_Clebsch_Gordan_coef=" << gen_Clebsch_Gordan_coef << endl;
#endif

        //prepare (msMS vector; generalized Clebsch-Gordan coefficient) pair
        ms_cg_pair mcs_pair{};
        mcs_pair.ms.resize(rank);
        //fill-in ms
        for (RANK_TYPE rr = 0; rr < rank; rr++)
            mcs_pair.ms[rr] = ms[rr];

        //if "half-basis" option is switched off and sign is positive,
        // then double the coefficient, in order to take into account
        // the anti-"symmetric" negative combination
        // with exactly the same coefficient
        if (half_basis && sign > 0)
            gen_Clebsch_Gordan_coef *= 2;

        mcs_pair.c = gen_Clebsch_Gordan_coef;
        mcs_pair.sign = sign;
        ms_cs_pairs_list.push_back(mcs_pair);
    }// END LOOP OVER M-SPACE
    return 0;
}

int generate_basis_function_n_body(const RANK_TYPE rank, const NS_TYPE *ns_rad, const LS_TYPE *ls, const LS_TYPE *LS,
                                   ACEBBasisFunction &b_basis_function, const ACEClebschGordan &cs,
                                   const bool half_basis) noexcept (false){
#ifdef DEBUG_COUPLING
    cout<<"generate_basis_function_n_body"<<endl;
    for(int i = 0; i<rank; i++)
        cout<<ls[i]<<" ";
    cout<<endl;
#endif

    RANK_TYPE rankL = 0;
    if (rank > 2)
        rankL = rank - 2; // rankL should be -2, as we count even the last node in the tree, where L = 0

    list<ms_cg_pair> ms_cs_pairs_list;
    generate_ms_cg_list(rank, ls, LS, half_basis, cs, ms_cs_pairs_list);

#ifdef DEBUG_COUPLING
    cout<<endl <<ms_cs_pairs_list.size() << " (ms,cs) pairs in a list"<<endl;
#endif

    auto pairs_count = static_cast<SHORT_INT_TYPE>(ms_cs_pairs_list.size());
    int i;

    b_basis_function.rank = rank;
    b_basis_function.mus = new SPECIES_TYPE[rank]{0};
    b_basis_function.rankL = rankL;
    b_basis_function.ls = new LS_TYPE[rank];
    for (RANK_TYPE j = 0; j < rank; j++)
        b_basis_function.ls[j] = ls[j];

    b_basis_function.ns = new NS_TYPE[rank];
    for (RANK_TYPE j = 0; j < rank; j++)
        b_basis_function.ns[j] = ns_rad[j];

    b_basis_function.LS = new LS_TYPE[rankL];
    for (RANK_TYPE j = 0; j < rankL; j++)
        b_basis_function.LS[j] = LS[j];

    b_basis_function.is_half_ms_basis = half_basis;
    if (ms_cs_pairs_list.empty()) {
        return -1;
    }
    b_basis_function.num_ms_combs = pairs_count;

    b_basis_function.ms_combs = new MS_TYPE[rank * b_basis_function.num_ms_combs];
    b_basis_function.gen_cgs = new DOUBLE_TYPE[b_basis_function.num_ms_combs];

    list<ms_cg_pair>::iterator ms_cg_pair_iterator;

    for (ms_cg_pair_iterator = ms_cs_pairs_list.begin(), i = 0;
         ms_cg_pair_iterator != ms_cs_pairs_list.end(); ++ms_cg_pair_iterator, i++) {
        for (RANK_TYPE r = 0; r < rank; r++)
            b_basis_function.ms_combs[i * rank + r] = (*ms_cg_pair_iterator).ms[r];

        b_basis_function.gen_cgs[i] = (*ms_cg_pair_iterator).c;

    }
#ifdef DEBUG_COUPLING
    printf("rank=%d, rankL = %d, num_ms_combs=%d\n", b_basis_function.rank, b_basis_function.rankL, b_basis_function.num_ms_combs);
#endif

    return 1;
}


ACECouplingTreesCache::ACECouplingTreesCache(const RANK_TYPE rank_max) {
    this->rank_max = rank_max;
    coupling_trees_vector.resize(rank_max + 1);
    for (RANK_TYPE r = 1; r <= rank_max; r++) {
        coupling_trees_vector[r] = ACECouplingTree(r);
    }
}

ACECouplingTree::ACECouplingTree(RANK_TYPE rank) {
    if (rank > 0) {
        this->rank = rank;
        this->tree_map_size = 3 * (rank - 1);
        tree_indices_array.resize(this->tree_map_size);
        initialize_coupling_tree();
    } else if (rank == 0) {
        this->rank = rank;
        this->tree_map_size = 0;
        tree_indices_array.resize(this->tree_map_size);
    }
}

void ACECouplingTree::initialize_coupling_tree() {
    this->tree_indices_array = generate_coupling_tree(rank);
}

bool validate_ls_LS(vector<LS_TYPE> ls, vector<LS_TYPE> LS) {
    int rank = ls.size();
    int rankL = LS.size();
    if (rank <= 2) {
        if (rankL != 0) {
            stringstream s;
            s << "len of LS should be " << 0 << ", but " << rankL << " is found";
            throw std::invalid_argument(s.str());
            //return false;
        }
    } else if (rank > 2)
        if (rankL != rank - 2) {
            stringstream s;
            s << "len of LS should be " << rank - 2 << ", but " << rankL << " is found";
            throw std::invalid_argument(s.str());
            //return false;
        }

    //validation according to ls-LS relations
    if(rank==1) {
        if (ls.at(0)!=0) { //ls[0]==0
            stringstream s;
            s << "ls(";
            for (auto l_val: ls) s << " " << l_val << " ";
            s << ") should be (0)";
            throw std::invalid_argument(s.str());
        }
    } else if (rank==2) {// ls[1]==ls[0]
        if (ls.at(0)!=ls.at(1)) {
            stringstream s;
            s << "All elements of ls (";
            for (auto l_val: ls) s << " " << l_val << " ";
            s << ") should be equal";
            throw std::invalid_argument(s.str());
        }
    } else if (rank == 3 || rank == 5) { //L(-1) = l(-1)
        if (LS[LS.size() - 1] != ls[ls.size() - 1]) {
            stringstream s;
            s << "Last element of LS (";
            for (auto lint_val: LS) s << " " << lint_val << " ";
            s << ") != last element of ls (";
            for (auto ls_val: ls) s << " " << ls_val << " ";
            s << ")";
            throw std::invalid_argument(s.str());
            //return false;
        }
    } else if (rank >= 4) {//L(-1) = L(-2)
        if (LS[LS.size() - 1] != LS[LS.size() - 2]) {
            stringstream s;
            s << "Last element of LS (";
            for (auto lint_val: LS) s << " " << lint_val << " ";
            s << ") != to its next-to-last element";
            throw std::invalid_argument(s.str());
            //return false;
        }
    }

    int sum_of_ls = 0;
    for (auto l: ls)
        sum_of_ls += l;

    if (sum_of_ls % 2 != 0) {
        stringstream s;
        s << "Sum of ls is not even: ";
        s << "ls = (";
        for (RANK_TYPE i = 0; i < rank; i++) s << ls[i] << " ";
        s << ")";
        throw std::invalid_argument(s.str());
        //return false;
    }

    return true;
}

void expand_ls_LS(RANK_TYPE rank, vector<LS_TYPE> &ls, vector<LS_TYPE> &LS)  {// expand ls and LS from symmetry relations
    if (rank==1) { //ls[0]==0
        if(ls.empty())
            ls.emplace_back(0);
    } else if (rank==2) { // ls[1]==ls[0]
        if(ls.size() == 1)
            ls.emplace_back(ls.at(0));
    } else if (rank==3) { //       //L(-1) = l(-1)
        if(LS.empty() and ls.size() == rank)
            LS.emplace_back(ls.at(2)); // LS[0] ==LS[-1] == ls[2]==ls[-1]
    } else if (rank==4)  {
        if(LS.size() == 1)
            LS.emplace_back(LS.at(0)); // LS[1]==LS[0]
    } else if (rank==5) {
        if(LS.size() == rank - 3  and ls.size() == rank)
            LS.emplace_back(ls.at(4)); // LS[0] ==LS[-1] == ls[2]==ls[-1]
    } else if (rank==6) { // LS[-1]==LS[-2]
        if(LS.size() == rank - 3 and ls.size() == rank)
            LS.emplace_back(LS.at(LS.size() - 1));
    }
}
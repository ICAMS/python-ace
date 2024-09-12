//
// Created by Yury Lysogorskiy on 24.06.24
//

#ifndef GRACE_FS_EVALUATOR_H
#define GRACE_FS_EVALUATOR_H

#include "ace-evaluator/ace_arraynd.h"
#include "ace-evaluator/ace_array2dlm.h"
#include "ace-evaluator/ace_complex.h"
#include "ace-evaluator/ace_timing.h"
#include "ace-evaluator/ace_types.h"
#include "ace-evaluator/ace_radial.h"
#include "ace-evaluator/ace_spherical_cart.h"

#include <map>

#include "yaml-cpp/yaml.h"

struct GRACEFSBasisFunction {
    SPECIES_TYPE mu0 = 0;

    RANK_TYPE rank = 0;
    DENSITY_TYPE ndensity = 0;
    short num_ms_combs = 0;

    NS_TYPE ns;
    Array1D<LS_TYPE> ls; // [rank]
    Array1D<MS_TYPE> ms_combs; // size = num_ms_combs * rank, effective shape: [num_ms_combs][rank]
    Array1D<DOUBLE_TYPE> gen_cgs; // [num_ms_combs]
    Array1D<DOUBLE_TYPE> coeff; // [ndensity]

    void from_YAML(YAML_PACE::Node);
    void print() const;

};

struct GRACEFSEmbeddingSpecification {
    DENSITY_TYPE ndensity;
    vector<DOUBLE_TYPE> FS_parameters; ///< parameters for cluster functional, see Eq.(3) in implementation notes or Eq.(53) in <A HREF="https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104">  PRB 99, 014104 (2019) </A>
    string type = "FinnisSinclairShiftedScaled"; ///< FS and embedding function combination

    void from_YAML(YAML_PACE::Node emb_yaml);
};

class GRACEFSRadialFunction {
public:
    SPECIES_TYPE nelemets;

    NS_TYPE nradmax;
    LS_TYPE lmax;
    NS_TYPE nradbasemax;

    string radbasename;

    DOUBLE_TYPE rad_lamba;
    DOUBLE_TYPE deltaSplineBins = 0.001;
    Array3D<DOUBLE_TYPE> crad;///< crad_nlk order: [n=0..nradmax-1][l=0..lmax][k=0..nradbase-1]
    Array2D<DOUBLE_TYPE> Z; // chemical embedding Z [nelements][nradmax]

    vector<int> crad_shape;
    vector<int> Z_shape;

    DOUBLE_TYPE rcut;
    DOUBLE_TYPE dcut;

    DOUBLE_TYPE rcut_in = 0;
    DOUBLE_TYPE dcut_in = 0;

    void from_YAML(YAML_PACE::Node bond_yaml);

    // Arrays for look-up tables.
    SplineInterpolator splines_gk;
    SplineInterpolator splines_rnl;

    /**
   Arrays to store radial functions.
   */
    // g_k(r) functions, shape: [nradbase]
    Array1D<DOUBLE_TYPE> gr = Array1D<DOUBLE_TYPE>("gr");
    // derivatives of g_k(r) functions, shape: [nradbase]
    Array1D<DOUBLE_TYPE> dgr = Array1D<DOUBLE_TYPE>("dgr");

    // R_nl(r) functions, shape: [nradial][lmax+1]
    Array2D<DOUBLE_TYPE> fr = Array2D<DOUBLE_TYPE>("fr");
    // derivatives of R_nl(r) functions, shape: [nradial][lmax+1]
    Array2D<DOUBLE_TYPE> dfr = Array2D<DOUBLE_TYPE>("dfr");

    void init();

    void setuplookupRadspline();

    void evaluate(DOUBLE_TYPE r, SPECIES_TYPE mu_i, SPECIES_TYPE mu_j);

    void simplified_bessel(DOUBLE_TYPE rc, DOUBLE_TYPE x);

    void radbase(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, string radbasename, DOUBLE_TYPE r,
                 DOUBLE_TYPE cut_in, DOUBLE_TYPE dcut_in);

    void all_radfunc(DOUBLE_TYPE r);

    void radfunc();
};

class GRACEFSBasisSet {
public:
    SPECIES_TYPE nelements = 0;        ///< number of elements in basis set
    RANK_TYPE rankmax = 0;             ///< maximum value of rank
    DENSITY_TYPE ndensitymax = 0;      ///< maximum number of densities \f$ \rho^{(p)} \f$
    NS_TYPE nradbase = 0; ///< maximum number of radial \f$\textbf{basis}\f$ function \f$ g_{k}(r) \f$
    LS_TYPE lmax = 0;  ///< \f$ l_\textrm{max} \f$ - maximum value of orbital moment \f$ l \f$
    NS_TYPE nradmax = 0;  ///< maximum number \f$ n \f$ of radial function \f$ R_{nl}(r) \f$
    DOUBLE_TYPE cutoffmax = 0;  ///< maximum value of cutoff distance among all species in basis set
    DOUBLE_TYPE deltaSplineBins = 0;  ///< Spline interpolation density
    vector<string> elements_name; ///< Array of elements name for mapping from index (0..nelements-1) to element symbol (string)
    map<string, SPECIES_TYPE> elements_to_index_map;
    string filename;
    int max_dB_array_size;
    DOUBLE_TYPE nnorm = 1.0;

    GRACEFSEmbeddingSpecification embedding_specifications;

    ACECartesianSphericalHarmonics spherical_harmonics;
    GRACEFSRadialFunction radial_functions;

    vector<vector<GRACEFSBasisFunction>> basis; // [nelements][nfunctions]

    GRACEFSBasisSet() = default;

    explicit GRACEFSBasisSet(const string &filename);

    void load(const string &filename);

    void FS_values_and_derivatives(Array1D<DOUBLE_TYPE> &rhos, DOUBLE_TYPE &value,
                                   Array1D<DOUBLE_TYPE> &derivatives,
                                   SPECIES_TYPE mu_i);

    // flatten arrays

    vector<Array1D<RANK_TYPE>> ranks_flatten; //[el][func_ind]
    vector<Array1D<DENSITY_TYPE>> ndensity_flatten; //[el][func_ind]
    vector<Array1D<short>> num_ms_combs_flatten; // [el][func_ind]]
    vector<Array1D<NS_TYPE>> ns_flatten; //[el][func_ind] -> func_ind x ns(1)
    vector<Array1D<LS_TYPE>> ls_flatten; //[el][func_ind][rank] -> func_ind x ls(rank)
    vector<Array1D<MS_TYPE>> ms_combs_flatten; // [el][func_ind][num_ms_comb][rank] -> func_ind x num_ms_comb x rank

    vector<Array1D<DOUBLE_TYPE>> gen_cgs_flatten; // [el][func_ind][num_ms_comb] -> func_ind x num_ms_comb x rank
    vector<Array1D<DOUBLE_TYPE>> coeff_flatten; //[el][func_ind][n_density]

    vector<Array1D<unsigned int>> func_ind_to_ls;
    vector<Array1D<unsigned int>> func_ind_to_ms_combs;
    vector<Array1D<unsigned int>> func_ind_to_gen_cgs;
    vector<Array1D<unsigned int>> func_ind_to_coeff;

    // per-atom E0s, global shift, global scale
    vector<DOUBLE_TYPE> E0_shift;
    DOUBLE_TYPE shift=0.0;
    DOUBLE_TYPE scale=1.0;

    void convert_to_flatten_arrays();
    void print_functions();

    SPECIES_TYPE get_species_index_by_name(const string &elemname) {
        for (SPECIES_TYPE t = 0; t < nelements; t++) {
            if (this->elements_name[t] == elemname)
                return t;
        }
        return -1;
    }
};

class GRACEFSBEvaluator {
public:
    ///3D array with (l,m) last indices  for storing A's for rank>1: A(n, l, m)
    Array3DLM<DOUBLE_TYPE> A = Array3DLM<DOUBLE_TYPE>("A");

    Array1D<DOUBLE_TYPE> rhos = Array1D<DOUBLE_TYPE>(
            "rhos"); ///< densities \f$ \rho^{(p)} \f$(ndensity), p  = 0 .. ndensity-1
    Array1D<DOUBLE_TYPE> dF_drho = Array1D<DOUBLE_TYPE>(
            "dF_drho"); ///< derivatives of cluster functional wrt. densities, index = 0 .. ndensity-1

    /**
    * Mapping from external atoms types, i.e. LAMMPS, to internal SPECIES_TYPE, used in basis functions
    */
    Array1D<int> element_type_mapping = Array1D<int>("element_type_mapping");

    DOUBLE_TYPE e_atom = 0; ///< energy of current atom, including core-repulsion

    /**
     * The key method to compute energy and forces for atom 'i'.
     * Method will update the  "e_atom" variable and "neighbours_forces(jj, alpha)" array
     *
     * @param i atom index
     * @param x atomic positions array of the real and ghost atoms, shape: [atom_ind][3]
     * @param type  atomic types array of the real and ghost atoms, shape: [atom_ind]
     * @param jnum  number of neighbours of atom_i
     * @param jlist array of neighbour indices, shape: [jnum]
     */
    void compute_atom(int i, DOUBLE_TYPE **x, const SPECIES_TYPE *type, const int jnum, const int *jlist);

    /**
     * Resize all caches over neighbours atoms
     * @param max_jnum  maximum number of neighbours
     */
    void resize_neighbours_cache(int max_jnum);

#ifdef EXTRA_C_PROJECTIONS
    bool compute_projections = false;
    /* 1D array to store projections of basis function (all ranks), shape: [func_ind] */
    Array1D<DOUBLE_TYPE> projections = Array1D<DOUBLE_TYPE>("projections");

//    Array1D<DOUBLE_TYPE> dE_dc = Array1D<DOUBLE_TYPE>("dE_dc");

    // active sets
    map<SPECIES_TYPE, Array2D<DOUBLE_TYPE>> A_active_set_inv;

    DOUBLE_TYPE max_gamma_grade = 0;

    void load_active_set(const string &asi_filename);
#endif

    /**
     * Weights \f$ \omega_{i \mu n l m} \f$ for rank > 1, see Eq.(10) from implementation notes,
     * 'i' is fixed for the current atom, shape: [nelements][nradbase][l=0..lmax, m]
     */
    Array3DLM<DOUBLE_TYPE> weights = Array3DLM<DOUBLE_TYPE>("weights");


    /**
     * cache for gradients of \f$ g(r)\f$: grad_phi(jj,n)=A2DLM(l,m)
     * shape:[max_jnum][nradbase]
     */
    Array2D<DOUBLE_TYPE> DG_cache = Array2D<DOUBLE_TYPE>("DG_cache");

    /**
 * cache for \f$ R_{nl}(r)\f$
 * shape:[max_jnum][nradbase][0..lmax]
 */
    Array3D<DOUBLE_TYPE> R_cache = Array3D<DOUBLE_TYPE>("R_cache");
    /**
     * cache for derivatives of \f$ R_{nl}(r)\f$
     * shape:[max_jnum][nradbase][0..lmax]
     */
    Array3D<DOUBLE_TYPE> DR_cache = Array3D<DOUBLE_TYPE>("DR_cache");
    /**
     * cache for \f$ Y_{lm}(\hat{r})\f$
     * shape:[max_jnum][0..lmax][m]
     */
    Array3DLM<DOUBLE_TYPE> Y_cache = Array3DLM<DOUBLE_TYPE>("Y_cache");
    /**
     * cache for \f$ \nabla Y_{lm}(\hat{r})\f$
     * shape:[max_jnum][0..lmax][m]
     */
    Array3DLM<ACEDRealYcomponent> DY_cache = Array3DLM<ACEDRealYcomponent>("dY_dense_cache");

    /**
     * cache for derivatives of hard-core repulsion
     * shape:[max_jnum]
     */
    Array1D<DOUBLE_TYPE> DCR_cache = Array1D<DOUBLE_TYPE>("DCR_cache");

    /**
    * Partial derivatives \f$ dB_{i \mu n l m t}^{(r)} \f$  with sequential numbering over [func_ind][ms_ind][r],
    * shape:[func_ms_r_ind]
    */
    Array1D<DOUBLE_TYPE> dB_flatten = Array1D<DOUBLE_TYPE>("dB_flatten");
    Array1D<DOUBLE_TYPE> r_norms = Array1D<DOUBLE_TYPE>("r_norms");
    Array1D<DOUBLE_TYPE> inv_r_norms = Array1D<DOUBLE_TYPE>("inv_r_norms");
    Array2D<DOUBLE_TYPE> rhats = Array2D<DOUBLE_TYPE>("rhats");//normalized vector
    Array1D<SPECIES_TYPE> elements = Array1D<SPECIES_TYPE>("elements");

    GRACEFSBasisSet basis_set;


    /**
 * temporary array for the pair forces between current atom_i and its neighbours atom_k
 * neighbours_forces(k,3),  k = 0..num_of_neighbours(atom_i)-1
 */
    Array2D<DOUBLE_TYPE> neighbours_forces = Array2D<DOUBLE_TYPE>("neighbours_forces");

    /**
     * Initialize internal arrays according to basis set sizes
     * @param basis_set
     */
    void init(GRACEFSBasisSet &basis_set);

    void set_basis(GRACEFSBasisSet &basis_set);

    ACETimer setup_timer;
    ACETimer A_construction_timer; ///< timer for loop over neighbours when constructing A's for single central atom
    ACETimer A_construction_timer2; ///< timer for loop over neighbours when constructing A's for single central atom
    ACETimer weights_and_theta_timer;
    ACETimer per_atom_calc_timer; ///< timer for single compute_atom call


    ACETimer forces_calc_loop_timer; ///< timer for forces calculations for single central atom
//    ACETimer forces_calc_neighbour_timer; ///< timer for loop over neighbour atoms for force calculations

    ACETimer energy_calc_timer; ///< timer for energy calculation
    ACETimer total_time_calc_timer; ///< timer for total calculations of all atoms within given atomic environment system

    /**
     * Initialize all timers
     */
    void init_timers();
};

#endif //GRACE_FS_EVALUATOR_H

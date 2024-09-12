//
// Created by Yury Lysogorskiy on 24.06.24
//
#include "ace/grace_fs_evaluator.h"
#include "ace-evaluator/ace_types.h"
#include "ace/ace_yaml_input.h"
#include "ace-evaluator/ace_radial.h"
#include "cnpy/cnpy.h"

#define sqr(x) ((x)*(x))
const DOUBLE_TYPE pi = 3.14159265358979323846264338327950288419; // pi

void GRACEFSEmbeddingSpecification::from_YAML(YAML_PACE::Node emb_yaml) {
    this->type = emb_yaml["type"].as<string>();
    this->FS_parameters = emb_yaml["params"].as<vector<double>>();
    if (this->FS_parameters.size() % 2 == 0)
        this->ndensity = this->FS_parameters.size() / 2;
    else {
        throw invalid_argument("Number of parameters in emb_spec::params is not even");
    }
//    this->ndensity = emb_yaml["ndensity"].as<DENSITY_TYPE>();
}

void GRACEFSRadialFunction::from_YAML(YAML_PACE::Node bond_yaml) {

    this->radbasename = bond_yaml["radbasename"].as<string>();

    this->nradmax = bond_yaml["nradmax"].as<NS_TYPE>();
    this->nradbasemax = bond_yaml["n_rad_base"].as<NS_TYPE>();

    this->crad_shape = bond_yaml["crad_shape"].as<vector<int>>(); // [n_radial][lmax+1][n_rad_base]
    this->Z_shape = bond_yaml["Z_shape"].as<vector<int>>(); // [n_elements][n_radial]
    this->nelemets = this->Z_shape.at(0);

    this->rad_lamba = 0; // TODO: check ?
    this->lmax = crad_shape[1] - 1; // crad = [n,l,k], crad_shape[..., lmax+1, ...]

    auto crad_flat = bond_yaml["crad"].as<vector<DOUBLE_TYPE>>();
    this->crad.init(crad_shape[0], crad_shape[1], crad_shape[2]);
    this->crad.set_flatten_vector(crad_flat);

    auto Z_flat = bond_yaml["Z"].as<vector<DOUBLE_TYPE>>();
    this->Z.init(Z_shape[0], Z_shape[1], "Z");
    Z.set_flatten_vector(Z_flat);


    rcut = bond_yaml["cutoff"].as<DOUBLE_TYPE>();
    if (bond_yaml["dcut"]) dcut = bond_yaml["dcut"].as<DOUBLE_TYPE>();

    if (bond_yaml["rcut_in"]) rcut_in = bond_yaml["rcut_in"].as<DOUBLE_TYPE>();
    if (bond_yaml["dcut_in"]) dcut_in = bond_yaml["dcut_in"].as<DOUBLE_TYPE>();
}

void GRACEFSBasisFunction::from_YAML(YAML_PACE::Node node) {
    this->mu0 = node["mu0"].as<SPECIES_TYPE>();

    this->ns = node["ns"].as<NS_TYPE>();
    this->ls = node["ls"].as<vector<LS_TYPE>>();
//    if (this->ns.size() != this->ls.size())
//        throw invalid_argument("ns.size!=ls.size");
    this->rank = this->ls.get_size();


    this->ms_combs = node["ms_combs"].as<vector<MS_TYPE>>();
    this->gen_cgs = node["gen_cgs"].as<vector<DOUBLE_TYPE >>();
    if (this->gen_cgs.get_size() * this->rank != this->ms_combs.get_size())
        throw invalid_argument("gen_cg.size * rank != ms_combs.size");
    this->num_ms_combs = this->gen_cgs.get_size();

    this->ndensity = node["ndensity"].as<DENSITY_TYPE>();
    this->coeff = node["coeff"].as<vector<DOUBLE_TYPE>>();
    if (this->ndensity != this->coeff.get_size())
        throw invalid_argument("ndensity != coeff.size");
}

void GRACEFSBasisFunction::print() const {
    cout << "TDACEBasisFunction(mu0=" << this->mu0 << ", rank=" << (int) this->rank << endl;
    int rank = this->rank;

    for (int ms_ind = 0; ms_ind < this->num_ms_combs; ms_ind++) {
        for (int r = 0; r < rank; r++)
            cout << "A(" << this->ns << "," << this->ls(r) << "," << this->ms_combs(ms_ind * rank + r) << ")";
        cout << " * " << this->gen_cgs(ms_ind) << endl;
    }
    cout << ")" << endl;
}

GRACEFSBasisSet::GRACEFSBasisSet(const string &filename) {
    this->load(filename);
}

void GRACEFSBasisSet::load(const string &filename) {
    this->filename = filename;

    if (!if_file_exist(filename)) {
        stringstream s;
        s << "Potential file " << filename << " doesn't exists";
        cerr << "Exception: " << s.str();
        throw invalid_argument(s.str());
    }

    //load the file with yaml
    YAML_PACE::Node YAML_input = YAML_PACE::LoadFile(filename);
    this->elements_name = YAML_input["elements"].as<vector<string>>();
    this->nelements = this->elements_name.size();
    for (int i = 0; i < this->nelements; i++)
        this->elements_to_index_map[this->elements_name[i]] = i;

    this->embedding_specifications.from_YAML(YAML_input["emb_spec"]);

    this->radial_functions.from_YAML(YAML_input["radial_basis"]);

    if (YAML_input["nnorm"])
        this->nnorm = YAML_input["nnorm"].as<DOUBLE_TYPE>(); //inv_avg_n_neigh

    if (this->radial_functions.nelemets != this->nelements)
        throw invalid_argument("radial_basis.Z_shape != elements.size");

    // load functions
    this->basis.resize(this->nelements);
    RANK_TYPE rankmax = 0;
    for (const auto &mu_functions_node: YAML_input["functions"]) {
        SPECIES_TYPE mu0 = mu_functions_node.first.as<SPECIES_TYPE>();
        const auto &functions_list = mu_functions_node.second;
        vector<GRACEFSBasisFunction> basis_functions_vec;
        for (const auto &func_node: functions_list) {
            GRACEFSBasisFunction func;
            func.from_YAML(func_node);
            basis_functions_vec.emplace_back(func);
            if (func.rank > rankmax)
                rankmax = func.rank;
        }
        this->basis.at(mu0) = basis_functions_vec;
    }

    //////////////// initialize variables /////////////////
    this->nradmax = this->radial_functions.nradmax;
    this->lmax = this->radial_functions.lmax;
    this->nradbase = this->radial_functions.nradbasemax;
    this->ndensitymax = this->embedding_specifications.ndensity;
    this->cutoffmax = this->radial_functions.rcut;
    this->rankmax = rankmax;

    // compute  max_dB_array_size (among different central species)
    this->max_dB_array_size = 0;
    int cur_ms_rank_size = 0;

    for (SPECIES_TYPE mu = 0; mu < nelements; ++mu) {
        cur_ms_rank_size = 0;

        for (const auto &func: this->basis.at(mu))
            cur_ms_rank_size += func.rank * func.num_ms_combs;

        if (cur_ms_rank_size > max_dB_array_size)
            max_dB_array_size = cur_ms_rank_size;
    }

    this->radial_functions.init();
    this->spherical_harmonics.init(this->lmax);

    this->convert_to_flatten_arrays();

    //load E0, shift and scale
    if (YAML_input["E0"]) this->E0_shift = YAML_input["E0"].as<vector<DOUBLE_TYPE>>();
    else this->E0_shift.resize(nelements, 0.0);

    if (YAML_input["shift"]) this->shift = YAML_input["shift"].as<DOUBLE_TYPE>();
    else this->shift = 0.0;

    if (YAML_input["scale"]) this->scale = YAML_input["scale"].as<DOUBLE_TYPE>();
    else this->scale = 1.0;
}

void GRACEFSBasisSet::convert_to_flatten_arrays() {
    ranks_flatten.resize(nelements);
    ndensity_flatten.resize(nelements);
    num_ms_combs_flatten.resize(nelements);
    ns_flatten.resize(nelements);
    ls_flatten.resize(nelements);
    ms_combs_flatten.resize(nelements);
    gen_cgs_flatten.resize(nelements);
    coeff_flatten.resize(nelements);

    func_ind_to_ls.resize(nelements);
    func_ind_to_ms_combs.resize(nelements);
    func_ind_to_gen_cgs.resize(nelements);
    func_ind_to_coeff.resize(nelements);

    for (int el = 0; el < nelements; el++) {
//        printf("Element: %d\n", el);

        const auto num_funcs = basis[el].size();
        ranks_flatten[el].init(num_funcs, "ranks_flatten");
        ndensity_flatten[el].init(num_funcs, "ndensity_flatten");
        num_ms_combs_flatten[el].init(num_funcs, "num_ms_combs_flatten");

        ns_flatten[el].init(num_funcs, "ns_flatten");

        int num_ls = 0;
        int num_ms = 0;
        int num_gen_cgs = 0;
        int num_coeff = 0;
        for (const auto &func: basis[el]) {
            num_ls += func.rank;
            num_ms += func.num_ms_combs * func.rank;
            num_gen_cgs += func.num_ms_combs;
            num_coeff += func.ndensity;
        }
//        printf("num_ls=%d\n", num_ls);
//        printf("num_ms=%d\n", num_ms);
//        printf("num_gen_cgs=%d\n", num_gen_cgs);
//        printf("num_coeff=%d\n", num_coeff);

        ls_flatten[el].init(num_ls, "ls_flatten");
        ms_combs_flatten[el].init(num_ms, "ms_combs_flatten");

        gen_cgs_flatten[el].init(num_gen_cgs, "gen_cgs_flatten");
        coeff_flatten[el].init(num_coeff, "coeff_flatten");

        func_ind_to_ls[el].init(num_funcs, "func_ind_to_ls");
        func_ind_to_ms_combs[el].init(num_funcs, "func_ind_to_ms_combs");
        func_ind_to_gen_cgs[el].init(num_funcs, "func_ind_to_gen_cgs");
        func_ind_to_coeff[el].init(num_funcs, "func_ind_to_coeff");
    }

    // population of arrays
    for (int el = 0; el < nelements; el++) {
        int ind_ls = 0;
        int ind_ms = 0;
        int ind_gen_cgs = 0;
        int ind_coeff = 0;

        int func_ind = 0;
        for (const auto &func: basis[el]) {
            ranks_flatten[el](func_ind) = func.rank;
            ndensity_flatten[el](func_ind) = func.ndensity;
            num_ms_combs_flatten[el](func_ind) = func.num_ms_combs;
            ns_flatten[el](func_ind) = func.ns;

            // ls_flatten
            func_ind_to_ls[el](func_ind) = ind_ls;
            ls_flatten[el].move_from(func.ls, ind_ls);
            ind_ls += func.rank;

            // ms_combs_flatten
            func_ind_to_ms_combs[el](func_ind) = ind_ms;
            ms_combs_flatten[el].move_from(func.ms_combs, ind_ms);
            ind_ms += func.num_ms_combs * func.rank;

            // gen_cgs_flatten
            func_ind_to_gen_cgs[el](func_ind) = ind_gen_cgs;
            gen_cgs_flatten[el].move_from(func.gen_cgs, ind_gen_cgs);
            ind_gen_cgs += func.num_ms_combs;

            // coeff_flatten
            func_ind_to_coeff[el](func_ind) = ind_coeff;
            coeff_flatten[el].move_from(func.coeff, ind_coeff);
            ind_coeff += func.ndensity;

            func_ind++;
        }
    }
}

void GRACEFSBasisSet::FS_values_and_derivatives(Array1D<DOUBLE_TYPE> &rhos, DOUBLE_TYPE &value,
                                                Array1D<DOUBLE_TYPE> &derivatives,
                                                SPECIES_TYPE mu_i) {
    DOUBLE_TYPE F, DF = 0, wpre, mexp;
    DENSITY_TYPE ndensity = this->ndensitymax;

    for (int p = 0; p < ndensity; p++) {
        wpre = this->embedding_specifications.FS_parameters[p * 2 + 0];
        mexp = this->embedding_specifications.FS_parameters[p * 2 + 1];
        string npoti = this->embedding_specifications.type;

        if (npoti == "FinnisSinclair")
            Fexp(rhos(p), mexp, F, DF);
        else if (npoti == "FinnisSinclairShiftedScaled")
            FexpShiftedScaled(rhos(p), mexp, F, DF);

        value += F * wpre; // * weight (wpre)
        derivatives(p) = DF * wpre;// * weight (wpre)
    }
}

void GRACEFSBasisSet::print_functions() {
    for (int el = 0; el < nelements; el++) {
        printf("============================= Element: %d =====================\n", el);
        for (const auto &func: basis[el]) {
            func.print();
        }
    }
}

void GRACEFSBEvaluator::set_basis(GRACEFSBasisSet &basis_set) {
    this->basis_set = basis_set;
    init(this->basis_set);
}

void GRACEFSBEvaluator::init(GRACEFSBasisSet &basis_set) {
    A.init(basis_set.nradmax + 1, basis_set.lmax + 1, "A");

    rhos.init(basis_set.ndensitymax, "rhos");
    dF_drho.init(basis_set.ndensitymax, "dF_drho");

    weights.init(basis_set.nradmax + 1, basis_set.lmax + 1, "weights");


    DG_cache.init(1, basis_set.nradbase, "DG_cache");
    DG_cache.fill(0);

    R_cache.init(1, basis_set.nradmax, basis_set.lmax + 1, "R_cache");
    R_cache.fill(0);

    DR_cache.init(1, basis_set.nradmax, basis_set.lmax + 1, "DR_cache");
    DR_cache.fill(0);

    Y_cache.init(1, basis_set.lmax + 1, "Y_cache");
    Y_cache.fill(0);

    DY_cache.init(1, basis_set.lmax + 1, "dY_dense_cache");
    DY_cache.fill({0., 0, 0});

    //hard-core repulsion
    DCR_cache.init(1, "DCR_cache");
    DCR_cache.fill(0);

    dB_flatten.init(basis_set.max_dB_array_size, "dB_flatten");
}


void GRACEFSBEvaluator::resize_neighbours_cache(int max_jnum) {
    if (R_cache.get_dim(0) < max_jnum) {

        //TODO: implement grow
        R_cache.resize(max_jnum, basis_set.nradmax, basis_set.lmax + 1);
        R_cache.fill(0);

        DR_cache.resize(max_jnum, basis_set.nradmax, basis_set.lmax + 1);
        DR_cache.fill(0);

        DG_cache.resize(max_jnum, basis_set.nradbase);
        DG_cache.fill(0);

        Y_cache.resize(max_jnum, basis_set.lmax + 1);
        Y_cache.fill(0);

        DY_cache.resize(max_jnum, basis_set.lmax + 1);
        DY_cache.fill({0, 0, 0});

        //hard-core repulsion
        DCR_cache.resize(max_jnum);
        DCR_cache.fill(0);

        r_norms.resize(max_jnum);
        inv_r_norms.resize(max_jnum);
        rhats.resize(max_jnum, 3);
        elements.resize(max_jnum);
    }
}


void GRACEFSBEvaluator::compute_atom(int i, DOUBLE_TYPE **x, const SPECIES_TYPE *type, const int jnum, const int *jlist) {
    per_atom_calc_timer.start();

    setup_timer.start();
    DOUBLE_TYPE evdwl = 0;
    DOUBLE_TYPE r_norm;
    DOUBLE_TYPE xn, yn, zn, r_xyz;
    DOUBLE_TYPE R, R_over_r, DR;
    DOUBLE_TYPE *r_hat;

    SPECIES_TYPE mu_j;
    RANK_TYPE r, rank, t;
    NS_TYPE n;
    LS_TYPE l;
    MS_TYPE m, m_t;

    NS_TYPE ns;
    LS_TYPE *ls;
    MS_TYPE *ms;

    int j, jj, func_ind, ms_ind;

    DOUBLE_TYPE Y, Y_DR;
    DOUBLE_TYPE B;
    DOUBLE_TYPE dB;
    Array1D<DOUBLE_TYPE> A_cache(basis_set.rankmax);
    ACEDRealYcomponent grad_phi_nlm, DY;

    //size is +1 of max to avoid out-of-boundary array access in double-triangular scheme
    Array1D<DOUBLE_TYPE> A_forward_prod(basis_set.rankmax + 1);
    Array1D<DOUBLE_TYPE> A_backward_prod(basis_set.rankmax + 1);

    DOUBLE_TYPE inv_r_norm;
    const DOUBLE_TYPE xtmp = x[i][0];
    const DOUBLE_TYPE ytmp = x[i][1];
    const DOUBLE_TYPE ztmp = x[i][2];
    DOUBLE_TYPE f_ji[3];

    bool is_element_mapping = element_type_mapping.get_size() > 0;
    SPECIES_TYPE mu_i;
    if (is_element_mapping)
        mu_i = element_type_mapping(type[i]);
    else
        mu_i = type[i];

    const SHORT_INT_TYPE total_basis_size = basis_set.basis[mu_i].size();

    const DENSITY_TYPE ndensity = basis_set.embedding_specifications.ndensity;

    neighbours_forces.resize(jnum, 3);
    neighbours_forces.fill(0.0);

    weights.fill(0);
    A.fill(0);
    rhos.fill(0);
    dF_drho.fill(0);

#ifdef EXTRA_C_PROJECTIONS
    if (this->compute_projections) {
        projections.init(total_basis_size, "projections");
        projections.fill(0);
    }
#endif

    setup_timer.stop();

    A_construction_timer.start();
    const auto &ylm = basis_set.spherical_harmonics.real_ylm;
    const auto &dylm = basis_set.spherical_harmonics.real_dylm;

    const auto &fr = basis_set.radial_functions.fr;
    const auto &dfr = basis_set.radial_functions.dfr;

    const auto &ranks_flatten = basis_set.ranks_flatten[mu_i];
    const auto &ns_flatten = basis_set.ns_flatten[mu_i];
    auto &ls_flatten = basis_set.ls_flatten[mu_i];
    const auto &func_ind_to_ls = basis_set.func_ind_to_ls[mu_i];
    const auto &num_ms_combs_flatten = basis_set.num_ms_combs_flatten[mu_i];
    const auto &func_ind_to_ms_combs = basis_set.func_ind_to_ms_combs[mu_i];
    auto &ms_combs_flatten = basis_set.ms_combs_flatten[mu_i];
    const auto &func_ind_to_gen_cgs = basis_set.func_ind_to_gen_cgs[mu_i];
    const auto &func_ind_to_coeff = basis_set.func_ind_to_coeff[mu_i];
    const auto &gen_cgs_flatten = basis_set.gen_cgs_flatten[mu_i];
    const auto &coeff_flatten = basis_set.coeff_flatten[mu_i];
    int jj_actual = 0;
    SPECIES_TYPE type_j = 0;
    Array1D<int> neighbour_index_mapping(jnum); // jj_actual -> jj
    //loop over neighbours, compute distance, consider only atoms within with r<cutoff(mu_i, mu_j)
    for (jj = 0; jj < jnum; ++jj) {

        j = jlist[jj];
        xn = x[j][0] - xtmp;
        yn = x[j][1] - ytmp;
        zn = x[j][2] - ztmp;
        type_j = type[j];
        if (is_element_mapping)
            mu_j = element_type_mapping(type_j);
        else
            mu_j = type_j;


        r_xyz = sqrt(xn * xn + yn * yn + zn * zn);

        if (r_xyz >= basis_set.cutoffmax)
            continue;

        inv_r_norm = 1 / r_xyz;

        r_norms(jj_actual) = r_xyz;
        inv_r_norms(jj_actual) = inv_r_norm;
        rhats(jj_actual, 0) = xn * inv_r_norm;
        rhats(jj_actual, 1) = yn * inv_r_norm;
        rhats(jj_actual, 2) = zn * inv_r_norm;
        elements(jj_actual) = mu_j;
        neighbour_index_mapping(jj_actual) = jj;
        jj_actual++;
    }

    int jnum_actual = jj_actual;

    //ALGORITHM 1: Atomic base A
    for (jj = 0; jj < jnum_actual; ++jj) {
        r_norm = r_norms(jj);
        mu_j = elements(jj);
        r_hat = &rhats(jj, 0);

        //proxies
        auto &Y_jj = Y_cache(jj);
        auto &DY_jj = DY_cache(jj);

        basis_set.radial_functions.evaluate(r_norm, mu_i, mu_j);
        basis_set.spherical_harmonics.compute_real_ylm(r_hat[0], r_hat[1], r_hat[2], basis_set.lmax);

        //loop for computing A's
        // for rank > 1
        //        A_construction_timer2.start();

        for (n = 0; n < basis_set.nradmax; n++) {
            auto &A_lm = A(n);
            for (l = 0; l <= basis_set.lmax; l++) {
                R = fr(n, l) * basis_set.nnorm;

                DR_cache(jj, n, l) = dfr(n, l) * basis_set.nnorm;
                R_cache(jj, n, l) = R;

                for (m = -l; m <= l; m++) {
                    Y = ylm(l, m);
                    A_lm(l, m) += R * Y; //accumulation sum over neighbours
                    Y_jj(l, m) = Y;
                    DY_jj(l, m) = dylm(l, m);
                }

            }
        }
//        A_construction_timer2.stop();

    } //end loop over neighbours
    A_construction_timer.stop();

    //now A's are constructed (no need for complex conjugation form -m, since it was constructed

    // ==================== ENERGY ====================

    energy_calc_timer.start();
    //ALGORITHM 2: Basis functions B with iterative product and density rho(p) calculation
    // all rank>=1
    int func_ms_ind = 0;
    int func_ms_t_ind = 0;// index for dB
    int num_ms_combs = 0;


    for (func_ind = 0; func_ind < total_basis_size; ++func_ind) {
        rank = ranks_flatten(func_ind);
        r = rank - 1;
        ns = ns_flatten(func_ind);
        ls = &ls_flatten(func_ind_to_ls(func_ind));

        //loop over {ms} combinations in sum
        num_ms_combs = num_ms_combs_flatten(func_ind);
        const auto ms_ind_shift = func_ind_to_ms_combs(func_ind);
        const auto gen_cgs_shift = func_ind_to_gen_cgs(func_ind);
        const auto coeff_shift = func_ind_to_coeff(func_ind);

        for (ms_ind = 0; ms_ind < num_ms_combs; ++ms_ind, ++func_ms_ind) {
            ms = &ms_combs_flatten(ms_ind_shift + ms_ind * rank);

            //loop over m, collect B  = product of A with given ms
            A_forward_prod(0) = 1;
            A_backward_prod(r) = 1;

            //fill forward A-product triangle
            for (t = 0; t < rank; t++) {
                A_cache(t) = A(ns - 1, ls[t], ms[t]);
                A_forward_prod(t + 1) = A_forward_prod(t) * A_cache(t);
            }

            B = A_forward_prod(t);

            //fill backward A-product triangle
            for (t = r; t >= 1; t--)
                A_backward_prod(t - 1) = A_backward_prod(t) * A_cache(t);

            for (t = 0; t < rank; ++t, ++func_ms_t_ind) {
                dB = A_forward_prod(t) * A_backward_prod(t); //dB - product of all A's except t-th
                dB_flatten(func_ms_t_ind) = dB;
            }

#ifdef EXTRA_C_PROJECTIONS
            if (this->compute_projections) {
                //aggregate C-projections separately
                projections(func_ind) += B * gen_cgs_flatten(gen_cgs_shift + ms_ind);
            }
#endif

            for (DENSITY_TYPE p = 0; p < ndensity; ++p) {
                //real-part only multiplication
                rhos(p) += B * gen_cgs_flatten(gen_cgs_shift + ms_ind) *
                           coeff_flatten(coeff_shift + p);
            }
        }//end of loop over {ms} combinations in sum
    } // end loop for rank>=1

    basis_set.FS_values_and_derivatives(rhos, evdwl, dF_drho, mu_i);
    energy_calc_timer.stop();

    //ALGORITHM 3: Weights and theta calculation
    // all ranks>=1
    weights_and_theta_timer.start();
    func_ms_ind = 0;
    func_ms_t_ind = 0;// index for dB
    DOUBLE_TYPE theta;
    for (func_ind = 0; func_ind < total_basis_size; ++func_ind) {
        rank = ranks_flatten(func_ind);
        ns = ns_flatten(func_ind);
        ls = &ls_flatten(func_ind_to_ls(func_ind));
        num_ms_combs = num_ms_combs_flatten(func_ind);
        const auto ms_ind_shift = func_ind_to_ms_combs(func_ind);
        const auto gen_cgs_shift = func_ind_to_gen_cgs(func_ind);
        const auto coeff_shift = func_ind_to_coeff(func_ind);

        for (ms_ind = 0; ms_ind < num_ms_combs; ++ms_ind, ++func_ms_ind) {
//            ms = &func.ms_combs.get_data()[ms_ind * rank]; // current ms-combination (of length = rank)
            ms = &ms_combs_flatten(ms_ind_shift + ms_ind * rank);

            theta = 0;
            for (DENSITY_TYPE p = 0; p < ndensity; ++p)
                theta += dF_drho(p) * gen_cgs_flatten(gen_cgs_shift + ms_ind) *
                         coeff_flatten(coeff_shift + p);


            for (t = 0; t < rank; ++t, ++func_ms_t_ind) {
                m_t = ms[t];
                dB = dB_flatten(func_ms_t_ind);
                weights(ns - 1, ls[t], m_t) += theta * dB; //Theta_array(func_ms_ind);
            }
        }
    }
    weights_and_theta_timer.stop();

    // ==================== FORCES ====================
    // loop over neighbour atoms for force calculations
    forces_calc_loop_timer.start();
    for (jj = 0; jj < jnum_actual; ++jj) {
        r_hat = &rhats(jj, 0);
        inv_r_norm = inv_r_norms(jj);
        const auto &Y_cache_jj = Y_cache(jj);
        const auto &DY_cache_jj = DY_cache(jj);

        f_ji[0] = f_ji[1] = f_ji[2] = 0;

        //for rank >= 1
        for (n = 0; n < basis_set.nradmax; n++) {
            for (l = 0; l <= basis_set.lmax; l++) {
                R_over_r = R_cache(jj, n, l) * inv_r_norm;
                DR = DR_cache(jj, n, l);

                // for all m
                for (m = -l; m <= l; m++) {
                    auto w = weights(n, l, m);
                    if (w == 0)
                        continue;
                    DY = DY_cache_jj(l, m);
                    Y_DR = Y_cache_jj(l, m) * DR;

                    grad_phi_nlm.a[0] = Y_DR * r_hat[0] + DY.a[0] * R_over_r;
                    grad_phi_nlm.a[1] = Y_DR * r_hat[1] + DY.a[1] * R_over_r;
                    grad_phi_nlm.a[2] = Y_DR * r_hat[2] + DY.a[2] * R_over_r;

                    f_ji[0] += w * grad_phi_nlm.a[0];
                    f_ji[1] += w * grad_phi_nlm.a[1];
                    f_ji[2] += w * grad_phi_nlm.a[2];
                }
            }
        }
        neighbours_forces(neighbour_index_mapping(jj), 0) = basis_set.scale * f_ji[0];
        neighbours_forces(neighbour_index_mapping(jj), 1) = basis_set.scale * f_ji[1];
        neighbours_forces(neighbour_index_mapping(jj), 2) = basis_set.scale * f_ji[2];

    } // end for-loop over neighbour atoms for force calculations
    forces_calc_loop_timer.stop();

    e_atom = basis_set.scale * evdwl + basis_set.shift + basis_set.E0_shift.at(mu_i);

#ifdef EXTRA_C_PROJECTIONS
    if (this->compute_projections) {
        //check if active set is loaded
        // use dE_dc or projections as asi_vector
        if (A_active_set_inv.find(mu_i) != A_active_set_inv.end()) {
            Array1D<DOUBLE_TYPE> &asi_vector = this->projections;
            // get inverted active set for current species type
            const auto &A_as_inv = A_active_set_inv.at(mu_i);

            DOUBLE_TYPE gamma_max = 0;
            for (int i = 0; i < A_as_inv.get_dim(0); i++) {
                DOUBLE_TYPE current_gamma = 0;
                // compute row-matrix-multiplication asi_vector * A_as_inv (transposed matrix)
                for (int k = 0; k < asi_vector.get_dim(0); k++)
                    current_gamma += asi_vector(k) * A_as_inv(i, k);

                if (abs(current_gamma) > gamma_max)
                    gamma_max = abs(current_gamma);
            }

            max_gamma_grade = gamma_max;
        }
    }
#endif
    per_atom_calc_timer.stop();
}

void GRACEFSBEvaluator::init_timers() {
    A_construction_timer.init();
    A_construction_timer2.init();
    forces_calc_loop_timer.init();
    energy_calc_timer.init();
    per_atom_calc_timer.init();
    total_time_calc_timer.init();
    weights_and_theta_timer.init();
    setup_timer.init();
}


void GRACEFSBEvaluator::load_active_set(const string &asi_filename) {
    cnpy::npz_t asi_npz = cnpy::npz_load(asi_filename);
    if (asi_npz.size() != this->basis_set.nelements) {
        stringstream ss;
        ss << "Number of species types in ASI `" << asi_filename << "` (" << asi_npz.size() << ")";
        ss << "not equal to number of species in TDACEBBassiSet (" << this->basis_set.nelements << ")";
        throw std::runtime_error(ss.str());
    }

    for (auto &kv: asi_npz) {
        auto element_name = kv.first;
        SPECIES_TYPE st = this->basis_set.elements_to_index_map.at(element_name);
        auto shape = kv.second.shape;
        // auto_determine extrapolation grade type: linear or non-linear
//        validate_ASI_square_shape(st, shape);
        if (shape.at(0) != shape.at(1)) {
            stringstream ss;
            ss << "Active Set Inverted for element `" << element_name << "`:";
            ss << "should be square matrix, but has shape (" << shape.at(0) << ", " << shape.at(1) << ")";
            throw runtime_error(ss.str());
        }
//        validate_ASI_shape(element_name, st, shape);
        int expected_ASI_size = this->basis_set.basis[st].size();
        if (expected_ASI_size != shape.at(0)) {
            stringstream ss;
            ss << "Active Set Inverted for element `" << element_name << "`:";
            ss << "expected shape: (" << expected_ASI_size << ", " << expected_ASI_size << ") , but has shape ("
               << shape.at(0) << ", " << shape.at(1) << ")";
            throw runtime_error(ss.str());
        }

        Array2D<DOUBLE_TYPE> A0_inv(shape.at(0), shape.at(1), element_name);
        auto data_vec = kv.second.as_vec<DOUBLE_TYPE>();
        A0_inv.set_flatten_vector(data_vec);
        //transpose matrix to speed-up vec-mat multiplication
        Array2D<DOUBLE_TYPE> A0_inv_transpose(A0_inv.get_dim(1), A0_inv.get_dim(0));

        for (int i = 0; i < A0_inv.get_dim(0); i++)
            for (int j = 0; j < A0_inv.get_dim(1); j++)
                A0_inv_transpose(j, i) = A0_inv(i, j);

        this->A_active_set_inv[st] = A0_inv_transpose;
    }

//    resize_projections();// no need, all projections are the same length
}


void GRACEFSRadialFunction::init() {

    gr.init(nradbasemax, "gr");
    dgr.init(nradbasemax, "dgr");

    fr.init(nradmax, lmax + 1, "fr");
    dfr.init(nradmax, lmax + 1, "dfr");

//    splines_gk.init(nelements, nelements, "splines_gk");
//    splines_rnl.init(nelements, nelements, "splines_rnl");
    setuplookupRadspline();
}


/**
 * Simplified Bessel function
 * @param rc
 * @param x
 */
void GRACEFSRadialFunction::simplified_bessel(DOUBLE_TYPE rc, DOUBLE_TYPE x) {
    if (x < rc) {
        gr(0) = simplified_bessel_aux(x, rc, 0);
        dgr(0) = dsimplified_bessel_aux(x, rc, 0);

        DOUBLE_TYPE d_prev = 1.0, en, dn;
        for (NS_TYPE n = 1; n < this->nradbasemax; n++) {
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

void GRACEFSRadialFunction::radbase(DOUBLE_TYPE lam, DOUBLE_TYPE cut, DOUBLE_TYPE dcut, string radbasename, DOUBLE_TYPE r,
                                    DOUBLE_TYPE cut_in, DOUBLE_TYPE dcut_in) {
    /*lam is given by the formula (24), that contains cut */
    if (r <= cut_in - dcut_in || r >= cut) {
        gr.fill(0);
        dgr.fill(0);
    } else { // cut_in < r < cut
        if (radbasename == "SBessel") {
            simplified_bessel(cut, r);
        } else {
            throw invalid_argument("Unknown radial basis function name: " + radbasename);
        }
    }
}

void GRACEFSRadialFunction::all_radfunc(DOUBLE_TYPE r) {

    // set up radial functions
    radbase(rad_lamba, rcut, dcut, radbasename, r, 0, 0); //update gr, dgr
    radfunc(); // update fr(nr, l),  dfr(nr, l)
}

void GRACEFSRadialFunction::radfunc() {
    DOUBLE_TYPE frval, dfrval;
    for (NS_TYPE n = 0; n < nradmax; n++) {
        for (LS_TYPE l = 0; l <= lmax; l++) {
            frval = 0.0;
            dfrval = 0.0;
            for (NS_TYPE k = 0; k < nradbasemax; k++) {
                frval += crad(n, l, k) * gr(k);
                dfrval += crad(n, l, k) * dgr(k);
            }
            // IMPORTANT!!! MULTIPLICATION BY Z(mu_j,n) will happen in evaluate,
            // otherwise we should store too much almost identical SplineInterpolators
            fr(n, l) = frval;
            dfr(n, l) = dfrval;
        }
    }
}

void GRACEFSRadialFunction::setuplookupRadspline() {
    using namespace std::placeholders;
    splines_gk.setupSplines(gr.get_size(),
                            std::bind(&GRACEFSRadialFunction::radbase, this, this->rad_lamba,
                                      this->rcut, this->dcut,
                                      this->radbasename,
                                      _1, 0, 0),//update gr, dgr
                            gr.get_data(),
                            dgr.get_data(), deltaSplineBins, this->rcut);

    splines_rnl.setupSplines(fr.get_size(),
                             std::bind(&GRACEFSRadialFunction::all_radfunc, this, _1), // update fr(nr, l),  dfr(nr, l)
                             fr.get_data(),
                             dfr.get_data(), deltaSplineBins, rcut);
}

void GRACEFSRadialFunction::evaluate(DOUBLE_TYPE r, SPECIES_TYPE mu_i, SPECIES_TYPE mu_j) {
    splines_gk.calcSplines(r);
    for (NS_TYPE nr = 0; nr < nradbasemax; nr++) {
        gr(nr) = splines_gk.values(nr);
        dgr(nr) = splines_gk.derivatives(nr);
    }

    splines_rnl.calcSplines(r);
    // copy in flatten format
    for (size_t ind = 0; ind < fr.get_size(); ind++) {
        fr.get_data(ind) = splines_rnl.values.get_data(ind);
        dfr.get_data(ind) = splines_rnl.derivatives.get_data(ind);
    }

    //multiply by Z
    for (int n = 0; n < nradmax; n++) {
        for (int l = 0; l <= lmax; l++) {
            fr(n, l) *= Z(mu_j, n);
            dfr(n, l) *= Z(mu_j, n);
        }
    }

    //TODO: hardcore repulsion
}
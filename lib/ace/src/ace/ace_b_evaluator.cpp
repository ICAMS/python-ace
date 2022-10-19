//
// Created by Yury Lysogorskiy on 31.01.20.
//

#include "ace/ace_b_evaluator.h"

#include "cnpy/cnpy.h"

#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_types.h"
#include "ace-evaluator/ace_abstract_basis.h"

//================================================================================================================

void ACEBEvaluator::set_basis(ACEBBasisSet &bas) {
    basis_set = &bas;
    init(basis_set);
}

void ACEBEvaluator::init(ACEBBasisSet *basis_set) {

    ACEEvaluator::init(basis_set);

    weights.init(basis_set->nelements, basis_set->nradmax + 1, basis_set->lmax + 1,
                 "weights");

    weights_rank1.init(basis_set->nelements, basis_set->nradbase, "weights_rank1");


    DG_cache.init(1, basis_set->nradbase, "DG_cache");
    DG_cache.fill(0);

    R_cache.init(1, basis_set->nradmax, basis_set->lmax + 1, "R_cache");
    R_cache.fill(0);

    DR_cache.init(1, basis_set->nradmax, basis_set->lmax + 1, "DR_cache");
    DR_cache.fill(0);

    Y_cache.init(1, basis_set->lmax + 1, "Y_cache");
    Y_cache.fill({0, 0});

    DY_cache.init(1, basis_set->lmax + 1, "dY_dense_cache");
    DY_cache.fill({0.});

    //hard-core repulsion
    DCR_cache.init(1, "DCR_cache");
    DCR_cache.fill(0);
    dB_flatten.init(basis_set->max_dB_array_size, "dB_flatten");


}

void ACEBEvaluator::resize_neighbours_cache(int max_jnum) {
    if (basis_set == nullptr) {
        throw std::invalid_argument("ACEBEvaluator: basis set is not assigned");
    }
    if (R_cache.get_dim(0) < max_jnum) {

        //TODO: implement grow
        R_cache.resize(max_jnum, basis_set->nradmax, basis_set->lmax + 1);
        R_cache.fill(0);

        DR_cache.resize(max_jnum, basis_set->nradmax, basis_set->lmax + 1);
        DR_cache.fill(0);

        DG_cache.resize(max_jnum, basis_set->nradbase);
        DG_cache.fill(0);

        Y_cache.resize(max_jnum, basis_set->lmax + 1);
        Y_cache.fill({0, 0});

        DY_cache.resize(max_jnum, basis_set->lmax + 1);
        DY_cache.fill({0});

        //hard-core repulsion
        DCR_cache.init(max_jnum, "DCR_cache");
        DCR_cache.fill(0);
    }
}



// double** r - atomic coordinates of atom I
// int* types - atomic types if atom I
// int **firstneigh -  ptr to 1st J int value of each I atom. Usage: jlist = firstneigh[i];
// Usage: j = jlist_of_i[jj];
// jnum - number of J neighbors for each I atom.  jnum = numneigh[i];

void
ACEBEvaluator::compute_atom(int i, DOUBLE_TYPE **x, const SPECIES_TYPE *type, const int jnum, const int *jlist) {
    if (basis_set == nullptr) {
        throw std::invalid_argument("ACEBEvaluator: basis set is not assigned");
    }
    per_atom_calc_timer.start();
    DOUBLE_TYPE evdwl = 0, evdwl_cut = 0, rho_core = 0;
    DOUBLE_TYPE r_norm;
    DOUBLE_TYPE xn, yn, zn, r_xyz;
    DOUBLE_TYPE R, GR, DGR, R_over_r, DR, DCR;
    DOUBLE_TYPE *r_hat;

    SPECIES_TYPE mu_j;
    RANK_TYPE r, rank, t;
    NS_TYPE n;
    LS_TYPE l;
    MS_TYPE m, m_t;

    SPECIES_TYPE *mus;
    NS_TYPE *ns;
    LS_TYPE *ls;
    MS_TYPE *ms;

    int j, jj, func_ind, ms_ind;
    SHORT_INT_TYPE factor;

    ACEComplex Y{0}, Y_DR{0.};
    ACEComplex B{0.};
    ACEComplex dB{0};
    ACEComplex A_cache[basis_set->rankmax];

    dB_flatten.fill({0.});

    ACEDYcomponent grad_phi_nlm{0}, DY{0.};

    //size is +1 of max to avoid out-of-boundary array access in double-triangular scheme
    ACEComplex A_forward_prod[basis_set->rankmax + 1];
    ACEComplex A_backward_prod[basis_set->rankmax + 1];

    DOUBLE_TYPE inv_r_norm;
    DOUBLE_TYPE r_norms[jnum];
    DOUBLE_TYPE inv_r_norms[jnum];
    DOUBLE_TYPE rhats[jnum][3];//normalized vector
    SPECIES_TYPE elements[jnum];
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

    const SHORT_INT_TYPE total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu_i];
    const SHORT_INT_TYPE total_basis_size = basis_set->total_basis_size[mu_i];

    auto basis_rank1 = basis_set->basis_rank1[mu_i];
    auto basis = basis_set->basis[mu_i];

    DOUBLE_TYPE rho_cut, drho_cut, fcut, dfcut;
    DOUBLE_TYPE dF_drho_core;

    //TODO: lmax -> lmaxi (get per-species type)
    const LS_TYPE lmaxi = basis_set->lmax;

    //TODO: nradmax -> nradiali (get per-species type)
    const NS_TYPE nradiali = basis_set->nradmax;

    //TODO: nradbase -> nradbasei (get per-species type)
    const NS_TYPE nradbasei = basis_set->nradbase;

    //TODO: get per-species type number of densities
    const DENSITY_TYPE ndensity = basis_set->map_embedding_specifications[mu_i].ndensity;

    neighbours_forces.resize(jnum, 3);
    neighbours_forces.fill(0);

    //TODO: shift nullifications to place where arrays are used
    weights.fill({0});
    weights_rank1.fill(0);
    A.fill({0});
    A_rank1.fill(0);
    rhos.fill(0);
    dF_drho.fill(0);

#ifdef EXTRA_C_PROJECTIONS
    //TODO: resize arrays or let them be of maximal size?
    projections.init(total_basis_size_rank1 + total_basis_size, "projections");
    projections.fill(0);

    dE_dc.init((total_basis_size_rank1 + total_basis_size) * ndensity, "dE_dc");
    dE_dc.fill(0);
#endif

    //proxy references to spherical harmonics and radial functions arrays
    const Array2DLM<ACEComplex> &ylm = basis_set->spherical_harmonics.ylm;
    const Array2DLM<ACEDYcomponent> &dylm = basis_set->spherical_harmonics.dylm;

    const Array2D<DOUBLE_TYPE> &fr = basis_set->radial_functions->fr;
    const Array2D<DOUBLE_TYPE> &dfr = basis_set->radial_functions->dfr;

    const Array1D<DOUBLE_TYPE> &gr = basis_set->radial_functions->gr;
    const Array1D<DOUBLE_TYPE> &dgr = basis_set->radial_functions->dgr;

    loop_over_neighbour_timer.start();

    int jj_actual = 0;
    SPECIES_TYPE type_j = 0;
    int neighbour_index_mapping[jnum]; // jj_actual -> jj
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

        DOUBLE_TYPE current_cutoff = basis_set->radial_functions->cut(mu_i, mu_j);
        r_xyz = sqrt(xn * xn + yn * yn + zn * zn);

        if (r_xyz >= current_cutoff)
            continue;

        inv_r_norm = 1 / r_xyz;

        r_norms[jj_actual] = r_xyz;
        inv_r_norms[jj_actual] = inv_r_norm;
        rhats[jj_actual][0] = xn * inv_r_norm;
        rhats[jj_actual][1] = yn * inv_r_norm;
        rhats[jj_actual][2] = zn * inv_r_norm;
        elements[jj_actual] = mu_j;
        neighbour_index_mapping[jj_actual] = jj;
        jj_actual++;
    }

    int jnum_actual = jj_actual;

    //ALGORITHM 1: Atomic base A
    for (jj = 0; jj < jnum_actual; ++jj) {
        r_norm = r_norms[jj];
        mu_j = elements[jj];
        r_hat = rhats[jj];

        //proxies
        Array2DLM<ACEComplex> &Y_jj = Y_cache(jj);
        Array2DLM<ACEDYcomponent> &DY_jj = DY_cache(jj);


        basis_set->radial_functions->evaluate(r_norm, basis_set->nradbase, nradiali, mu_i, mu_j);
        basis_set->spherical_harmonics.compute_ylm(r_hat[0], r_hat[1], r_hat[2], lmaxi);
        //loop for computing A's
        //rank = 1
        for (n = 0; n < basis_set->nradbase; n++) {
            GR = gr(n);
#ifdef DEBUG_ENERGY_CALCULATIONS
            printf("-neigh atom %d\n", jj);
            printf("gr(n=%d)(r=%f) = %f\n", n, r_norm, gr(n));
            printf("dgr(n=%d)(r=%f) = %f\n", n, r_norm, dgr(n));
#endif
            DG_cache(jj, n) = dgr(n);
            A_rank1(mu_j, n) += GR * Y00;
        }
        //loop for computing A's
        // for rank > 1
        for (n = 0; n < nradiali; n++) {
            auto &A_lm = A(mu_j, n);
            for (l = 0; l <= lmaxi; l++) {
                R = fr(n, l);
#ifdef DEBUG_ENERGY_CALCULATIONS
                printf("R(nl=%d,%d)(r=%f)=%f\n", n + 1, l, r_norm, R);
#endif

                DR_cache(jj, n, l) = dfr(n, l);
                R_cache(jj, n, l) = R;

                for (m = 0; m <= l; m++) {
                    Y = ylm(l, m);
#ifdef DEBUG_ENERGY_CALCULATIONS
                    printf("Y(lm=%d,%d)=(%f, %f)\n", l, m, Y.real, Y.img);
#endif
                    A_lm(l, m) += R * Y; //accumulation sum over neighbours
                    Y_jj(l, m) = Y;
                    DY_jj(l, m) = dylm(l, m);
                }
            }
        }

        //hard-core repulsion
        rho_core += basis_set->radial_functions->cr;
        DCR_cache(jj) = basis_set->radial_functions->dcr;

    } //end loop over neighbours

    //complex conjugate A's (for NEGATIVE (-m) terms)
    // for rank > 1
    for (mu_j = 0; mu_j < basis_set->nelements; mu_j++) {
        for (n = 0; n < nradiali; n++) {
            auto &A_lm = A(mu_j, n);
            for (l = 0; l <= lmaxi; l++) {
                //fill in -m part in the outer loop using the same m <-> -m symmetry as for Ylm
                for (m = 1; m <= l; m++) {
                    factor = m % 2 == 0 ? 1 : -1;
                    A_lm(l, -m) = A_lm(l, m).conjugated() * factor;
                }
            }
        }
    }    //now A's are constructed
    loop_over_neighbour_timer.stop();

    // ==================== ENERGY ====================

    energy_calc_timer.start();
#ifdef EXTRA_C_PROJECTIONS
    projections.fill(0);
#endif

    //ALGORITHM 2: Basis functions B with iterative product and density rho(p) calculation
    //rank=1
    for (int func_rank1_ind = 0; func_rank1_ind < total_basis_size_rank1; ++func_rank1_ind) {
        auto func = &basis_rank1[func_rank1_ind];
//        ndensity = func->ndensity;
#ifdef PRINT_LOOPS_INDICES
        printf("Num density = %d r = 0\n",(int) ndensity );
        print_C_tilde_B_basis_function(*func);
#endif
        double A_cur = A_rank1(func->mus[0], func->ns[0] - 1);
#ifdef DEBUG_ENERGY_CALCULATIONS
        printf("A_r=1(x=%d, n=%d)=(%f)\n", func->mus[0], func->ns[0], A_cur);
#endif
#ifdef EXTRA_C_PROJECTIONS
        //aggregate C-projections separately
        projections(func_rank1_ind) += A_cur;
#endif
        for (DENSITY_TYPE p = 0; p < ndensity; ++p) {
            //for rank=1 (r=0) only 1 ms-combination exists (ms_ind=0), so index of func.ctildes is 0..ndensity-1
            rhos(p) += func->coeff[p] * A_cur;// * func->gen_cgs[0];
        }
    } // end loop for rank=1

    //rank>1
    int func_ms_ind = 0;
    int func_ms_t_ind = 0;// index for dB

    for (func_ind = 0; func_ind < total_basis_size; ++func_ind) {
        auto func = &basis[func_ind];
        //TODO: check if func->ctildes are zero, then skip
//        ndensity = func->ndensity;
        rank = func->rank;
        r = rank - 1;
#ifdef PRINT_LOOPS_INDICES
        printf("Num density = %d r = %d\n",(int) ndensity, (int)r );
        print_C_tilde_B_basis_function(*func);
#endif
        mus = func->mus;
        ns = func->ns;
        ls = func->ls;

        //loop over {ms} combinations in sum
        for (ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind, ++func_ms_ind) {
            ms = &func->ms_combs[ms_ind * rank]; // current ms-combination (of length = rank)

            //loop over m, collect B  = product of A with given ms
            A_forward_prod[0] = 1;
            A_backward_prod[r] = 1;

            //fill forward A-product triangle
            for (t = 0; t < rank; t++) {
                //TODO: optimize ns[t]-1 -> ns[t] during functions construction
                A_cache[t] = A(mus[t], ns[t] - 1, ls[t], ms[t]);
#ifdef DEBUG_ENERGY_CALCULATIONS
                printf("A(x=%d, n=%d, l=%d, m=%d)=(%f,%f)\n", mus[t], ns[t], ls[t], ms[t], A_cache[t].real,
                       A_cache[t].img);
#endif
                A_forward_prod[t + 1] = A_forward_prod[t] * A_cache[t];
            }

            B = A_forward_prod[t];
#ifdef DEBUG_FORCES_CALCULATIONS
            printf("B = (%f, %f)\n", (B).real, (B).img);
#endif
            //fill backward A-product triangle
            for (t = r; t >= 1; t--) {
                A_backward_prod[t - 1] =
                        A_backward_prod[t] * A_cache[t];
            }

            for (t = 0; t < rank; ++t, ++func_ms_t_ind) {
                dB = A_forward_prod[t] * A_backward_prod[t]; //dB - product of all A's except t-th
                dB_flatten(func_ms_t_ind) = dB;
#ifdef DEBUG_FORCES_CALCULATIONS
                m_t = ms[t];
                printf("dB(n,l,m)(%d,%d,%d) = (%f, %f)\n", ns[t], ls[t], m_t, (dB).real, (dB).img);
#endif
            }
#ifdef EXTRA_C_PROJECTIONS
            //aggregate C-projections separately
            projections(total_basis_size_rank1 + func_ind) += B.real_part_product(func->gen_cgs[ms_ind]);
#endif
            for (DENSITY_TYPE p = 0; p < ndensity; ++p) {
                //real-part only multiplication
                rhos(p) += B.real_part_product(func->gen_cgs[ms_ind] * func->coeff[p]);
#ifdef PRINT_INTERMEDIATE_VALUES
                printf("rhos(%d) += %f\n", p, B.real_part_product(func->ctildes[ms_ind * ndensity + p]));
                printf("Rho[i = %d][p = %d] = %f\n",  i , p , rhos(p));
#endif
            }
        }//end of loop over {ms} combinations in sum
    }// end loop for rank>1

#ifdef DEBUG_FORCES_CALCULATIONS
    printf("rhos = ");
    for(DENSITY_TYPE p =0; p<ndensity; ++p) printf(" %.20f ",rhos(p));
    printf("\n");
#endif


    // energy cutoff
    rho_cut = basis_set->map_embedding_specifications.at(mu_i).rho_core_cutoff;
    drho_cut = basis_set->map_embedding_specifications.at(mu_i).drho_core_cutoff;

    basis_set->inner_cutoff(rho_core, rho_cut, drho_cut, fcut, dfcut);
    basis_set->FS_values_and_derivatives(rhos, evdwl, dF_drho, mu_i);

#ifdef EXTRA_C_PROJECTIONS
    int projections_size = projections.get_size();
    int dE_dc_ind = 0;
    for (DENSITY_TYPE p = 0; p < ndensity; p++) {
        for (int proj_ind = 0; proj_ind < projections_size; proj_ind++, dE_dc_ind++)
            dE_dc(dE_dc_ind) = dF_drho(p) * projections(proj_ind);
    }
#endif
    dF_drho_core = evdwl * dfcut + 1;
    for (DENSITY_TYPE p = 0; p < ndensity; ++p)
        dF_drho(p) *= fcut;
    evdwl_cut = evdwl * fcut + rho_core;

#ifdef DEBUG_FORCES_CALCULATIONS
    printf("dFrhos = ");
    for(DENSITY_TYPE p =0; p<ndensity; ++p) printf(" %f ",dF_drho(p));
    printf("\n");
#endif

    //ALGORITHM 3: Weights and theta calculation
    // rank = 1
    for (int f_ind = 0; f_ind < total_basis_size_rank1; ++f_ind) {
        auto func = &basis_rank1[f_ind];
//        ndensity = func->ndensity;
        for (DENSITY_TYPE p = 0; p < ndensity; ++p) {
            //for rank=1 (r=0) only 1 ms-combination exists (ms_ind=0), so index of func.ctildes is 0..ndensity-1
            weights_rank1(func->mus[0], func->ns[0] - 1) += dF_drho(p) * func->coeff[p];
        }
    }

    // rank>1
    func_ms_ind = 0;
    func_ms_t_ind = 0;// index for dB
    DOUBLE_TYPE theta = 0;
    for (func_ind = 0; func_ind < total_basis_size; ++func_ind) {
        auto func = &basis[func_ind];
//        ndensity = func->ndensity;
        rank = func->rank;
        mus = func->mus;
        ns = func->ns;
        ls = func->ls;
        for (ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind, ++func_ms_ind) {
            ms = &func->ms_combs[ms_ind * rank];
            theta = 0;
            for (DENSITY_TYPE p = 0; p < ndensity; ++p) {
                theta += dF_drho(p) * func->gen_cgs[ms_ind] * func->coeff[p];
#ifdef DEBUG_FORCES_CALCULATIONS
                printf("(p=%d) theta += dF_drho[p] * func.ctildes[ms_ind * ndensity + p] = %f * %f = %f\n",p, dF_drho(p), func->gen_cgs[ms_ind] * func->coeff[p],dF_drho(p)*func->gen_cgs[ms_ind] * func->coeff[p]);
                printf("theta=%f\n",theta);
#endif
            }

            theta *= 0.5; // 0.5 factor due to possible double counting ???
            for (t = 0; t < rank; ++t, ++func_ms_t_ind) {
                m_t = ms[t];
                factor = (m_t % 2 == 0 ? 1 : -1);
                dB = dB_flatten(func_ms_t_ind);
                weights(mus[t], ns[t] - 1, ls[t], m_t) += theta * dB; //Theta_array(func_ms_ind);
                // update -m_t (that could also be positive), because the basis is half_basis
                weights(mus[t], ns[t] - 1, ls[t], -m_t) +=
                        theta * (dB).conjugated() * factor;// Theta_array(func_ms_ind);
#ifdef DEBUG_FORCES_CALCULATIONS
                printf("dB(n,l,m)(%d,%d,%d) = (%f, %f)\n", ns[t], ls[t], m_t, (dB).real, (dB).img);
                printf("theta = %f\n",theta);
                printf("weights(n,l,m)(%d,%d,%d) += (%f, %f)\n", ns[t], ls[t], m_t, (theta * dB * 0.5).real,
                       (theta * dB * 0.5).img);
                printf("weights(n,l,-m)(%d,%d,%d) += (%f, %f)\n", ns[t], ls[t], -m_t,
                       ( theta * (dB).conjugated() * factor * 0.5).real,
                       ( theta * (dB).conjugated() * factor * 0.5).img);
#endif
            }
        }
    }
    energy_calc_timer.stop();

// ==================== FORCES ====================
#ifdef PRINT_MAIN_STEPS
    printf("\nFORCE CALCULATION\n");
    printf("loop over neighbours\n");
#endif

    forces_calc_loop_timer.start();
// loop over neighbour atoms for force calculations
    for (jj = 0; jj < jnum_actual; ++jj) {
        mu_j = elements[jj];
        r_hat = rhats[jj];
        inv_r_norm = inv_r_norms[jj];

        Array2DLM<ACEComplex> &Y_cache_jj = Y_cache(jj);
        Array2DLM<ACEDYcomponent> &DY_cache_jj = DY_cache(jj);

#ifdef PRINT_LOOPS_INDICES
        printf("\nneighbour atom #%d\n", jj);
        printf("rhat = (%f, %f, %f)\n", r_hat[0], r_hat[1], r_hat[2]);
#endif

        forces_calc_neighbour_timer.start();

        f_ji[0] = f_ji[1] = f_ji[2] = 0;

//for rank = 1
        for (n = 0; n < nradbasei; ++n) {
            if (weights_rank1(mu_j, n) == 0)
                continue;
            auto &DG = DG_cache(jj, n);
            DGR = DG * Y00;
            DGR *= weights_rank1(mu_j, n);
#ifdef DEBUG_FORCES_CALCULATIONS
            printf("r=1: (n,l,m)=(%d, 0, 0)\n",n+1);
            printf("\tGR(n=%d, r=%f)=%f\n",n+1,r_norm, gr(n));
            printf("\tDGR(n=%d, r=%f)=%f\n",n+1,r_norm, dgr(n));
            printf("\tdF+=(%f, %f, %f)\n",DGR * r_hat[0], DGR * r_hat[1], DGR * r_hat[2]);
#endif
            f_ji[0] += DGR * r_hat[0];
            f_ji[1] += DGR * r_hat[1];
            f_ji[2] += DGR * r_hat[2];
        }

//for rank > 1
        for (n = 0; n < nradiali; n++) {
            for (l = 0; l <= lmaxi; l++) {
                R_over_r = R_cache(jj, n, l) * inv_r_norm;
                DR = DR_cache(jj, n, l);

                // for m>=0
                for (m = 0; m <= l; m++) {
                    ACEComplex w = weights(mu_j, n, l, m);
                    if (w == 0)
                        continue;
                    //counting for -m cases if m>0
                    if (m > 0) w *= 2;
                    DY = DY_cache_jj(l, m);
                    Y_DR = Y_cache_jj(l, m) * DR;

                    grad_phi_nlm.a[0] = Y_DR * r_hat[0] + DY.a[0] * R_over_r;
                    grad_phi_nlm.a[1] = Y_DR * r_hat[1] + DY.a[1] * R_over_r;
                    grad_phi_nlm.a[2] = Y_DR * r_hat[2] + DY.a[2] * R_over_r;
#ifdef DEBUG_FORCES_CALCULATIONS
                    printf("d_phi(n=%d, l=%d, m=%d) = ((%f,%f), (%f,%f), (%f,%f))\n",n+1,l,m,
                           grad_phi_nlm.a[0].real, grad_phi_nlm.a[0].img,
                           grad_phi_nlm.a[1].real, grad_phi_nlm.a[1].img,
                           grad_phi_nlm.a[2].real, grad_phi_nlm.a[2].img);

                    printf("weights(n,l,m)(%d,%d,%d) = (%f,%f)\n", n+1, l, m,w.real, w.img);
                    //if (m>0) w*=2;
                    printf("dF(n,l,m)(%d, %d, %d) += (%f, %f, %f)\n", n + 1, l, m,
                           w.real_part_product(grad_phi_nlm.a[0]),
                           w.real_part_product(grad_phi_nlm.a[1]),
                           w.real_part_product(grad_phi_nlm.a[2])
                    );
#endif
// real-part multiplication only
                    f_ji[0] += w.real_part_product(grad_phi_nlm.a[0]);
                    f_ji[1] += w.real_part_product(grad_phi_nlm.a[1]);
                    f_ji[2] += w.real_part_product(grad_phi_nlm.a[2]);
                }
            }
        }


#ifdef PRINT_INTERMEDIATE_VALUES
        printf("f_ji(jj=%d, i=%d)=(%f, %f, %f)\n", jj, i,
               f_ji[0], f_ji[1], f_ji[2]
        );
#endif

        //hard-core repulsion
        DCR = DCR_cache(jj);
#ifdef   DEBUG_FORCES_CALCULATIONS
        printf("DCR = %f\n",DCR);
#endif
        f_ji[0] += dF_drho_core * DCR * r_hat[0];
        f_ji[1] += dF_drho_core * DCR * r_hat[1];
        f_ji[2] += dF_drho_core * DCR * r_hat[2];
#ifdef PRINT_INTERMEDIATE_VALUES
        printf("with core-repulsion\n");
        printf("f_ji(jj=%d, i=%d)=(%f, %f, %f)\n", jj, i,
               f_ji[0], f_ji[1], f_ji[2]
        );
        printf("neighbour_index_mapping[jj=%d]=%d\n",jj,neighbour_index_mapping[jj]);
#endif

        neighbours_forces(neighbour_index_mapping[jj], 0) = f_ji[0];
        neighbours_forces(neighbour_index_mapping[jj], 1) = f_ji[1];
        neighbours_forces(neighbour_index_mapping[jj], 2) = f_ji[2];

        forces_calc_neighbour_timer.stop();
    }// end loop over neighbour atoms for forces

    forces_calc_loop_timer.stop();

    //now, energies and forces are ready
    //energies(i) = evdwl + rho_core;
    e_atom = evdwl_cut;
#ifdef EXTRA_C_PROJECTIONS
    //check if active set is loaded
    // use dE_dc or projections as asi_vector
    Array1D<DOUBLE_TYPE> &asi_vector = (is_linear_extrapolation_grade ? projections : dE_dc);
    if (A_active_set_inv.find(mu_i) != A_active_set_inv.end()) {
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
#endif

#ifdef PRINT_INTERMEDIATE_VALUES
    printf("energies(i) = FS(...rho_p_accum...) = %f\n", evdwl);
#endif
    per_atom_calc_timer.stop();
}

void ACEBEvaluator::load_active_set(const string &asi_filename, bool is_linear, bool is_auto_determine) {
    //load the entire npz file
    cnpy::npz_t asi_npz = cnpy::npz_load(asi_filename);
    if (asi_npz.size() != this->basis_set->nelements) {
        stringstream ss;
        ss << "Number of species types in ASI `" << asi_filename << "` (" << asi_npz.size() << ")";
        ss << "not equal to number of species in ACEBBassiSet (" << this->basis_set->nelements << ")";
        throw std::runtime_error(ss.str());
    }

    // auto-determine is_linear_extrapolation
    if (is_auto_determine) {
        vector<bool> linear_extrapolation_flag_vector(basis_set->nelements);
        for (auto &kv: asi_npz) {
            auto element_name = kv.first;
            SPECIES_TYPE st = basis_set->elements_to_index_map.at(element_name);
            auto shape = kv.second.shape;
            // auto_determine extrapolation grade type: linear or non-linear
            validate_ASI_square_shape(st, shape);
            int number_of_functions = basis_set->total_basis_size_rank1[st] + basis_set->total_basis_size[st];
            int ndensity = basis_set->map_embedding_specifications[st].ndensity;

            if (shape.at(0) == number_of_functions) {
                linear_extrapolation_flag_vector.at(st) = true;
            } else if (shape.at(0) == number_of_functions * ndensity) {
                linear_extrapolation_flag_vector.at(st) = false;
            } else {
                stringstream ss;
                ss << "Active Set Inverted for element `" << element_name << "`:";
                ss << "expected size " << number_of_functions << " (linear) or " << number_of_functions * ndensity
                   << " (nonlinear) , but has size " << shape.at(0);
                throw runtime_error(ss.str());
            }
        }
        if (!equal(linear_extrapolation_flag_vector.begin() + 1, linear_extrapolation_flag_vector.end(),
                   linear_extrapolation_flag_vector.begin())) {
            stringstream ss;
            ss
                    << "Active Set Inverted: could not determine extrapolation type (linear or non-linear) automatically, because it differs for different elements";
            throw runtime_error(ss.str());
        }

        is_linear_extrapolation_grade = linear_extrapolation_flag_vector.at(0);
    } else {
        is_linear_extrapolation_grade = is_linear;
    }


    for (auto &kv: asi_npz) {
        auto element_name = kv.first;
        SPECIES_TYPE st = this->basis_set->elements_to_index_map.at(element_name);
        auto shape = kv.second.shape;
        // auto_determine extrapolation grade type: linear or non-linear
        validate_ASI_square_shape(st, shape);
        validate_ASI_shape(element_name, st, shape);

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
    resize_projections();
}

void ACEBEvaluator::validate_ASI_shape(const string &element_name, SPECIES_TYPE st,
                                       const vector<size_t> &shape) {
    //check that array shape corresponds to number of projections (linear case) or number of projections * ndensity
    int expected_ASI_size;
    int number_of_functions = basis_set->total_basis_size_rank1[st] + basis_set->total_basis_size[st];
    if (is_linear_extrapolation_grade)
        expected_ASI_size = number_of_functions;
    else
        expected_ASI_size = number_of_functions * basis_set->map_embedding_specifications[st].ndensity;
    if (expected_ASI_size != shape.at(0)) {
        stringstream ss;
        ss << "Active Set Inverted for element `" << element_name << "`:";
        ss << "expected shape: (" << expected_ASI_size << ", " << expected_ASI_size << ") , but has shape ("
           << shape.at(0) << ", " << shape.at(1) << ")";
        throw runtime_error(ss.str());
    }
}

void ACEBEvaluator::validate_ASI_square_shape(SPECIES_TYPE st, const vector<size_t> &shape) {
    // check for square shape of ASI
    if (shape.at(0) != shape.at(1)) {
        stringstream ss;
        string element_name = this->basis_set->elements_name[st];
        ss << "Active Set Inverted for element `" << element_name << "`:";
        ss << "should be square matrix, but has shape (" << shape.at(0) << ", " << shape.at(1) << ")";
        throw runtime_error(ss.str());
    }
}

//TODO: add is_linear and is_auto_determine options
void ACEBEvaluator::set_active_set(const vector<vector<vector<DOUBLE_TYPE>>> &species_type_active_set_inv) {
    for (SPECIES_TYPE mu = 0; mu < species_type_active_set_inv.size(); mu++) {
        const auto &active_set = species_type_active_set_inv.at(mu);
        Array2D<DOUBLE_TYPE> A0_inv(active_set.size(), active_set.at(0).size());
        validate_ASI_square_shape(mu, A0_inv.get_shape());
        A0_inv.set_vector(active_set);
        //transpose matrix to speed-up vec-mat multiplication
        Array2D<DOUBLE_TYPE> A0_inv_transpose(A0_inv.get_dim(1), A0_inv.get_dim(0));

        for (int i = 0; i < A0_inv.get_dim(0); i++)
            for (int j = 0; j < A0_inv.get_dim(1); j++)
                A0_inv_transpose(j, i) = A0_inv(i, j);

        this->A_active_set_inv[mu] = A0_inv_transpose;
    }
    resize_projections();
}

void ACEBEvaluator::resize_projections() {
    // find the maximal basis size per element and resize projections array correspondingly
    size_t max_basis_size = 0; // include rank1 + rank>1
    for (SPECIES_TYPE mu = 0; mu < basis_set->nelements; mu++) {
        size_t curr_basis_size = this->basis_set->total_basis_size_rank1[mu] + this->basis_set->total_basis_size[mu];
        if (curr_basis_size > max_basis_size) {
            max_basis_size = curr_basis_size;
        }
    }

    this->projections.resize(max_basis_size);
}

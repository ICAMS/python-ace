//
// Created by Yury Lysogorskiy on 26.02.20.
//
#include "ace_b_basis.h"

int main(int argc, char *argv[]) {
    string yaml_filename, python_filename;

    if (argc < 2) {
        printf("Usage: %s <potential_file.yaml> [<potential_file_for_python.dat>]\n", argv[0]);
        return 0;
    } else if (argc == 2) {
        yaml_filename = argv[1];
        python_filename = yaml_filename.substr(0, yaml_filename.find_last_of('.')) + ".dat";
    } else if (argc >= 3) {
        yaml_filename = argv[1];
        python_filename = argv[2];
    }
    printf("B-basis expansion for Python utility\n");
    printf("B-basis file: %s\n", yaml_filename.c_str());
    printf("Expanded basis file: %s\n", python_filename.c_str());

    ACEBBasisSet basis;
    basis.load(yaml_filename);


    printf("Basis set writing \n");
    FILE *fout = fopen(python_filename.c_str(), "w");
    fprintf(fout, "#crad: mu_i, mu_j, n,l,k, crad^(mui,muj)_(n,l,k)\n");
    for (SPECIES_TYPE ele_i = 0; ele_i < basis.nelements; ele_i++)
        for (SPECIES_TYPE ele_j = 0; ele_j < basis.nelements; ele_j++)
            for (NS_TYPE n = 0; n < basis.nradmax; n++)
                for (LS_TYPE l = 0; l <= basis.lmax; l++)
                    for (NS_TYPE k = 0; k < basis.nradbase; k++) {
                        fprintf(fout, "%d %d %d %d %d %.20f\n", ele_i, ele_j, n + 1, l, k + 1,
                                basis.radial_functions->crad(ele_i, ele_j, n, l, k));
                    }
    fprintf(fout, "# end crad\n");
    fprintf(fout, "# mu0, func_ind, mind, (mu, n, l, m)(1), ..., (mu, n, l, m)(r), GenCG, coeff(1),...coeff(ndens)\n");
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        fprintf(fout, "# another species\n");
        int func_ind = 0;
        const auto &subbasis = basis.mu0_bbasis_vector[mu];

        fprintf(fout, "# another rank\n");
        for (const auto &func: subbasis) {
            //print mu(1), func_ind, mus(r), ns(r), ls(r), ms(r), CG(1), coeff(ndens)
            printf("func #%d\n", func_ind);
            func.print();
            for (int m_ind = 0; m_ind < func.num_ms_combs; ++m_ind) {
                //mu0(1), func_ind(1)
                fprintf(fout, "mu0=%d\tind=%d\t", func.mu0, func_ind);
                //m_ind
                fprintf(fout, "mind=%d\t", m_ind);
                //mus(r), ns(r), ls(r), ms(r)
                for (RANK_TYPE r = 0; r < func.rank; r++)
                    fprintf(fout, "( %d %d %d %d )\t", func.mus[r], func.ns[r], func.ls[r],
                            func.ms_combs[m_ind * func.rank + r]);
                //fprintf(fout, "mu(%d)=%d n(%d)=%d l=%d m=%d ", func.mus[r], func.ns[r], func.ls[r], func.ms[m_ind * func.rank + r]);

                //CG(1)
                fprintf(fout, "cg=%.20f\t", func.gen_cgs[m_ind]);

                //coeff(ndens)
                DENSITY_TYPE p = 0;
                for (p = 0; p < func.ndensity - 1; ++p)
                    fprintf(fout, "coef=%.20f\t", func.coeff[p]);
                fprintf(fout, "coef=%.20f", func.coeff[p]);
                fprintf(fout, "\n");
            }
            func_ind++;
        }

    }
    fclose(fout);

    printf("Done.\n");

    return 0;
}

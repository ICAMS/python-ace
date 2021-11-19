//
// Created by Yury Lysogorskiy on 26.02.20.
//
#include "ace_b_basis.h"

int main(int argc, char *argv[]) {
    string yaml_filename, ace_filename;

    if (argc < 2) {
        printf("Usage: %s <potential_file.yaml> [<potential_file.ace>]\n", argv[0]);
        return 0;
    } else if (argc == 2) {
        yaml_filename = argv[1];
        ace_filename = yaml_filename.substr(0, yaml_filename.find_last_of('.')) + ".ace";
    } else if (argc >= 3) {
        yaml_filename = argv[1];
        ace_filename = argv[2];
    }
    printf("B- to c-tilde basis conversion utility\n");
    printf("B-basis file: %s\n", yaml_filename.c_str());
    printf("C-basis file: %s\n", ace_filename.c_str());

    ACEBBasisSet basis;
    basis.load(yaml_filename);
    printf("Basis set:\n");
    for (SPECIES_TYPE mu = 0; mu < basis.nelements; mu++) {
        printf("Element: %s, rank=1: %d B-functions, rank>1: %d B-functions\n",
               basis.elements_name[mu].c_str(),
               basis.total_basis_size_rank1[mu],
               basis.total_basis_size[mu]);
    }
    ACECTildeBasisSet cbasis = basis.to_ACECTildeBasisSet();
    cbasis.save(ace_filename);

    printf("Done.\n");

    return 0;
}

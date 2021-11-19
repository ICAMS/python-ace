import argparse
from collections import defaultdict

from pyace import ACEBBasisSet, BBasisConfiguration


def analyse_potential_shape(ace_potential):
    if isinstance(ace_potential, str):
        bbasis = ACEBBasisSet(ace_potential)
    elif isinstance(ace_potential, BBasisConfiguration):
        bbasis = ACEBBasisSet(ace_potential)
    elif isinstance(ace_potential, ACEBBasisSet):
        bbasis = ace_potential
    else:
        raise ValueError("Unsupported format for 'ace_potential' argument ({}), ".format(type(ace_potential)) + \
                         "must be .yaml filename, BBasisConfiguration or ACEBBasisSet")

    elements_name = bbasis.elements_name
    map_embedding_specifications = bbasis.map_embedding_specifications
    map_bond_specifications = bbasis.map_bond_specifications

    print("Available element(s):", " ".join(elements_name))
    print("=" * 80)
    print(" " * 10, "Embeddings")
    print("=" * 80)
    print("Element\tndens\tFSfunc\t\t\tFS_params")
    print("=" * 80)
    tot_n_func = 0
    for i, el in enumerate(elements_name):
        num_r1 = len(bbasis.basis_rank1[i])
        num_r = len(bbasis.basis[i])
        embedding_specifications = map_embedding_specifications[i]
        print("{:3}\t{}\t{}\t{}".format(el,
                                        embedding_specifications.ndensity,
                                        embedding_specifications.npoti,
                                        embedding_specifications.FS_parameters
                                        ))
        tot_n_func += num_r1 + num_r

    print("=" * 80)
    print("Total number of functions: ", tot_n_func)
    print()

    rcuts = {(elements_name[mu_i], elements_name[mu_j]): bond_spec.rcut for (mu_i, mu_j), bond_spec in
             map_bond_specifications.items()}
    rcut_max = max(rcuts.values())

    print("Maximum cutoff: ", rcut_max, "Ang")

    print("=" * 40)
    print(" " * 10, "Bonds")
    print("=" * 40)
    print("Bond\tradbasename\trcut\tr_in\tdelta_in prehc\tlambdahc")
    print("=" * 40)
    for (mu_i, mu_j), bond_spec in map_bond_specifications.items():
        el_i = elements_name[mu_i]
        el_j = elements_name[mu_j]
        rc = bond_spec.rcut
        print("{}-{}".format(el_i, el_j),
              "\t", bond_spec.radbasename,
              "\t", rc,
              "\t", bond_spec.rcut_in,
              "\t", bond_spec.dcut_in,
              "\t", bond_spec.prehc,
              "\t", bond_spec.lambdahc,
              )

    print("=" * 40)

    print("\n")
    print("=" * 40)
    print(" " * 10, "Functions (per order)")
    print("=" * 40)
    for i, el in enumerate(bbasis.elements_name):
        nradmax_dd = defaultdict(lambda: 0)  # [order] -> ns
        lmax_dd = defaultdict(lambda: 0)  # [order] -> ls
        nfuncs_dd = defaultdict(lambda: 0)  # [order] -> nfuncs

        # order 1 - determine nradbase
        nradbase = max([max(func.ns) for func in bbasis.basis_rank1[i]])

        nradmax_dd[1] = nradbase
        nfuncs_dd[1] = len(bbasis.basis_rank1[i])
        max_r = 0
        # order > 1 - determine nradmax, lmax
        for func in bbasis.basis[i]:
            r = func.rank
            nradmax_dd[r] = max(nradmax_dd[r], max(func.ns))
            lmax_dd[r] = max(lmax_dd[r], max(func.ls))
            max_r = max(max_r, r)
            nfuncs_dd[r] += 1

        # element
        print(el)
        print("order  :", end="\t")
        for r in range(1, max_r + 1):
            print(r, end="\t")
        print()

        # nradmax per order
        print("nradmax:", end="\t")
        for r in range(1, max_r + 1):
            print(nradmax_dd[r], end="\t")
        print()

        # lmax per order
        print("lmax   :", end="\t")
        for r in range(1, max_r + 1):
            print(lmax_dd[r], end="\t")
        print()

        # nfuncs per order
        print("nfuncs :", end="\t")
        for r in range(1, max_r + 1):
            print(nfuncs_dd[r], end="\t")
        print("sum=", sum(nfuncs_dd.values()))

        print()
        print("-" * 20)
    print("=" * 40)


parser = argparse.ArgumentParser(prog="pace_info",
                                 description="Utility to analyze PACE (.yaml) potential shape and other parameters")

parser.add_argument("potential_file", help="B-basis file name (.yaml)", type=str, nargs='+', default=[])

args_parse = parser.parse_args()
potential_files = args_parse.potential_file

for potential_file in potential_files:
    # print("potential file: ", potential_file)
    analyse_potential_shape(potential_file)

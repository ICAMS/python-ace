import numpy as np
import pandas as pd
import timeit

from ase.build import bulk
from pyace import *


def run_timing_test(ace_potential, n_struct=10, n_iter=100, atoms_list = None, verbose=True):
    if isinstance(ace_potential, str):
        bbasis = ACEBBasisSet(ace_potential)
    elif  isinstance(ace_potential, BBasisConfiguration):
        bbasis = ACEBBasisSet(ace_potential)
    elif isinstance(ace_potential, ACEBBasisSet):
        bbasis = ace_potential
    else:
        raise ValueError("Unsupported format for 'ace_potential' argument ({}), ".format(type(ace_potential)) + \
                         "must be .yaml filename, BBasisConfiguration or ACEBBasisSet")

    if verbose:
        print("Available element(s):", " ".join(bbasis.elements_name))

        print("=" * 60)
        print("Element\t\t# funcs. order=1\t# funcs. order>1")
        print("=" * 60)
        for num_r1, num_r, el in zip(map(len, bbasis.basis_rank1), map(len, bbasis.basis), bbasis.elements_name):
            print("{:3}\t\t{:5}\t\t\t{:5}".format(el, num_r1, num_r))
        print("=" * 60)
        print()

    rcuts = [bond_spec.rcut for _, bond_spec in bbasis.map_bond_specifications.items()]

    rcut = max(rcuts)

    if verbose:
        print("Maximum cutoff: ", rcut, "Ang")

    protos = []

    if atoms_list is None:
        atoms_list = []
        for _ in range(n_struct):
            proto = np.random.choice(["fcc", "bcc"])
            atoms = bulk("Al", proto,
                         a=np.random.uniform(low=3.5, high=4.5),
                         cubic=True) * (2, 2, 2)

            pos = atoms.get_positions()
            dpos = np.random.randn(*np.shape(pos)) * 0.1
            atoms.set_positions(pos + dpos)
            chemsymb = np.random.choice(bbasis.elements_name, size=len(atoms))
            atoms.set_chemical_symbols(chemsymb)
            atoms_list.append(atoms)
            protos.append(proto)
    else:
        for atoms in atoms_list:
            protos.append(atoms.get_chemical_formula())

    ae_list = []

    for atoms in atoms_list:
        ae = aseatoms_to_atomicenvironment(atoms, elements_mapper_dict=bbasis.elements_to_index_map, cutoff=rcut)
        ae_list.append(ae)

    n_struct = len(atoms_list)

    mean_neighbour_per_atom = []
    for ae in ae_list:
        mean_neighbour_per_atom.append(np.mean(list(map(len, ae.neighbour_list))))

    beval = ACEBEvaluator()
    beval.set_basis(bbasis)
    calc = ACECalculator()
    calc.set_evaluator(beval)

    times = []
    if verbose:
        print("Running tests ({} strctures):".format(len(ae_list)))
    for i, ae in enumerate(ae_list):
        time_res = timeit.timeit(stmt="calc.compute(ae)", globals={"calc": calc, "ae": ae}, number=n_iter)
        times.append(time_res / n_iter)
        if verbose:
            print("\t{}/{}: {:.0f} mcs/atom".format(i + 1, n_struct,
                                                    time_res / n_iter / ae.n_atoms_real * 1e6)+" "*40, end="\r")

    n_atom_list = np.array([len(atoms) for atoms in atoms_list])

    times = np.array(times)

    time_per_atom = times / n_atom_list

    res_df = pd.DataFrame({"Prototype": protos,
                           "n_atoms": n_atom_list,
                           "n_neigh": mean_neighbour_per_atom,
                           "time_per_atom": time_per_atom * 1e6  # Time, mcs/atom
                           })

    gdf = res_df.groupby("Prototype").agg(["mean", "std"])

    if verbose:
        print("=" * 80)
        print("Prototype\tNum. of atoms\tNeighbours per atom\t Time per atom (mcs/atom)")
        print("=" * 80)
        for proto, row in gdf.iterrows():
            print(proto, end="\t\t")
            print("{: <4}".format(row[("n_atoms", "mean")]), end="\t\t")
            print("{:6.2f} (+/-{:6.2f})".format(row[("n_neigh", "mean")], row[("n_neigh", "std")]), end="\t\t")
            print("{:3.0f} (+/-{:2.0f})".format(row[("time_per_atom", "mean")], row[("time_per_atom", "std")]))
        print("=" * 80)

    tpa_mean, tpa_std = gdf[("time_per_atom", "mean")].mean(), gdf[("time_per_atom", "mean")].std()

    if verbose:
        print("Overall timing (mcs/atom):\t {:.0f} (+/- {:.0f}) ".format(tpa_mean, tpa_std))

    return res_df
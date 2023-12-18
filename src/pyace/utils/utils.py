import logging

from ase.data import covalent_radii, atomic_numbers
from pyace.atomicenvironment import aseatoms_to_atomicenvironment


def compute_nn_dist_per_bond(atoms, cutoff=3, elements_mapper_dict=None):
    ae = aseatoms_to_atomicenvironment(atoms, cutoff, elements_mapper_dict=elements_mapper_dict)
    return ae.get_minimal_nn_distance_per_bond()


def get_vpa(atoms):
    try:
        return atoms.get_volume() / len(atoms)
    except ValueError as e:
        return 0


def complement_min_dist_dict(min_dist_per_bond_dict, bond_quantile_dict, elements, verbose):
    res = min_dist_per_bond_dict.copy()
    for mu_i in range(len(elements)):
        for mu_j in range(mu_i, len(elements)):
            k = tuple(sorted([mu_i, mu_j]))
            if k not in res:
                if k in bond_quantile_dict:
                    res[k] = bond_quantile_dict[k]
                else:
                    # get some default values  as covalent radii
                    # covalent_radii[1] for H, ...
                    z_i = atomic_numbers[elements[mu_i]]
                    z_j = atomic_numbers[elements[mu_j]]
                    r_in = covalent_radii[z_i] + covalent_radii[z_j]
                    if verbose:
                        logging.warning(f"No minimal distance for bond {k}, using sum of covalent  radii: {r_in:.3f}")
                    res[k] = r_in
    return res

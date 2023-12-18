import logging

import pandas as pd

import numpy as np

from collections import Counter, defaultdict
from scipy.spatial import ConvexHull
from pyace.const import *

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(seq, *args, **kwargs):
        return seq

SINGLE_ATOM_ENERGY_DICT = {
    # energy computed with default VASP_NM settings (500eV, Gaussian smearing, sigma=0.1, min 10x10x10 A cell, single atom, NM)
    "VASP_PBE_500_0.125_0.1_NM":
        {'Ac': -0.20899015,
         'Ag': -0.06689862,
         'Al': -0.10977253,
         'Ar': -0.0235748,
         'As': -0.17915798,
         'Au': -0.06427594,
         'B': -0.1100114,
         'Ba': -0.13720005,
         'Be': -0.04039112,
         'Bi': -0.1854587,
         'Br': -0.12188631,
         'C': -0.1612024,
         'Ca': -0.02585929,
         'Cd': -0.01423875,
         'Ce': -0.89994905,
         'Cl': -0.12414702,
         'Co': -0.27078113,
         'Cr': -0.65569845,
         'Cs': -0.23242861,
         'Cu': -0.06659418,
         'Dy': -0.15964479,
         'Er': -0.1606716,
         'Eu': -0.05150557,
         'F': -0.11863189,
         'Fe': -0.35795312,
         'Fr': -0.20499698,
         'Ga': -0.10956489,
         'Gd': -0.16319873,
         'Ge': -0.16372306,
         'H': -0.05660531,
         'He': 0.00161163,
         'Hf': -2.79098326,
         'Hg': -0.01069226,
         'Ho': -0.15904603,
         'I': -0.11835143,
         'In': -0.12375132,
         'Ir': -0.27663991,
         'K': -0.07961456,
         'Kr': -0.02228048,
         'La': -0.40271396,
         'Li': -0.06085926,
         'Lu': -0.16197678,
         'Mg': -0.00056567,
         'Mn': -0.47998212,
         'Mo': -0.4297273,
         'N': -0.17273379,
         'Na': -0.06438759,
         'Nb': -0.82269379,
         'Nd': -0.18393967,
         'Ne': -0.01244429,
         'Ni': -0.18976919,
         'Np': -4.40174385,
         'O': -0.16665686,
         'Os': -0.60389947,
         'P': -0.18102125,
         'Pa': -1.40133833,
         'Pb': -0.16754695,
         'Pd': -1.47571017,
         'Pm': -0.17541143,
         'Po': -0.17215562,
         'Pt': -0.25488222,
         'Pu': -6.19533094,
         'Ra': -0.10539559,
         'Rb': -0.07907932,
         'Re': -1.20230599,
         'Rh': -1.03428996,
         'Rn': -0.00530062,
         'Ru': -0.60938402,
         'S': -0.16942446,
         'Sb': -0.17675287,
         'Sc': -1.82317913,
         'Se': -0.1669939,
         'Si': -0.16208649,
         'Sm': -0.17184578,
         'Sn': -0.17573341,
         'Sr': -0.02800043,
         'Ta': -2.3423585,
         'Tb': -0.16170736,
         'Tc': -0.36811991,
         'Te': -0.16457883,
         'Th': -0.65883239,
         'Ti': -1.35069169,
         'Tl': -0.11645191,
         'Tm': -0.16111118,
         'U': -2.73254914,
         'V': -0.93934306,
         'W': -1.81667003,
         'Xe': -0.01207016,
         'Y': -1.98890231,
         'Yb': -0.02974997,
         'Zn': -0.0113131,
         'Zr': -1.43289387}
}


def compdict_to_comptuple(comp_dict):
    n_atoms = sum([v for v in comp_dict.values()])
    return tuple(sorted([(k, v / n_atoms) for k, v in comp_dict.items()]))


def comptuple_to_str(comp_tuple):
    return " ".join(("{}_{:.3f}".format(e, c) for e, c in comp_tuple))


def compute_compositions(df: pd.DataFrame, ase_atoms_column=ASE_ATOMS, compute_composition_tuples=True):
    """
    Generate new columns:
       'comp_dict' - composition dictionary
       'n_atom' - number of atoms
       'n_'+Elements, 'c_'+Elements - number and concentration of elements
    """
    df[COMP_DICT] = df[ase_atoms_column].map(lambda atoms: Counter(atoms.get_chemical_symbols()))
    df[NUMBER_OF_ATOMS] = df[ase_atoms_column].map(len)

    if compute_composition_tuples:
        df[COMP_TUPLE] = df[COMP_DICT].map(compdict_to_comptuple)

    elements = extract_elements(df)

    for el in elements:
        df["n_" + el] = df[COMP_DICT].map(lambda d: d.get(el, 0))
        df["c_" + el] = df["n_" + el] / df[NUMBER_OF_ATOMS]

    return elements


def extract_elements(df: pd.DataFrame, composition_dict_column=COMP_DICT):
    elements_set = set()
    for cd in df[composition_dict_column]:
        elements_set.update(cd.keys())
    elements = sorted(elements_set)
    return elements


def compute_formation_energy(df: pd.DataFrame, elements=None, epa_gs_dict=None,
                             energy_per_atom_column='energy_per_atom',
                             verbose=True):
    if elements is None:
        elements = extract_elements(df)

    c_elements = ["c_" + el for el in elements]

    if epa_gs_dict is None:
        epa_gs_dict = {}
        for el in elements:
            subdf = df[df["c_" + el] == 1.0]
            if len(subdf) > 0:
                e_min_pa = subdf[energy_per_atom_column].min()
            else:
                e_min_pa = 0.0
                if verbose:
                    print("No pure element energy for {} is available, assuming 0  eV/atom".format(el))
            epa_gs_dict[el] = e_min_pa
    element_emin_array = np.array([epa_gs_dict[el] for el in elements])
    c_conc = df[c_elements].values
    e_formation_ideal = np.dot(c_conc, element_emin_array)
    df[E_FORMATION_PER_ATOM] = df[energy_per_atom_column] - e_formation_ideal


# TODO: write tests
def compute_convexhull_dist(df: pd.DataFrame,
                            ase_atoms_column=ASE_ATOMS,
                            energy_per_atom_column='energy_per_atom',
                            verbose=True):
    """
    df: pd.DataFrame with ASE atoms column and energy-per-atom column
    ase_atoms_column: (str) name of ASE atoms column
    energy_per_atom_column: (str) name of energy-per-atom column

    return: list of elements (str)

    construct new columns to dataframe:
     'comp_dict': composition dictionary
     'n_'+element, 'c_'+element - number and concentration of elements
     'e_formation_per_atom': formation energy per atom
     'e_chull_dist_per_atom': distance to convex hull
    """
    elements = compute_compositions(df, ase_atoms_column=ase_atoms_column)
    c_elements = ["c_" + el for el in elements]

    compute_formation_energy(df, elements, energy_per_atom_column=energy_per_atom_column, verbose=verbose)

    # check if more than one unique compositions
    uniq_compositions = df[COMP_TUPLE].unique()
    # df.drop(columns=["comp_tuple"], inplace=True)

    if len(uniq_compositions) > 1:
        if verbose:
            print("Structure dataset: multiple unique compositions found, trying to construct convex hull")
        chull_values = df[c_elements[:-1] + [E_FORMATION_PER_ATOM]].values
        hull = ConvexHull(chull_values)
        ok = hull.equations[:, -2] < 0
        selected_simplices = hull.simplices[ok]
        selected_equations = hull.equations[ok]

        norms = selected_equations[:, :-1]
        offsets = selected_equations[:, -1]

        norms_c = norms[:, :-1]
        norms_e = norms[:, -1]

        e_chull_dist_list = []
        for p in chull_values:
            p_c = p[:-1]
            p_e = p[-1]
            e_simplex_projections = []
            for nc, ne, b, simplex in zip(norms_c, norms_e, offsets, selected_simplices):
                if ne != 0:
                    e_simplex = (-b - np.dot(nc, p_c)) / ne
                    e_simplex_projections.append(e_simplex)
                elif np.abs(b + np.dot(nc, p_c)) < 2e-15:  # ne*e_simplex + b + np.dot(nc,p_c), ne==0
                    e_simplex = p_e
                    e_simplex_projections.append(e_simplex)

            e_simplex_projections = np.array(e_simplex_projections)

            mask = e_simplex_projections < p_e + 1e-15

            e_simplex_projections = e_simplex_projections[mask]

            e_dist_to_chull = np.min(p_e - e_simplex_projections)

            e_chull_dist_list.append(e_dist_to_chull)

        e_chull_dist_list = np.array(e_chull_dist_list)
    else:
        if verbose:
            logging.info(
                "Structure dataset: only single unique composition found, switching to cohesive energy reference")
        emin = df[energy_per_atom_column].min()
        e_chull_dist_list = df[energy_per_atom_column] - emin

    df[E_CHULL_DIST_PER_ATOM] = e_chull_dist_list
    return elements


def compute_corrected_energy(df: pd.DataFrame, esa_dict=None, calculator_name='VASP_PBE_500_0.125_0.1_NM',
                             n_atoms_column=NUMBER_OF_ATOMS):
    elements = compute_compositions(df)
    n_elements = ["n_" + e for e in elements]
    if esa_dict is None:
        esa_dict = {e: SINGLE_ATOM_ENERGY_DICT[calculator_name][e] for e in elements}
    esa_array = np.array([esa_dict.get(e, 0) for e in elements])
    corr_mask = ~df[ENERGY].isna()
    df.loc[corr_mask, ENERGY_CORRECTED_COL] = df.loc[corr_mask, ENERGY] - (
            df.loc[corr_mask, n_elements] * esa_array).sum(axis=1)
    e_corr_shift = esa_dict.get("shift", 0)
    df[ENERGY_CORRECTED_COL] += e_corr_shift * df[n_atoms_column]
    df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df[n_atoms_column]
    return esa_dict


def compute_shifted_scaled_corrected_energy(df: pd.DataFrame, n_atoms_column=NUMBER_OF_ATOMS):
    elements = compute_compositions(df)
    n_elements = ["n_" + e for e in elements]
    corr_mask = ~df[ENERGY].isna()
    comp_array = df.loc[corr_mask, n_elements].values
    e_array = df.loc[corr_mask, ENERGY].values
    assert not np.any(np.isnan(comp_array)), "Compositions columns contain NaN"
    assert not np.any(np.isnan(e_array)), "Energy column contain NaN"
    esa_array = np.linalg.pinv(comp_array, rcond=1e-10) @ e_array  # solve equation
    esa_dict = {e: esa for e, esa in zip(elements, esa_array)}
    df.loc[corr_mask, ENERGY_CORRECTED_COL] = e_array - np.dot(comp_array, esa_array)
    df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df[n_atoms_column]

    def safe_get_volume_per_atom(at):
        try:
            return at.get_volume() / len(at)
        except Exception as e:
            return 0

    df["volume_per_atom"] = df[ASE_ATOMS].map(safe_get_volume_per_atom)
    max_vpa = df["volume_per_atom"].max()
    e_corr_shift = 0
    if max_vpa > 0:
        # "-", becese +=shift
        e_corr_shift = -df[df["volume_per_atom"] >= max_vpa].iloc[0][E_CORRECTED_PER_ATOM_COLUMN]
    else:  # max_vpa==0
        e_corr_shift = 0

    df[ENERGY_CORRECTED_COL] += e_corr_shift * df[n_atoms_column]
    esa_dict["shift"] = e_corr_shift
    return esa_dict

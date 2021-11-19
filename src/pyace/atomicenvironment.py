import numpy as np
# get everything from atomic environment
import pandas as pd
from pyace.catomicenvironment import ACEAtomicEnvironment, get_minimal_nn_distance_tp, build_atomic_env
from pyace.catomicenvironment import get_nghbrs_tp_atoms
from pyace.pyneighbor import ACENeighborList
import warnings
from ase import Atoms
from ase import data


def create_cube(dr, cube_side_length):
    """
    Create a simple cube without pbc. Test function
    """
    warnings.warn("This function is retained for testing purposes only")
    posx = np.arange(-cube_side_length / 2, cube_side_length / 2 + dr / 2, dr)
    posy = np.arange(-cube_side_length / 2, cube_side_length / 2 + dr / 2, dr)
    posz = np.arange(-cube_side_length / 2, cube_side_length / 2 + dr / 2, dr)
    n_atoms = len(posx) * len(posy) * len(posz)

    positions = []
    for x in posx:
        for y in posy:
            for z in posz:
                positions.append([x, y, z])

    atoms = Atoms(positions=positions, symbols=["W"] * len(positions))
    nl = ACENeighborList()
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    return ae


def create_linear_chain(natoms, axis=2):
    """
    Create a linear chain along partcular axis

    Parameters
    ----------
    natoms: int
        number of atoms

    axis : int, optional
        default 2. Axis along which linear chain is created
    """
    warnings.warn("This function is retained for testing purposes only")

    positions = []
    for i in range(natoms):
        pos = [0, 0, 0]
        pos[axis] = (i - natoms / 2)
        positions.append(pos)
    atoms = Atoms(positions=positions, symbols=["W"] * len(positions))
    nl = ACENeighborList()
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    return ae


def aseatoms_to_atomicenvironment_old(atoms, cutoff=9, skin=0, elements_mapper_dict=None):
    """
    Function to read from a ASE atoms objects

    Parameters
    ----------
    atoms : ASE Atoms object
        name of the ASE atoms object

    cutoff: float
        cutoff value for calculating neighbors
    """

    nl = ACENeighborList(cutoff=cutoff, skin=skin)
    if elements_mapper_dict is not None:
        nl.types_mapper_dict = elements_mapper_dict
    nl.make_neighborlist(atoms)
    ae = ACEAtomicEnvironment()
    ae.x = nl.x
    ae.species_type = nl.species_type
    ae.neighbour_list = nl.jlists
    ae.origins = nl.origins
    return ae


def aseatoms_to_atomicenvironment(atoms, cutoff=9, elements_mapper_dict=None):
    """
    Function to read from a ASE atoms objects

    Parameters
    ----------
    atoms : ASE Atoms object
        name of the ASE atoms object

    cutoff: float
        cutoff value for calculating neighbors
    """

    positions_ = atoms.get_positions()
    species_type_ = atoms.get_atomic_numbers()
    cell_ = np.array(atoms.get_cell())
    if np.all(atoms.get_pbc()):
        pbc = True
    elif np.all(~atoms.get_pbc()):
        pbc = False
    else:
        raise ValueError("Only fully periodic or non-periodic cell are supported")

    ae = build_atomic_env(positions_, cell_, species_type_, pbc, cutoff)

    chem_symbs = np.array([data.chemical_symbols[st] for st in ae.species_type], dtype="S2")
    if elements_mapper_dict is None:
        elements_mapper_dict = {el: i for i, el in enumerate(sorted(set(chem_symbs)))}
    else:
        elements_s2 = np.array(list(elements_mapper_dict.keys()), dtype="S2")
        inds = list(elements_mapper_dict.values())
        elements_mapper_dict = {k: v for k, v in zip(elements_s2, inds)}
    normalized_species_types = [elements_mapper_dict[cs] for cs in chem_symbs]
    ae.species_type = normalized_species_types
    return ae


def calculate_minimal_nn_atomic_env(atomic_env):
    return atomic_env.get_minimal_nn_distance()


def calculate_minimal_nn_tp_atoms(tp_atoms):
    _positions = tp_atoms["_positions"]
    _cell = tp_atoms["_cell"][0]
    _ind_i = tp_atoms["_ind_i"]
    _ind_j = tp_atoms["_ind_j"]
    _offsets = tp_atoms["_offsets"]

    return get_minimal_nn_distance_tp(_positions, _cell, _ind_i, _ind_j, _offsets)


def copy_atoms(atoms):
    if atoms.get_calculator() is not None:
        calc = atoms.get_calculator()
        new_atoms = atoms.copy()
        new_atoms.set_calculator(calc)
    else:
        new_atoms = atoms.copy()

    return new_atoms


def enforce_pbc(atoms, cutoff):
    if (atoms.get_pbc() == 0).all():
        pos = atoms.get_positions()
        max_xyz = np.max(pos, axis=0)
        min_xyz = np.min(pos, axis=0)
        d_xyz = max_xyz - min_xyz
        max_d = np.max(d_xyz)
        cell = np.eye(3) * ((max_d + cutoff) * 2)
        atoms.set_cell(cell, scale_atoms=False)
        atoms.center()
        atoms.set_pbc(True)

    return atoms


def generate_tp_atoms(ase_atoms, cutoff=8.7, verbose=False):
    energy = ase_atoms.get_potential_energy()
    forces = ase_atoms.get_forces()
    atoms = ase_atoms.copy()
    atoms = enforce_pbc(atoms, cutoff)

    positions_ = atoms.get_positions()
    species_type_ = atoms.get_atomic_numbers()
    cell_ = np.array(atoms.get_cell())
    if np.all(atoms.get_pbc()):
        pbc = True
    elif np.all(~atoms.get_pbc()):
        pbc = False
    else:
        raise ValueError("Only fully periodic or non-periodic cell are supported")

    env = get_nghbrs_tp_atoms(positions_, cell_, species_type_, pbc, cutoff)

    # _ind_i, _ind_j, _mu_i, _mu_j, _offsets, true[5], scaled_positions[6]

    if env[5] is True:  # successfull
        cell = cell_.reshape(1, 3, 3)
        return {'_ind_i': env[0], '_ind_j': env[1], '_mu_i': env[2],
                '_mu_j': env[3], '_offsets': env[4].astype(np.float64),
                '_eweights': np.ones((1, 1)), '_fweights': np.ones((len(atoms), 1)),
                '_energy': np.array(energy).reshape(-1, 1), '_forces': forces.astype(np.float64),
                '_positions': env[6].astype(np.float64), '_cell': cell.astype(np.float64)}
    else:
        return None


def calculate_minimal_nn_distance(df: pd.DataFrame, target_column_name = "min_distance"):
    if target_column_name in df.columns:
        return
    if "atomic_env" in df.columns:
        # computing minimum NN distance per structure from atomic_env
        df[target_column_name] = df["atomic_env"].map(calculate_minimal_nn_atomic_env)
    elif "tp_atoms" in df.columns:
        # computing minimum NN distance per structure from tp_atoms
        df[target_column_name] = df["tp_atoms"].map(calculate_minimal_nn_tp_atoms)
    else:
        raise ValueError("Neither `atomic_env` nor `tp_atoms` columns are presented in dataframe")

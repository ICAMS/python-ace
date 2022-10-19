import numpy as np
import os

from collections import Counter, defaultdict

from ase import Atoms
from ase.io.lammpsrun import read_lammps_dump_text
from maxvolpy.maxvol import maxvol

from pyace.asecalc import PyACECalculator
from pyace.atomicenvironment import aseatoms_to_atomicenvironment, ACEAtomicEnvironment
from pyace.basis import BBasisConfiguration, ACEBBasisSet
from pyace.calculator import ACECalculator
from pyace.evaluator import ACEBEvaluator
from typing import Dict, List, Optional, Union, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def save_active_inverse_set(filename, A_active_inverse_set, elements_name=None):
    if elements_name is None:
        elements_name = list(map(str, range(len(A_active_inverse_set))))
    with open(filename, "wb") as f:
        np.savez(f, **{elements_name[st]: v for st, v in A_active_inverse_set.items()})


def count_number_total_atoms_per_species_type(atomic_env_list: List[ACEAtomicEnvironment]) -> Dict[int, int]:
    """
    Helper function to count total number of atoms of each type in dataset

    :param atomic_env_list: List[ACEAtomicEnvironment] -  list of ACEAtomicEnvironment

    :return dictionary: species_type => number
    """
    n_total_atoms_per_species_type = Counter()
    for ae in atomic_env_list:
        n_total_atoms_per_species_type.update(ae.species_type[:ae.n_atoms_real])
    return n_total_atoms_per_species_type


def count_number_total_atoms_per_species_type_by_batches(atomic_env_batches: List[List[ACEAtomicEnvironment]]) -> Dict[
    int, int]:
    """
    Helper function to count total number of atoms of each type in dataset splitted by batches

    :param atomic_env_batches: List[List[ACEAtomicEnvironment]] -  list of batches(lists) of ACEAtomicEnvironment

    :return dictionary: species_type => number
    """
    n_total_atoms_per_species_type = defaultdict(lambda: 0)
    for ae_batch in atomic_env_batches:
        cnt_current_batch = count_number_total_atoms_per_species_type(ae_batch)
        for k, v in cnt_current_batch.items():
            n_total_atoms_per_species_type[k] += v
    return dict(n_total_atoms_per_species_type)


def compute_B_projections(bconf: Union[BBasisConfiguration, ACEBBasisSet, PyACECalculator],
                          atomic_env_list: Union[List[Atoms], List[ACEAtomicEnvironment]],
                          structure_ind_list: Optional[List[int]] = None,
                          is_full=False) -> Tuple[Dict[int, np.array],
                                                  Optional[Dict[int, np.array]]]:
    """
    Function to compute the B-basis projection using basis configuration
    `bconf` for list of atomic environments `atomic_env_list`.
    :param bconf: BBasisConfiguration
    :param atomic_env_list: list of ACEAtomicEnvironment
    :param structure_ind_list: optional, list of corresponding indices of structures/atomic envs.
    :param is_full: MaxVol on linear B-projections (false, default) or on full non-linear atomic energy (True)
    :return A0_projections_dict, structure_ind_dict

        A0_projections_dict:
            dictionary {species_type => B-basis projections}
            B-basis projections shape = [n_atoms[species_type], n_funcs[species_type]]

        structure_ind_dict:
            dictionary {species_type => indices of correcponding structure}
            shape = [n_atoms[species_type]]
    """
    if structure_ind_list is None:
        tmp_structure_ind_list = range(len(atomic_env_list))
    else:
        tmp_structure_ind_list = structure_ind_list

    # create BBasis configuration
    if isinstance(bconf, BBasisConfiguration):
        pot = ACEBBasisSet(bconf)
    elif isinstance(bconf, ACEBBasisSet):
        pot = bconf
    elif isinstance(bconf, PyACECalculator):
        pot = bconf.basis

    n_projections = compute_number_of_functions(pot)
    if is_full:
        n_projections = [p * pot.map_embedding_specifications[st].ndensity for st, p in enumerate(n_projections)]

    elements_mapper_dict = pot.elements_to_index_map

    if isinstance(atomic_env_list[0], Atoms):
        atomic_env_list = [
            aseatoms_to_atomicenvironment(at,
                                          cutoff=pot.cutoffmax,
                                          elements_mapper_dict=pot.elements_to_index_map)
            for at in atomic_env_list]
    elif not isinstance(atomic_env_list[0], ACEAtomicEnvironment):
        raise ValueError("atomic_env_list should be list of ASE.Atoms or ACEAtomicEnvironment")

    # count total number of atoms of each species type in whole dataset atomiv_env_list
    n_total_atoms_per_species_type = count_number_total_atoms_per_species_type(atomic_env_list)

    beval = ACEBEvaluator(pot)

    calc = ACECalculator(beval)

    # prepare numpy arrays for A0_projections and  structure_ind_dict
    A0_projections_dict = {st: np.zeros((n_total_atoms_per_species_type[st],
                                         n_projections[st]
                                         ), dtype=np.float64) for _, st in elements_mapper_dict.items()}

    structure_ind_dict = {st: np.zeros(n_total_atoms_per_species_type[st], dtype=np.int)
                          for _, st in elements_mapper_dict.items()}

    cur_inds = [0] * len(elements_mapper_dict)
    for struct_ind, ae in tqdm(zip(tmp_structure_ind_list, atomic_env_list), total=len(atomic_env_list)):
        calc.compute(ae)
        if is_full:
            basis_projections = calc.dE_dc
        else:
            basis_projections = calc.projections

        for atom_ind, atom_type in enumerate(ae.species_type[:ae.n_atoms_real]):
            cur_ind = cur_inds[atom_type]
            structure_ind_dict[atom_type][cur_ind] = struct_ind
            cur_basis_proj = np.reshape(basis_projections[atom_ind], (-1,))
            A0_projections_dict[atom_type][cur_ind] = cur_basis_proj
            cur_inds[atom_type] += 1
    if structure_ind_list is not None:
        return A0_projections_dict, structure_ind_dict
    else:
        return A0_projections_dict


def compute_number_of_functions(pot):
    return [len(b1) + len(b) for b1, b in zip(pot.basis_rank1, pot.basis)]


def compute_extrapolation_grade(A0_projections_dict: Dict[int, np.array], A_active_set_inv_dict: Dict[int, np.array]) -> \
        Dict[int, np.array]:
    """
    Compute extrapolation grade `gamma` for given dictionary
    of projections `A0_projections_dict` and `A_active_set_inv_dict`

    :param A0_projections_dict:  dictionary {species_type => B-basis projections}
                shape = [n_atoms[species_type], n_funcs[species_type]]

    :param A_active_set_inv_dict:
            dictionary {species_type => A_active_set_inv}
                shape = [n_funcs[species_type], n_funcs[species_type]]

    :return gamma_dict:  dictionary {species_type => gamma}
            shape = [n_atoms[species_type]]
    """
    gamma_dict = {}
    for st in A0_projections_dict.keys():
        cur_gamma_grade = np.abs(np.dot(A0_projections_dict[st], A_active_set_inv_dict[st])).max(axis=1)
        gamma_dict[st] = cur_gamma_grade
    return gamma_dict


def compute_extrapolation_grade_by_batches(bbasis: Union[BBasisConfiguration, ACEBBasisSet],
                                           atomic_env_batches: List[List[ACEAtomicEnvironment]],
                                           A_active_set_inv_dict: Dict[int, np.array],
                                           structure_ind_batches: Optional[List[List[int]]] = None,
                                           gamma_threshold: Optional[float] = None,
                                           is_full=False):
    """
    Compute the extrapolation grade of big dataset (`atomic_env_batches`) by batches,
    so only projections of one batch are storing in memory at once

    :param bbasis: BBasisConfiguration or ACEBBasisSet
    :param atomic_env_batches: list of list of ACEAtomicEnvironments
    :param A_active_set_inv_dict: dict: {species type => inverted active set matrix}
    :param structure_ind_batches: (optional) list of list (or np.array) of structure indices
    :param gamma_threshold: (optional, float, default=None) - if provided,  then projection and structure indices
     for atoms with gamma > gamma_threshold (extrapolative) will be collected and returned

    :return:
        gamma_dict,
        extrapolative_A0_projs_dict (if gamma_threshold is not None),
        extrapolative_structure_ind_dict (if gamma_threshold and structure_ind_batches are not None)
    """
    cnt_specie_types = count_number_total_atoms_per_species_type_by_batches(atomic_env_batches)
    species_types = sorted(cnt_specie_types.keys())
    gamma_dict = {k: np.zeros((v,)) for k, v in cnt_specie_types.items()}

    if gamma_threshold is not None:
        # species_type -> projs
        extrapolative_A0_projs_dict = {st: [] for st in species_types}

        # species_type -> struct.ind
        extrapolative_structure_ind_dict = {st: [] for st in species_types}

    cur_inds = [0] * len(species_types)
    for batch_num in range(len(atomic_env_batches)):
        print("Batch #", batch_num)
        ae_batch = atomic_env_batches[batch_num]

        print("Compute B-projections")
        if structure_ind_batches is not None:
            si_batch = structure_ind_batches[batch_num]
            cur_A0_projections_dict, cur_structure_ind_dict = compute_B_projections(bbasis, ae_batch, si_batch,
                                                                                    is_full=is_full)
        else:
            cur_A0_projections_dict, _ = compute_B_projections(bbasis, ae_batch, is_full=is_full)

        for st in species_types:
            cur_A0_projections = cur_A0_projections_dict[st]

            cur_gamma_grade = np.abs(np.dot(cur_A0_projections, A_active_set_inv_dict[st])).max(axis=1)

            gamma_dict[st][cur_inds[st]:cur_inds[st] + len(cur_gamma_grade)] = cur_gamma_grade

            # keep track of extrapolative structures
            if gamma_threshold is not None:
                extrapolation_mask = cur_gamma_grade > gamma_threshold
                extrapolative_A0_projs_dict[st].append(cur_A0_projections[extrapolation_mask])
                if structure_ind_batches is not None:
                    extrapolative_structure_ind_dict[st].append(cur_structure_ind_dict[st][extrapolation_mask])
            cur_inds[st] += len(cur_gamma_grade)

    if gamma_threshold is not None:
        # flatten extrapolative A0_proj collections
        for st in species_types:
            extrapolative_A0_projs_dict[st] = np.vstack(extrapolative_A0_projs_dict[st])

        if structure_ind_batches is not None:
            # flatten extrapolative structure indices
            for st in species_types:
                extrapolative_structure_ind_dict[st] = np.hstack(extrapolative_structure_ind_dict[st])
            return gamma_dict, extrapolative_A0_projs_dict, extrapolative_structure_ind_dict

        else:
            return gamma_dict, extrapolative_A0_projs_dict
    else:
        return gamma_dict


def compute_active_set(A0_projections_dict: Dict[int, np.array],
                       structure_ind_dict: Optional[Dict[int, np.array]] = None,
                       tol: float = 1.001,
                       max_iters: int = 300,
                       verbose=False
                       ):
    """
    Compute active set using MaxVol algorithm
    for each species type from dictionary A0_projections_dict
    :param A0_projections_dict:  dictionary {species_type => B-basis projections}
                shape = [n_atoms[species_type], n_funcs[species_type]]
    :param tol: tolerance for MaxVol algorithm
    :param max_iters: maximum number of iterations for MaxVol algorithm

    :return A_active_set_dict:
            dictionary {species_type => A_active_set_inv}
                shape = [n_funcs[species_type], n_funcs[species_type]]
    """
    species_types = sorted(A0_projections_dict.keys())
    A_active_set_dict = {st: [] for st in species_types}

    if structure_ind_dict is not None:
        active_set_structure_inds_dict = {}

    for st in species_types:
        if verbose:
            print("Species type:", st)
        cur_A0 = A0_projections_dict[st]
        shape = cur_A0.shape
        if shape[0] < shape[1]:
            raise ValueError(
                "Insufficient atomic environments to determine active set for species type {}, "
                "system is under-determined, projections shape={},{}".format(st, shape))
        selected_rows, _ = maxvol(cur_A0, tol=tol, max_iters=max_iters)
        cur_A_active_set = cur_A0[selected_rows]
        A_active_set_dict[st] = cur_A_active_set
        if structure_ind_dict is not None:
            active_set_structure_inds_dict[st] = structure_ind_dict[st][selected_rows]

    if structure_ind_dict is not None:
        return A_active_set_dict, active_set_structure_inds_dict
    else:
        return A_active_set_dict


def compute_active_set_by_batches(bbasis: Union[BBasisConfiguration, ACEBBasisSet],
                                  atomic_env_batches: List[List[ACEAtomicEnvironment]],
                                  structure_ind_batches: Optional[List[List[int]]] = None,
                                  gamma_tolerance: float = 1.01,
                                  maxvol_iters: int = 300,
                                  n_refinement_iter: int = 5,
                                  save_interim_active_set: bool = False,
                                  is_full=False
                                  ):
    """

    Compute the active set for each species type of big dataset (`atomic_env_batches`) by batches,
    so only projections of one batch are storing in memory at once.
    Stage 1: Initial cumulative  MaxVol (i.e. current active set = MaxVol (current active set + current batch))
    over batches (however this is not EXACT solution).
    Stage 2. Refinement of active set by considering ONLY structures with gamma>1 and current active set

    :param bbasis: BBasisConfiguration or ACEBBasisSet
    :param atomic_env_batches: list of list of ACEAtomicEnvironments
    :param structure_ind_batches: (optional) list of list (or np.array) of structure indices
    :param gamma_tolerance: (float, default=1.01) - gamma tolerance for MaxVol algorithm
    :param maxvol_iters: int, maximum number of iterations for single MaxVol run (for each batch)
    :param n_refinement_iter: maximum number of refinement iterations (stage 2)
    :param save_interim_active_set: bool = False,
    :param is_full: MaxVol on linear B-projections (false, default) or on full non-linear atomic energy (True)

    :return:
    best_gamma: (dict)
            {species type => maximum gamma over dataset}
    best_active_sets_dict: (dict)
            {species type => active set matrix}
    best_active_sets_structure_indices_dict: (dict, optional, if structure_ind_batches is not None)
            {species type => list of structure indices for active set}
"""
    if isinstance(bbasis, BBasisConfiguration):
        bbasis = ACEBBasisSet(bbasis)
    elif not isinstance(bbasis, ACEBBasisSet):
        raise ValueError("compute_active_set_by_batches:bbasis should be BBasisConfiguration or ACEBBasisSet")
    elements_name = bbasis.elements_name
    species_types = list(range(len(elements_name)))

    # Stage 1. Initial cumulative maxvol over batches (however this is not EXACT solution).
    # One needs to do refinement in Stage 2.
    cur_A_active_set_dict = {v: [] for v in species_types}
    cur_structure_inds_active_set_dict = {v: [] for v in species_types}

    print("Stage 1. Cumulative MaxVol by batches")
    n_batches = len(atomic_env_batches)
    for batch_num in range(n_batches):
        print("Batch #{}/{}".format(batch_num + 1, n_batches))
        ae_batch = atomic_env_batches[batch_num]

        print("Compute B-projections")
        if structure_ind_batches is not None:
            si_batch = structure_ind_batches[batch_num]
            cur_A0_projections_dict, cur_structure_ind_dict = compute_B_projections(bbasis, ae_batch, si_batch,
                                                                                    is_full=is_full)
        else:
            cur_A0_projections_dict, cur_structure_ind_dict = compute_B_projections(bbasis, ae_batch,
                                                                                    is_full=is_full)

        # join current active set {cur_A_active_set_dict, cur_structure_inds_active_set_dict}
        # and current batch {cur_A0_projections_dict, cur_structure_ind_dict}
        joint_A0_projections_dict = {}
        joint_structure_ind_dict = {}
        for st in cur_A0_projections_dict.keys():

            if len(cur_A_active_set_dict[st]) and len(cur_A0_projections_dict[st]):
                joint_A0_projections_dict[st] = np.vstack((cur_A_active_set_dict[st],
                                                           cur_A0_projections_dict[st]))

                joint_structure_ind_dict[st] = np.hstack((cur_structure_inds_active_set_dict[st],
                                                          cur_structure_ind_dict[st]))

            elif len(cur_A_active_set_dict[st]):
                joint_A0_projections_dict[st] = cur_A_active_set_dict[st]
                joint_structure_ind_dict[st] = cur_structure_inds_active_set_dict[st]
            elif len(cur_A0_projections_dict[st]):
                joint_A0_projections_dict[st] = cur_A0_projections_dict[st]
                joint_structure_ind_dict[st] = cur_structure_ind_dict[st]
        print("Update active set (MaxVol)")
        cur_A_active_set_dict, cur_structure_inds_active_set_dict = compute_active_set(joint_A0_projections_dict,
                                                                                       joint_structure_ind_dict,
                                                                                       tol=gamma_tolerance,
                                                                                       max_iters=maxvol_iters)
        print("Done")

    # Stage 2. refinement of active set by considering ONLY structures with gamma>1 and current active set

    cur_A_active_set_inv_dict = compute_A_active_inverse(cur_A_active_set_dict)
    if save_interim_active_set:
        print("Save current active set to interim_active_set.asi")
        save_active_inverse_set("interim_active_set.asi", cur_A_active_set_inv_dict, elements_name=elements_name)

    print("Compute extrapolation grade for whole dataset by batches")
    gamma_grade, extrapolative_A0_projs_dict, extrapolative_structure_ind_dict = \
        compute_extrapolation_grade_by_batches(bbasis, atomic_env_batches, cur_A_active_set_inv_dict,
                                               structure_ind_batches,
                                               gamma_threshold=gamma_tolerance, is_full=is_full)
    best_gamma = {k: gg.max() for k, gg in gamma_grade.items()}
    # initilize best active set with current active set
    best_active_sets_dict = cur_A_active_set_dict.copy()
    best_active_sets_si_dict = cur_structure_inds_active_set_dict.copy()
    print("Current best gamma:", best_gamma)

    if max(best_gamma.values()) <= gamma_tolerance:
        if structure_ind_batches is not None:
            return best_gamma, best_active_sets_dict, best_active_sets_si_dict
        else:
            return best_gamma, best_active_sets_dict
    print("Stage 2. Refinement of active set")
    # do iterative refinement
    for i in range(n_refinement_iter):
        print()
        print("Refinement iteration #{}/{}".format(i + 1, n_refinement_iter))

        cur_A0_projections_dict = {}
        cur_structure_ind_dict = {}

        for st in species_types:
            print("Species type: {}, atomic environments outside active set: {}".format(st, len(
                extrapolative_A0_projs_dict[st])))
            cur_A0_projections_dict[st] = extrapolative_A0_projs_dict[st]
            cur_structure_ind_dict[st] = extrapolative_structure_ind_dict[st]

        # join current active set {cur_A_active_set_dict, cur_structure_inds_active_set_dict}
        # and current batch {cur_A0_projections_dict, cur_structure_ind_dict}
        joint_A0_projections_dict = {}
        joint_structure_ind_dict = {}
        for st in cur_A0_projections_dict.keys():

            if len(cur_A_active_set_dict[st]) and len(cur_A0_projections_dict[st]):
                joint_A0_projections_dict[st] = np.vstack((cur_A_active_set_dict[st],
                                                           cur_A0_projections_dict[st]))

                joint_structure_ind_dict[st] = np.hstack((cur_structure_inds_active_set_dict[st],
                                                          cur_structure_ind_dict[st]))

            elif len(cur_A_active_set_dict[st]):
                joint_A0_projections_dict[st] = cur_A_active_set_dict[st]
                joint_structure_ind_dict[st] = cur_structure_inds_active_set_dict[st]
            elif len(cur_A0_projections_dict[st]):
                joint_A0_projections_dict[st] = cur_A0_projections_dict[st]
                joint_structure_ind_dict[st] = cur_structure_ind_dict[st]
        print("Update active set")
        cur_A_active_set_dict, cur_structure_inds_active_set_dict = compute_active_set(joint_A0_projections_dict,
                                                                                       joint_structure_ind_dict)
        print("Done")
        print("PseudoInversion of current active set")
        cur_A_active_set_inv_dict = compute_A_active_inverse(cur_A_active_set_dict)

        if save_interim_active_set:
            print("Save current active set to interim_active_set.asi")
            save_active_inverse_set("interim_active_set.asi", cur_A_active_set_inv_dict, elements_name=elements_name)

        print("Compute extrapolation grade for complete dataset by batches")
        gamma_grade, extrapolative_A0_projs_dict, extrapolative_structure_ind_dict = \
            compute_extrapolation_grade_by_batches(bbasis, atomic_env_batches, cur_A_active_set_inv_dict,
                                                   structure_ind_batches,
                                                   gamma_threshold=gamma_tolerance, is_full=is_full)

        current_gamma = {k: gg.max() for k, gg in gamma_grade.items()}
        print("Current gamma_max:", current_gamma)

        # update best gamma grade
        for st in species_types:
            if current_gamma[st] < best_gamma[st]:
                # update best gamma grade, AS and SI
                best_gamma[st] = current_gamma[st]
                best_active_sets_dict[st] = cur_A_active_set_dict[st]
                best_active_sets_si_dict[st] = cur_structure_inds_active_set_dict[st]
                print("New best gamma({})={}".format(st, best_gamma[st]))
        print("Current best gamma:", best_gamma)

        # early stopping if max value of gamma is below tolerance
        if max(best_gamma.values()) <= gamma_tolerance:
            break

    if structure_ind_batches is not None:
        return best_gamma, best_active_sets_dict, best_active_sets_si_dict
    else:
        return best_gamma, best_active_sets_dict


def compute_A_active_inverse(A_active_set_dict: Dict[int, np.array]) -> Dict[int, np.array]:
    """
    Do the pseudo-inversion of active set matrices for each species type
    :param A_active_set_dict: dict {species_type => active set matrix}

    :return: A_active_set_inv_dict - dict {species_type => active set pseudoinverted matrix}
    """
    A_active_set_inv_dict = {k: np.linalg.pinv(v) for k, v in A_active_set_dict.items()}
    return A_active_set_inv_dict


# https://github.com/corochann/chainer-pointnet/blob/master/chainer_pointnet/utils/sampling.py
# LICENSE: MIT
def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=1)


def py_farthest_point_sampling_no_batch(pts,
                                        n_sample_to_select,
                                        initial_idx=None,
                                        metrics=l2_norm,
                                        verbose=False):
    """
    Farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (npts, nfeat)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        n_sample_to_select (int): number of points to sample
        initial_idx (list or None): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
    Returns: `indices`
        indices (numpy.ndarray): 1-dim array (n_sample_to_select, )
            indices of sampled farthest points.
    """

    npts, nfeat = pts.shape
    indices = np.zeros((n_sample_to_select,), dtype=int)

    if initial_idx is None:
        indices[0] = np.random.randint(npts)
        n_initial = 1
    else:
        n_initial = len(initial_idx)
        indices[:n_initial] = initial_idx

    farthest_point = pts[indices[n_initial - 1]]

    # minimum distances to the sampled farthest point
    min_distances = metrics(farthest_point, pts)

    monitor = lambda x: x

    if verbose:
        try:
            import tqdm
            monitor = tqdm.tqdm
        except ImportError:
            pass

    for i in monitor(range(n_initial, n_sample_to_select)):
        indices[i] = np.argmax(min_distances, axis=0)
        farthest_point = pts[indices[i]]
        dist = metrics(farthest_point[None, :], pts)
        min_distances = np.minimum(min_distances, dist)
    return indices


def read_extrapolation_data(extrapolation_filename, species_type_filename=None):
    if species_type_filename is None:
        species_type_filename = os.path.join(os.path.dirname(extrapolation_filename), "species_types.dat")

    with open(extrapolation_filename) as f:
        dat = read_lammps_dump_text(f, index=slice(None))

    with open(species_type_filename) as f:
        l = f.readlines()

    elements = l[0].strip().split()

    elements_remap = {i + 1: el for i, el in enumerate(elements)}

    for at in dat:
        new_symb = [elements_remap[i] for i in at.get_atomic_numbers()]
        at.set_chemical_symbols(new_symb)
        # this is the BUGFIX of ASE, that do not set pbc=True even when cell is provided
        if np.linalg.det(at.get_cell()) > 0:
            at.set_pbc(True)

    return dat

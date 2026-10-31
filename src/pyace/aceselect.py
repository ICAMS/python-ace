import logging
import os
import psutil

from collections import Counter

import numpy as np
import pandas as pd

from ase.io import write

from pyace import ACEBBasisSet
from pyace.activelearning import (compute_B_projections, compute_active_set_by_batches, read_extrapolation_data,
    compute_number_of_functions, count_number_total_atoms_per_species_type_aseatom, compute_active_set,
    convert_to_bbasis)
from pyace.preparedata import sizeof_fmt

LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
log = logging.getLogger()


def select_structures_maxvol(df, bconf, extra_A0_projections_dict=None, batch_size_option="auto", gamma_tolerance=1.01,
                             max_structures=-1, maxvol_iters=300,
                             maxvol_refinement=2, mem_lim_option="auto", verbose=False):
    """
    Select the structures from "df" with maximal amount of atoms, that entered into active set

    :param df: (pd.DataFrame) with "ase_atoms" column and other
    :param bconf: (BBasisConfiguration) ACE potential
    :param extra_A0_projections_dict: (dict of 2D arrays), default - None, otherwise - another active set
    :param batch_size_option: "auto" (default), "none" or int
    :param gamma_tolerance: default = 1.05
    :param max_structures: default = -1 - all structures
    :param maxvol_iters: default = 300
    :param maxvol_refinement: default = 2
    :param mem_lim_option: default = "auto"
    :param verbose: default = False

    :return: df_selected  - subset of input "df"
    """
    # create BBasis configuration
    bbasis = convert_to_bbasis(bconf)

    nfuncs = compute_number_of_functions(bbasis)
    elements_to_index_map = bbasis.elements_to_index_map
    elements_name = bbasis.elements_name
    ase_atoms_list = df["ase_atoms"]
    total_number_of_atoms_per_species_type = count_number_total_atoms_per_species_type_aseatom(ase_atoms_list,
                                                                                               elements_to_index_map)
    required_active_set_memory, required_projections_memory = compute_required_memory(
        total_number_of_atoms_per_species_type, elements_name, nfuncs, verbose=verbose)
    if verbose:
        log.info(
            "Required memory to store complete dataset projections: {}".format(sizeof_fmt(required_projections_memory)))
        log.info("Required memory to store active set: {}".format(sizeof_fmt(required_active_set_memory)))
    num_structures = len(ase_atoms_list)
    # compute mem_lim
    mem_lim = compute_mem_limit(mem_lim_option)
    batch_size = compute_batch_size(batch_size_option, mem_lim, num_structures, required_active_set_memory,
                                    required_projections_memory, verbose)
    if batch_size is None:
        # single shot MaxVol
        if verbose:
            log.info("Single-shot mode")
            log.info("Compute B-projections")
        b_proj, structure_ind_dict = compute_B_projections(bconf, ase_atoms_list,
                                                           return_structure_ind_dict=True, verbose=verbose)

        if verbose:
            log.info("Selecting structures")
        res_no_batch = compute_active_set(b_proj, structure_ind_dict,
                                          tol=gamma_tolerance, max_iters=maxvol_iters,
                                          extra_A0_projections_dict=extra_A0_projections_dict, verbose=verbose)
        selected_structures_inds_dict = res_no_batch[1]
    else:
        # batching
        if verbose:
            log.info("Batch mode")
        n_batches = num_structures // batch_size
        if verbose:
            log.info("Number of batches: {}".format(n_batches))
            log.info("Selecting structures in batch mode")
        res_batch = compute_active_set_by_batches(bconf, ase_atoms_list,
                                                  n_batches=n_batches,
                                                  gamma_tolerance=gamma_tolerance,
                                                  maxvol_iters=maxvol_iters,
                                                  n_refinement_iter=maxvol_refinement,
                                                  save_interim_active_set=True,
                                                  extra_A_active_set_dict=extra_A0_projections_dict,
                                                  verbose=verbose)
        selected_structures_inds_dict = res_batch[2]
    cnt = Counter()
    for mu, sel_structs in selected_structures_inds_dict.items():
        cnt.update(sel_structs)
    if -1 in cnt:
        cnt.pop(-1)  # remove structures from active set
    if verbose:
        log.info(f"Overall {len(cnt)} structures selected")
    selected_structures_inds = ([int(i) for i, _ in cnt.most_common() if i != -1])
    if max_structures > 0:
        if verbose:
            log.info(f"Selection top {max_structures} structures")
        if max_structures <= len(selected_structures_inds):
            selected_structures_inds = selected_structures_inds[:max_structures]
    df_selected = df.iloc[selected_structures_inds].copy()
    return df_selected


def compute_required_memory(total_number_of_atoms_per_species_type, elements_name, nfuncs, n_projections=None,
                            verbose=False):
    if n_projections is None:
        n_projections = nfuncs
    number_of_projection_entries = 0
    required_active_set_memory = 0
    for st in total_number_of_atoms_per_species_type.keys():
        if verbose:
            log.info("\tElement: {}, # atoms: {}, # B-func: {}, # projections: {}".format(elements_name[st],
                                                                                          total_number_of_atoms_per_species_type[
                                                                                              st], nfuncs[st],
                                                                                          n_projections[st]))
        number_of_projection_entries += total_number_of_atoms_per_species_type[st] * n_projections[st]
        required_active_set_memory += n_projections[st] ** 2
    required_projections_memory = number_of_projection_entries * 8  # float64
    required_active_set_memory *= 8  # in bytes, float64
    return required_active_set_memory, required_projections_memory


def load_multiple_datasets(dataset_filename, elements=None, verbose=False):
    df_list = []
    for i, dsfn in enumerate(dataset_filename):
        if not os.path.isfile(dsfn):
            raise RuntimeError("File {} not found".format(dsfn))

        if verbose:
            log.info("Loading dataset #{}/{} from {}".format(i + 1, len(dataset_filename), dsfn))
        if dsfn.endswith(".pckl.gzip") or dsfn.endswith(".pkl.gz"):
            df = pd.read_pickle(dsfn, compression="gzip")
            if verbose:
                log.info("Number of structures: {}".format(len(df)))
            df_list.append(df)
        elif dsfn.endswith(".dat") or dsfn.endswith(".dump"):
            if elements is None:
                raise ValueError('`elements` are missing. Provide it as in LAMMPS with --elements "A B C"')
            species_to_element_dict = {i + 1: e for i, e in enumerate(elements.split())}
            # LAMMPS dat format
            structures = read_extrapolation_data(dsfn, species_to_element_dict=species_to_element_dict)
            df = pd.DataFrame({"ase_atoms": structures})
            df_list.append(df)
        else:
            raise ValueError(f"Unsupported file type: {dsfn}")
    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_mem_limit(mem_lim):
    if mem_lim == "auto":
        # determine 80% of available memory
        mem_lim = int(0.8 * psutil.virtual_memory().available)
    else:
        mem_lim = mem_lim.replace("GB", "*2**30").replace("MB", "*2**20")
        mem_lim = eval(mem_lim)
    return mem_lim


def compute_batch_size(batch_size_option, mem_lim, num_structures, required_active_set_memory,
                       required_projections_memory,
                       verbose):
    if batch_size_option == "auto":
        if verbose:
            log.info("Automatic batch_size determination")
            log.info("Memory limit: {}".format(sizeof_fmt(mem_lim)))
        if 2 * required_projections_memory + required_active_set_memory < mem_lim:
            batch_size_option = None
        else:
            nsplits = int(np.ceil(2 * required_projections_memory // (mem_lim - required_active_set_memory)))
            batch_size_option = int(np.round(num_structures / nsplits))
    elif batch_size_option in ["None", "none"]:
        batch_size_option = None
    else:
        batch_size_option = int(batch_size_option)
    return batch_size_option


def save_selected_structures(df_selected, selected_structures_filename, verbose):
    if "POSCAR" in selected_structures_filename:
        if verbose:
            log.info(f"Saving selected structures to {selected_structures_filename.replace('POSCAR', '*.POSCAR')}")
        os.makedirs(os.path.dirname(selected_structures_filename), exist_ok=True)
        for i, at in enumerate(df_selected["ase_atoms"]):
            fname = selected_structures_filename.replace("POSCAR", "{}.POSCAR".format(i))
            write(fname, at, format="vasp", sort=True)
    else:
        if verbose:
            log.info(f"Saving selected structures to {selected_structures_filename} as pickled dataframe")
        df_selected.to_pickle(selected_structures_filename)
    if verbose:
        log.info("Done")


def load_datasets(dataset_filename, data_path, verbose=False):
    df_list = []
    for i, dsfn in enumerate(dataset_filename):
        if os.path.isfile(dsfn):
            dsfn = dsfn
        elif os.path.isfile(os.path.join(data_path, dsfn)):
            dsfn = os.path.join(data_path, dsfn)
        else:
            raise RuntimeError("File {} not found".format(dsfn))
        if verbose:
            log.info("Loading dataset #{}/{} from {}".format(i + 1, len(dataset_filename), dsfn))
        df = pd.read_pickle(dsfn, compression="gzip")
        if verbose:
            log.info("Number of structures: {}".format(len(df)))
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df

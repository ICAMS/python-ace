#!/usr/bin/env python
import argparse
import logging
import os
import psutil

import numpy as np
import pandas as pd

from pyace import BBasisConfiguration, ACEBBasisSet, aseatoms_to_atomicenvironment
from pyace.activelearning import compute_B_projections, compute_active_set, compute_active_set_by_batches, \
    compute_A_active_inverse, compute_extrapolation_grade, compute_number_of_functions, \
    count_number_total_atoms_per_species_type, save_active_inverse_set
from pyace.preparedata import sizeof_fmt

log = logging.getLogger()

parser = argparse.ArgumentParser(prog="pace_activeset",
                                 description="Utility to compute active set for PACE (.yaml) potential")

# parser.add_argument("potential_file", help="B-basis file name (.yaml)", type=str, nargs='+', default=[])
parser.add_argument("potential_file", help="B-basis file name (.yaml)", type=str)
parser.add_argument("-d", "--dataset", help="Dataset file name, ex.: filename.pckl.gzip", type=str)
parser.add_argument("-f", "--full", help="Compute active set on full (linearized) design matrix",
                    action='store_true')
parser.add_argument("-b", "--batch_size", help="Batch size (number of structures) considered simultaneously."
                                               "If not provided - all dataset at once is considered",
                    default="auto", type=str)
parser.add_argument("-g", "--gamma_tolerance", help="Gamma tolerance",
                    default=1.01, type=float)
parser.add_argument("-i", "--maxvol_iters", help="Number of maximum iteration in MaxVol algorithm",
                    default=300, type=int)

parser.add_argument("-r", "--maxvol_refinement", help="Number of refinements (epochs)",
                    default=5, type=int)

parser.add_argument("-m", "--memory-limit", help="Memory limit (i.e. 1GB, 500MB or 'auto')", default="auto", type=str)

args_parse = parser.parse_args()
potential_file = args_parse.potential_file
dataset_filename = args_parse.dataset
batch_size = args_parse.batch_size
gamma_tolerance = args_parse.gamma_tolerance
maxvol_iters = args_parse.maxvol_iters
maxvol_refinement = args_parse.maxvol_refinement
mem_lim = args_parse.memory_limit
is_full = args_parse.full
if mem_lim == "auto":
    # determine 80% of available memory
    mem_lim = int(0.8 * psutil.virtual_memory().available)
else:
    mem_lim = mem_lim.replace("GB", "*2**30").replace("MB", "*2**20")
    mem_lim = eval(mem_lim)

data_path = os.environ.get("PACEMAKERDATAPATH", "")
if data_path:
    log.info("Data path set to $PACEMAKERDATAPATH = {}".format(data_path))

if os.path.isfile(dataset_filename):
    dataset_filename = dataset_filename
elif os.path.isfile(os.path.join(data_path, dataset_filename)):
    dataset_filename = os.path.join(data_path, dataset_filename)
else:
    raise RuntimeError("File {} not found".format(dataset_filename))

df = pd.read_pickle(dataset_filename, compression="gzip")
df.reset_index(drop=True, inplace=True)
log.info("Number of structures: {}".format(len(df)))
log.info("Potential file: ".format(potential_file))

bconf = BBasisConfiguration(potential_file)
bbasis = ACEBBasisSet(bconf)
nfuncs = compute_number_of_functions(bbasis)
if is_full:
    n_projections = [p * bbasis.map_embedding_specifications[st].ndensity for st, p in enumerate(nfuncs)]
else:  # linear
    n_projections = nfuncs

elements_to_index_map = bbasis.elements_to_index_map
elements_name = bbasis.elements_name
cutoffmax = bbasis.cutoffmax

ATOMIC_ENV_COLUMN = "atomic_env"

rebuild_atomic_env = False
if ATOMIC_ENV_COLUMN not in df.columns:
    rebuild_atomic_env = True
else:
    # check if cutoff is not smaller than requested now
    try:
        metadata_kwargs = df.metadata_dict[ATOMIC_ENV_COLUMN + "_kwargs"]
        metadata_cutoff = metadata_kwargs["cutoff"]
        if metadata_cutoff < cutoffmax:
            log.warning("WARNING! Column {} was constructed with smaller cutoff ({}A) "
                        "that necessary now ({}A). "
                        "Neighbourlists will be re-built".format(ATOMIC_ENV_COLUMN, metadata_cutoff,
                                                                 cutoffmax))
            rebuild_atomic_env = True
        else:
            log.info("Column '{}': existing cutoff ({}A) >= "
                     "requested  cutoff ({}A), skipping...".format(ATOMIC_ENV_COLUMN, metadata_cutoff,
                                                                   cutoffmax))
            rebuild_atomic_env = False

    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        log.info("Could not extract cutoff metadata "
                 "for column '{}' (error: {}). Please ensure the valid cutoff for "
                 "precomputed neighbourlists".format(ATOMIC_ENV_COLUMN, e))
        rebuild_atomic_env = False

if rebuild_atomic_env:
    log.info("Constructing {} column, cutoffmax={}, elements_to_index_map={}".format(ATOMIC_ENV_COLUMN, cutoffmax,
                                                                                     elements_to_index_map))
    df[ATOMIC_ENV_COLUMN] = df["ase_atoms"].apply(aseatoms_to_atomicenvironment,
                                                  cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)

atomic_env_list = df[ATOMIC_ENV_COLUMN]
structure_ind_list = df.index
total_number_of_atoms_per_species_type = count_number_total_atoms_per_species_type(atomic_env_list)

number_of_projection_entries = 0
required_active_set_memory = 0
for st in total_number_of_atoms_per_species_type.keys():
    log.info("\tElement: {}, # atoms: {}, # B-func: {}, # projections: {}".format(elements_name[st],
                                                                                  total_number_of_atoms_per_species_type[
                                                                                      st],
                                                                                  nfuncs[st], n_projections[st]
                                                                                  ))
    number_of_projection_entries += total_number_of_atoms_per_species_type[st] * n_projections[st]
    required_active_set_memory += n_projections[st] ** 2

required_projections_memory = number_of_projection_entries * 8  # float64
required_active_set_memory *= 8  # in bytes, float64
log.info("Required memory to store complete dataset projections: {}".format(sizeof_fmt(required_projections_memory)))
log.info("Required memory to store active set: {}".format(sizeof_fmt(required_active_set_memory)))

if batch_size == "auto":
    log.info("Automatic batch_size determination")
    log.info("Memory limit: {}".format(sizeof_fmt(mem_lim)))
    if 2 * required_projections_memory + required_active_set_memory < mem_lim:
        batch_size = None
    else:
        nsplits = int(np.ceil(2 * required_projections_memory // (mem_lim - required_active_set_memory)))
        batch_size = int(np.round(len(atomic_env_list) / nsplits))
elif batch_size == "None" or batch_size == "none":
    batch_size = None
else:
    batch_size = int(batch_size)

if is_full:
    active_set_inv_filename = potential_file.replace(".yaml", ".asi.nonlinear")
    log.info("FULL (non-linear) matrix will be used for active set calculation")
else:
    active_set_inv_filename = potential_file.replace(".yaml", ".asi")
    log.info("LINEAR matrix will be used for active set calculation")

if batch_size is None:
    # single shot MaxVol
    log.info("Single-run (no batch_size is provided)")
    log.info("Compute B-projections")
    A0_proj_dict = compute_B_projections(bbasis, atomic_env_list, is_full=is_full)
    log.info("B-projections computed:")
    for st, A0_proj in A0_proj_dict.items():
        log.info("\tElement: {}, B-projections shape: {}".format(elements_name[st], A0_proj.shape))

    log.info("Compute active set (using MaxVol algorithm)")
    A_active_set_dict = compute_active_set(A0_proj_dict, tol=gamma_tolerance, max_iters=maxvol_iters, verbose=True)
    log.info("Compute pseudoinversion of active set")
    A_active_inverse_set = compute_A_active_inverse(A_active_set_dict)
    log.info("Done")
    gamma_dict = compute_extrapolation_grade(A0_proj_dict, A_active_inverse_set)
    gamma_max = {k: gg.max() for k, gg in gamma_dict.items()}

    for st, AS_inv in A_active_inverse_set.items():
        log.info("\tElement: {}, Active set inv. shape: {}, gamma_max: {:.3f}".format(elements_name[st], AS_inv.shape,
                                                                                      gamma_max[st]))
    log.info("Saving Active Set Inversion (ASI) to {}".format(active_set_inv_filename))
    with open(active_set_inv_filename, "wb") as f:
        np.savez(f, **{elements_name[st]: v for st, v in A_active_inverse_set.items()})
    log.info("Saving  done to {} ({})".format(active_set_inv_filename, sizeof_fmt(active_set_inv_filename)))
else:
    # multiple round maxvol
    log.info("Approximated MaxVol by batches")
    log.info("Batch size: {}".format(batch_size))
    nsplits = len(atomic_env_list) // batch_size
    atomic_env_batches = np.array_split(atomic_env_list, nsplits)
    atomic_env_batches = [b.values for b in atomic_env_batches]
    structure_env_batches = np.array_split(structure_ind_list, nsplits)
    structure_env_batches = [b.values for b in structure_env_batches]
    log.info("Number of batches: {}".format(len(atomic_env_batches)))

    log.info("Compute approximate active set (using batched MaxVol algorithm)")
    (best_gamma, best_active_sets_dict, _) = \
        compute_active_set_by_batches(
            bbasis,
            atomic_env_batches=atomic_env_batches,
            structure_ind_batches=structure_env_batches,
            gamma_tolerance=gamma_tolerance,
            maxvol_iters=maxvol_iters,
            n_refinement_iter=maxvol_refinement,
            save_interim_active_set=True,
            is_full=is_full
        )
    log.info("Compute pseudoinversion of active set")
    A_active_inverse_set = compute_A_active_inverse(best_active_sets_dict)
    for st, AS_inv in A_active_inverse_set.items():
        log.info("\tElement: {}, Active set inv. shape: {}, gamma_max: {:.3f}".format(elements_name[st], AS_inv.shape,
                                                                                      best_gamma[st]))
    log.info("Saving Active Set Inversion (ASI) to {}".format(active_set_inv_filename))
    save_active_inverse_set(active_set_inv_filename, A_active_inverse_set, elements_name=elements_name)
    log.info("Saving  done to {} ({})".format(active_set_inv_filename, sizeof_fmt(active_set_inv_filename)))

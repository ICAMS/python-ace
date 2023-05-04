import os
import pandas as pd
import numpy as np
import pytest
from pyace import *

from pyace.const import PYACE_EVAL
from pyace.preparedata import get_reference_dataset, generate_atomic_env_column
from pyace.activelearning import *

MULTISPECIES_TESTS_DF_PCKL = "tests/df_AlNi(murn).pckl.gzip"
TESTS_DF_PCKL = "tests/representative_df.pckl.gzip"
COMPRESSION = "gzip"


def create_block(NLMAX, NRADMAX, NRADBASE, NDENS):
    block = BBasisFunctionsSpecificationBlock()
    block.block_name = "Al"
    block.nradmaxi = NRADMAX
    block.lmaxi = NLMAX
    block.npoti = "FinnisSinclair"
    if NDENS == 2:
        block.fs_parameters = [1, 1, 1, 0.5]
    else:
        block.fs_parameters = [1, 1]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = NRADBASE
    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    # radcoefficients_len = (NLMAX + 1) * NRADBASE * NRADMAX
    block.radcoefficients = np.ones((NRADMAX, NLMAX + 1, NRADBASE))
    # self.bBasisSpecificationBlock = block
    return block


def prepare_test_basis_configuration_extended():
    NLMAX = 1
    NRADBASE = 2
    # NRADMAX = 1
    NRADMAX = 2

    bBasisConfiguration = BBasisConfiguration()
    bBasisConfiguration.deltaSplineBins = 0.001
    block = create_block(NLMAX=NLMAX, NRADMAX=NRADMAX, NRADBASE=NRADBASE, NDENS=1)
    block.funcspecs = [
        BBasisFunctionSpecification(elements=["Al", "Al"], ns=[1], ls=[0], coeffs=[1.]),
        BBasisFunctionSpecification(elements=["Al", "Al"], ns=[2], ls=[0], coeffs=[1.]),

        BBasisFunctionSpecification(elements=["Al", "Al", "Al"], ns=[1, 1], ls=[0], coeffs=[0.01]),
        BBasisFunctionSpecification(elements=["Al", "Al", "Al"], ns=[1, 2], ls=[1], coeffs=[0.01]),
        BBasisFunctionSpecification(elements=["Al", "Al", "Al"], ns=[2, 2], ls=[0], coeffs=[0.01]),
    ]
    bBasisConfiguration.funcspecs_blocks = [block]
    return bBasisConfiguration


def test_compute_active_set_by_batches_multispecies():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)

    elements_to_index_map = bbasis.elements_to_index_map
    cutoffmax = bbasis.cutoffmax

    generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    df = df.sample(frac=1, random_state=42)
    # batch_size = 50
    atomic_env_list = df["atomic_env"]
    structure_ind_list = df.index
    # nsplits = len(atomic_env_list) // batch_size
    # atomic_env_batches = np.array_split(atomic_env_list, nsplits)
    # structure_ind_batches = np.array_split(structure_ind_list, nsplits)
    # atomic_env_batches = [b.values for b in atomic_env_batches]

    # structure_ind_batches = [b.values for b in structure_ind_batches]

    (best_gamma, best_active_sets_dict, best_active_sets_si_dict) = \
        compute_active_set_by_batches(
            bbasis,
            atomic_env_list=atomic_env_list,
            structure_ind_list=structure_ind_list,
            n_batches=2,
        )
    print("best_gamma=", best_gamma)
    best_gamma_expected = {1: 1.0004078686401439, 0: 1.0000220624877256}
    active_set_shape_expected = {0: (9, 9), 1: (8, 8)}

    for st in [0, 1]:
        assert np.allclose(best_gamma_expected[st], best_gamma[st])
        print("best_active_sets_dict[{}].shape={}".format(st, best_active_sets_dict[st].shape))
        assert active_set_shape_expected[st] == best_active_sets_dict[st].shape


def test_compute_active_set_by_batches_multispecies_extra_projections():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)

    df_extra = df.sample(n=10)
    df = df.drop(df_extra.index).reset_index(drop=True)
    df_extra = df_extra.reset_index(drop=True)

    elements_to_index_map = bbasis.elements_to_index_map
    cutoffmax = bbasis.cutoffmax

    # generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    # generate_atomic_env_column(df_extra, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    atomic_env_list = df["atomic_env"]
    atomic_env_list_extra = df_extra["atomic_env"]

    # batch_size = 50

    structure_ind_list = df.index
    # nsplits = len(atomic_env_list) // batch_size

    # atomic_env_batches = np.array_split(atomic_env_list, nsplits)
    # atomic_env_batches = [b.values for b in atomic_env_batches]
    # structure_ind_batches = np.array_split(structure_ind_list, nsplits)
    # structure_ind_batches = [b.values for b in structure_ind_batches]

    A0_projections_dict_extra = compute_B_projections(bbasis, atomic_env_list=df_extra["ase_atoms"])

    (best_gamma, best_active_sets_dict, best_active_sets_si_dict) = \
        compute_active_set_by_batches(
            bbasis,
            atomic_env_list=atomic_env_list,
            structure_ind_list=structure_ind_list,
            n_batches=2,
            extra_A_active_set_dict=A0_projections_dict_extra
        )
    print("best_gamma=", best_gamma)
    best_gamma_expected = {0: 1.000022062491731, 1: 1.0000000000013989}
    active_set_shape_expected = {0: (9, 9), 1: (8, 8)}

    for st in [0, 1]:
        assert np.allclose(best_gamma_expected[st], best_gamma[st])
        print("best_active_sets_dict[{}].shape={}".format(st, best_active_sets_dict[st].shape))
        assert active_set_shape_expected[st] == best_active_sets_dict[st].shape


def test_compute_active_set_multispecies():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)

    elements_to_index_map = bbasis.elements_to_index_map
    cutoffmax = bbasis.cutoffmax

    generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    atomic_env_list = df["atomic_env"]

    A0_projections_dict = compute_B_projections(bbasis, atomic_env_list=atomic_env_list)
    assert isinstance(A0_projections_dict, dict)
    print("A0_projections_dict.keys=", A0_projections_dict.keys())

    best_active_sets_dict = compute_active_set(A0_projections_dict)

    active_set_shape_expected = {0: (9, 9), 1: (8, 8)}

    for st in [0, 1]:
        print("best_active_sets_dict[{}].shape={}".format(st, best_active_sets_dict[st].shape))
        assert active_set_shape_expected[st] == best_active_sets_dict[st].shape


def test_compute_active_set_multispecies_extra_projections():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df_extra = df.sample(n=10)
    df = df.drop(df_extra.index).reset_index(drop=True)
    df_extra = df_extra.reset_index(drop=True)

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)

    # generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    # generate_atomic_env_column(df_extra, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    atomic_env_list = df["atomic_env"]
    atomic_env_list_extra = df_extra["atomic_env"]

    A0_projections_dict, str_ind_dict = compute_B_projections(bbasis, atomic_env_list=df["ase_atoms"],
                                                              structure_ind_list=df.index)
    A0_projections_dict_extra = compute_B_projections(bbasis, atomic_env_list=df_extra["ase_atoms"])
    assert isinstance(A0_projections_dict, dict)
    assert isinstance(A0_projections_dict_extra, dict)
    print("A0_projections_dict.keys=", A0_projections_dict.keys())
    print("A0_projections_dict_extra.keys=", A0_projections_dict_extra.keys())

    best_active_sets_dict, sel_str_inds_dict = compute_active_set(A0_projections_dict, structure_ind_dict=str_ind_dict,
                                                                  extra_A0_projections_dict=A0_projections_dict_extra)

    active_set_shape_expected = {0: (9, 9), 1: (8, 8)}

    for st in [0, 1]:
        print("best_active_sets_dict[{}].shape={}".format(st, best_active_sets_dict[st].shape))
        assert active_set_shape_expected[st] == best_active_sets_dict[st].shape
        assert len(sel_str_inds_dict[st]) == len(best_active_sets_dict[st])


def test_compute_active_set_singlespecies():
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)

    bBasisConfiguration = prepare_test_basis_configuration_extended()
    bbasis = ACEBBasisSet(bBasisConfiguration)

    elements_to_index_map = bbasis.elements_to_index_map
    cutoffmax = bbasis.cutoffmax

    generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    atomic_env_list = df["atomic_env"]

    A0_projections_dict = compute_B_projections(bbasis, atomic_env_list=atomic_env_list)
    assert isinstance(A0_projections_dict, dict)
    print("A0_projections_dict.keys=", A0_projections_dict.keys())

    best_active_sets_dict = compute_active_set(A0_projections_dict)

    active_set_shape_expected = {0: (5, 5)}

    for st in [0]:
        print("best_active_sets_dict[{}].shape={}".format(st, best_active_sets_dict[st].shape))
        assert active_set_shape_expected[st] == best_active_sets_dict[st].shape


def test_compute_projections_multispecies_forces():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)

    elements_to_index_map = bbasis.elements_to_index_map
    cutoffmax = bbasis.cutoffmax

    generate_atomic_env_column(df, cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    atomic_env_list = df["atomic_env"]

    A0_projections_dict, forces_dict = compute_B_projections(bbasis, atomic_env_list=atomic_env_list,
                                                             compute_forces_dict=True)
    ref_forces_dict = extract_reference_forces_dict(df["ase_atoms"], df["forces"], elements_to_index_map)

    assert isinstance(A0_projections_dict, dict)
    print("A0_projections_dict.keys=", A0_projections_dict.keys())

    assert isinstance(forces_dict, dict)
    print("forces_dict.keys=", forces_dict.keys())

    forces_dict_shape_expected = {0: (231, 3), 1: (218, 3)}
    for st in [0, 1]:
        print("forces_dict[{}].shape={}".format(st, forces_dict[st].shape))
        print("A0_projections_dict[{}].shape={}".format(st, A0_projections_dict[st].shape))
        assert A0_projections_dict[st].shape[0] == forces_dict_shape_expected[st][0]
        assert forces_dict_shape_expected[st] == forces_dict[st].shape

    for st in [0, 1]:
        df = ref_forces_dict[st] - forces_dict[st]
        rmse = np.sqrt(np.mean(np.linalg.norm(df, axis=1) ** 2))
        print("st = {}, rmse={}".format(st, rmse))


def test_extract_reference_forces_dict():
    df = get_reference_dataset(PYACE_EVAL, MULTISPECIES_TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    bbasis = ACEBBasisSet(bBasisConfiguration)
    elements_to_index_map = bbasis.elements_to_index_map

    forces_dict = extract_reference_forces_dict(df["ase_atoms"], df["forces"], elements_to_index_map)
    assert isinstance(forces_dict, dict)
    print("forces_dict.keys=", forces_dict.keys())

    forces_dict_shape_expected = {0: (231, 3), 1: (218, 3)}
    for st in [0, 1]:
        print("forces_dict[{}].shape={}".format(st, forces_dict[st].shape))
        assert forces_dict_shape_expected[st] == forces_dict[st].shape

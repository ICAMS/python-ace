import numpy as np
import logging
import os
import pandas as pd
import sys

import pytest
from pyace.const import *
from pyace.preparedata import save_dataframe, compute_convexhull_dist, ExternalWeightingPolicy, ACEDataset, \
    EnergyBasedWeightingPolicy, apply_weights, adjust_aug_weights

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def test_ACEDataset():
    data_config = dict(
        filename="df-FHI-aims_PBE_tight-Al-ref.pckl.gzip",
        datapath="tests",
    )
    ace_ds = ACEDataset(data_config, evaluator_name='tensorpot', weighting_policy_spec=EnergyBasedWeightingPolicy())

    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    assert "w_energy" not in df0
    assert "w_forces" not in df0
    assert "w_factor" not in df0

    df = ace_ds.fitting_data
    print(df.columns)
    assert "w_energy" in df
    assert "w_forces" in df


def test_StructuresDatasetSpecification_weight_factor():
    # test that "w_factor" column remains in df after processing and final weights are multiplied by w_factor

    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")

    df1 = df0.copy()
    df1["w_factor"] = np.random.rand(len(df0))

    df0 = apply_weights(df0, EnergyBasedWeightingPolicy())

    print(df0.columns)
    assert "w_energy" in df0
    assert "w_forces" in df0
    assert "w_factor" not in df0

    df1 = apply_weights(df1, EnergyBasedWeightingPolicy())
    print(df1.columns)
    assert "w_energy" in df1
    assert "w_forces" in df1
    assert "w_factor" in df1
    assert np.all(df1["w_energy"].values != df0["w_energy"].values)
    assert np.all(np.hstack(df1["w_forces"]) != np.hstack(df0["w_forces"]))


def test_EnergyBasedWeightingPolicy_n_upper_n_lower():
    """ Test that for EnergyBasedWeightingPolicy(n_lower=1, n_upper=1)  only 2 structures remain"""
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0))
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    dfsel = apply_weights(df0, EnergyBasedWeightingPolicy(n_lower=1, n_upper=1))
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 2


def test_EnergyBasedWeightingPolicy_n_lower_high():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0))  # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=1)
    dfsel = apply_weights(df0, weights_policy)
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 4


def test_EnergyBasedWeightingPolicy_n_lower_high_nupper_0():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0))  # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=0)
    dfsel = apply_weights(df0, weights_policy)
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 3


def test_EnergyBasedWeightingPolicy_nlower_high_nupper_high():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0))  # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=100)
    dfsel = apply_weights(df0, weights_policy)
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 6


@pytest.mark.parametrize("fname",
                         ["df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", "df_AlNi(murn).pckl.gzip"])
def test_compute_convexhull_dist(fname):
    df0 = pd.read_pickle("tests/" + fname, compression="gzip")

    print("columns=", df0.columns.tolist())
    assert "e_chull_dist_per_atom" not in df0.columns.tolist()
    df = df0
    compute_convexhull_dist(df, energy_per_atom_column="energy_corrected_per_atom")
    assert "e_chull_dist_per_atom" in df0.columns.tolist()


def test_save_dataframe():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    filename = "tests/test_data/test_save_dataframe.pckl.gzip"
    save_dataframe(df0, filename)
    assert os.path.isfile(filename)


def test_ExternalWeightingPolicy():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print("columns=", df0.columns.tolist())
    assert sorted(df0.columns.tolist()) == sorted(
        ['prop_id', 'structure_id', 'gen_id', 'PROTOTYPE_NAME', 'COORDINATES_TYPE', '_COORDINATES', '_OCCUPATION',
         'NUMBER_OF_ATOMS', '_VALUE', 'pbc', 'energy', 'forces', 'energy_corrected', 'energy_corrected_per_atom',
         'cell', 'ase_atoms', 'atomic_env', 'tp_atoms']

    )

    df0[EWEIGHTS_COL] = 1.0
    df0[FWEIGHTS_COL] = df0["NUMBER_OF_ATOMS"].map(lambda nat: np.ones(nat))

    wdf = df0[[EWEIGHTS_COL, FWEIGHTS_COL]].sample(2, random_state=42)
    wdf.to_pickle("tests/wdf.pckl.gzip", compression="gzip", protocol=4)

    wpol = ExternalWeightingPolicy("tests/wdf.pckl.gzip")

    df = apply_weights(df0, weighting_policy_spec=wpol, ignore_weights=True)
    print("shape:", df.shape)
    print("columns:", df.columns)
    assert len(df) == 2
    assert len(df0) > 2
    assert "w_energy" in df
    assert "w_forces" in df

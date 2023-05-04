import numpy as np
import pytest
from pyace import *
from pyace.const import *
from pyace.preparedata import save_dataframe, compute_convexhull_dist, ExternalWeightingPolicy
import logging
import os
import pandas as pd
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def test_StructuresDatasetSpecification():
    data_config = dict(element="Al", calculator=StructuresDatasetSpecification.FHI_AIMS_PBE_TIGHT, )
    dataset_spec = StructuresDatasetSpecification(config=data_config,
                                                  datapath="tests",
                                                  parallel=False)
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy()
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print("columns=", df0.columns.tolist())
    assert sorted(df0.columns.tolist()) == sorted(
        ['prop_id', 'structure_id', 'gen_id', 'PROTOTYPE_NAME', 'COORDINATES_TYPE', '_COORDINATES', '_OCCUPATION',
         'NUMBER_OF_ATOMS', '_VALUE', 'pbc', 'energy', 'forces', 'energy_corrected', 'energy_corrected_per_atom',
         'cell', 'ase_atoms', 'atomic_env', 'tp_atoms']

    )

    df = dataset_spec.get_fit_dataframe()
    print(df.columns)
    assert "w_energy" in df
    assert "w_forces" in df


def test_StructuresDatasetSpecification_weight_factor():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")

    df1 = df0.copy()
    df1["w_factor"] = np.random.rand(len(df0))

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df0
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy()
    df0 = dataset_spec.get_fit_dataframe()
    print(df0.columns)
    assert "w_energy" in df0
    assert "w_forces" in df0
    assert "w_factor" not in df0

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df1
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy()
    df1 = dataset_spec.get_fit_dataframe()
    print(df1.columns)
    assert "w_energy" in df1
    assert "w_forces" in df1
    assert "w_factor" in df1
    assert np.all(df1["w_energy"].values != df0["w_energy"].values)
    assert np.all(np.hstack(df1["w_forces"]) != np.hstack(df0["w_forces"]))


def test_EnergyBasedWeightingPolicy_n_upper_n_lower():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0))
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df0
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy(n_lower=1, n_upper=1)
    dfsel = dataset_spec.get_fit_dataframe()
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 2

def test_EnergyBasedWeightingPolicy_n_lower_high():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0)) # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df0
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=1)
    dfsel = dataset_spec.get_fit_dataframe()
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 4

def test_EnergyBasedWeightingPolicy_n_lower_high_nupper_0():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0)) # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df0
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=0)
    dfsel = dataset_spec.get_fit_dataframe()
    print(dfsel.columns)
    assert "w_energy" in dfsel
    assert "w_forces" in dfsel
    assert len(dfsel) == 3

def test_EnergyBasedWeightingPolicy_nlower_high_nupper_high():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    print(len(df0)) # 6
    print(df0["energy_corrected_per_atom"].sort_values())
    assert "w_energy" not in df0
    assert "w_forces" not in df0

    dataset_spec = StructuresDatasetSpecification()
    dataset_spec.df = df0
    dataset_spec.weights_policy = EnergyBasedWeightingPolicy(n_lower=100, n_upper=100)
    dfsel = dataset_spec.get_fit_dataframe()
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
    compute_convexhull_dist(df)
    assert "e_chull_dist_per_atom" in df0.columns.tolist()


def test_save_dataframe():
    df0 = pd.read_pickle("tests/df-FHI-aims_PBE_tight-Al-ref.pckl.gzip", compression="gzip")
    filename = "tests/test_data/test_save_dataframe.pckl.gzip"
    save_dataframe(df0, filename)
    assert os.path.isfile(filename)


def test_ExternalWeightingPolicy():
    data_config = dict(element="Al", calculator=StructuresDatasetSpecification.FHI_AIMS_PBE_TIGHT, )
    dataset_spec = StructuresDatasetSpecification(config=data_config,
                                                  datapath="tests",
                                                  parallel=False)
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

    df = dataset_spec.get_fit_dataframe(weights_policy=wpol)
    print("shape:", df.shape)
    print("columns:", df.columns)
    assert len(df) == 2
    assert len(df0) > 2
    assert "w_energy" in df
    assert "w_forces" in df

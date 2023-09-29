import os
import pandas as pd
import numpy as np
import pytest
from pyace import *
from pyace.pyacefit import LossFunctionSpecification
from pyace.pyacefit import PyACEFit
from pyace.preparedata import generate_atomic_env_column, get_reference_dataset
from pyace.linearacefit import LinearACEDataset, LinearACEFit
from pyace.const import PYACE_EVAL

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


def prepare_test_basis_configuration():
    NLMAX = 0
    NRANKMAX = 1
    NRADBASE = 1
    NRADMAX = 0
    NDENS = 1
    bBasisConfiguration = BBasisConfiguration()
    bBasisConfiguration.deltaSplineBins = 0.001
    block = create_block(NLMAX=NLMAX, NRADMAX=NRADMAX, NRADBASE=NRADBASE, NDENS=NDENS)
    block.funcspecs = [
        BBasisFunctionSpecification(elements=["Al", "Al"], ns=[1], ls=[0], coeffs=[1.] * NDENS)
    ]
    bBasisConfiguration.funcspecs_blocks = [block]
    return bBasisConfiguration


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


def test_linear_fit():
    df_train = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)

    bconf = prepare_test_basis_configuration()

    train_ds = LinearACEDataset(bconf, df_train)
    train_ds.construct_design_matrix(verbose=True, max_workers=2)
    linear_fit = LinearACEFit(train_dataset=train_ds)
    linear_fit.fit()
    errors = linear_fit.compute_errors(train_ds)
    print(errors)
    errors_ref = {'epa_mae': 1.8672420164907269, 'epa_rmse': 2.161275544266956, 'f_comp_mae': 0.9338507863298846,
                  'f_comp_rmse': 1.8036620237453211}
    for k, ref_e in errors_ref.items():
        assert np.allclose(errors[k], ref_e), k

    basis = linear_fit.get_bbasis()
    calc = PyACECalculator(basis)

    e_pred, f_pred = linear_fit.predict(train_ds, reshape_forces=True)
    print("e_pred=", e_pred)
    print("f_pred=", f_pred)
    e_pred_ref = [0.21383798, 0.16503755, 0.01162006, 0.2028525, 0.21184446, 0.01921261]
    f_pred0_ref = [-1.37039297e-02,  3.71525064e-02,  4.82533231e-03]
    assert np.allclose(e_pred, e_pred_ref)
    assert np.allclose(f_pred[0], f_pred0_ref)
    # take first  ase_atoms
    at = df_train.iloc[0]["ase_atoms"].copy()
    at.set_calculator(calc)

    e_pred_calc = at.get_potential_energy()
    print(e_pred_calc)

    assert np.allclose(e_pred_calc / len(at), e_pred[0])
    assert np.allclose(at.get_forces(), f_pred[:len(at)])

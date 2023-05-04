import os
import pandas as pd
import numpy as np
import pytest
from pyace import *
from pyace.pyacefit import LossFunctionSpecification
from pyace.pyacefit import PyACEFit
from pyace.preparedata import generate_atomic_env_column, get_reference_dataset

from ase.build import bulk

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


def test_fit_serial():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = prepare_test_basis_configuration()
    loss_spec = LossFunctionSpecification(kappa=0.1, w1_coeffs=0, )
    fit = PyACEFit(bBasisConfiguration,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"))
    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 2, "disp": True})

    print("fit._losses=", fit.last_loss)
    print("fit.best_params=", fit.best_params)
    assert np.allclose(fit.last_loss, 906.0412315025584)
    assert np.allclose(fit.best_params, [0.9])


def test_fit_serial2():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = prepare_test_basis_configuration_extended()

    fit = PyACEFit(bBasisConfiguration,

                   loss_spec=LossFunctionSpecification(kappa=0.1),
                   executors_kw_args=dict(parallel_mode="serial"))
    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 5, "disp": True})

    print("fit.lasst_loss=", fit.last_loss)
    print("fit.best_params=", fit.best_params)
    assert np.allclose(fit.best_params, [1.01153846, 1.01153846, 1.01153846, 1.01153846, 1.01153846,
                                         1.01153846, 1.01153846, 1.01153846, 0.9, 1.01153846,
                                         0.01011538, 0.01011538, 0.01011538])


def test_fit_process():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    bBasisConfiguration = prepare_test_basis_configuration()
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    fit = PyACEFit(bBasisConfiguration,

                   loss_spec=LossFunctionSpecification(kappa=0.1),
                   executors_kw_args=dict(parallel_mode="process"))
    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 2, "disp": True})

    print("fit.last_loss=", fit.last_loss)
    print("fit.best_params=", fit.best_params)
    assert np.allclose(fit.last_loss, 906.0412315025584)
    assert np.allclose(fit.best_params, [0.9])


def test_fit_process2():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = prepare_test_basis_configuration_extended()

    fit = PyACEFit(bBasisConfiguration,

                   loss_spec=LossFunctionSpecification(kappa=0.1),
                   executors_kw_args=dict(parallel_mode="process"))
    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 5, "disp": True})

    print("fit.last_loss=", fit.last_loss)
    print("fit.best_params=", fit.best_params)
    assert np.allclose(fit.last_loss, 2442.6636983990384)
    assert np.allclose(fit.best_params, [1.01153846, 1.01153846, 1.01153846, 1.01153846, 1.01153846,
                                         1.01153846, 1.01153846, 1.01153846, 0.9, 1.01153846,
                                         0.01011538, 0.01011538, 0.01011538])


def test_fit_value_error_nrad():
    df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    bBasisConfiguration = prepare_test_basis_configuration()
    block = bBasisConfiguration.funcspecs_blocks[0]
    block.nradbaseij = 2

    with pytest.raises(ValueError):
        block.funcspecs += [
            BBasisFunctionSpecification(elements=["Al", "Al"], ns=[2], ls=[0], coeffs=[1, 2])
        ]


def test_fit_contiunation():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = prepare_test_basis_configuration()
    loss_spec = LossFunctionSpecification(kappa=0.1, w1_coeffs=0, )
    fit = PyACEFit(bBasisConfiguration,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"))
    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 20, "disp": True})
    print("fit.last_loss=", fit.last_loss)
    print("fit.params_opt=", fit.params_opt)
    fit.bbasis_opt.save("cont.yaml")

    all_params1 = fit.bbasis_opt.all_coeffs
    ef1 = fit.predict_energy_forces(fit.best_params)

    bBasisConfiguration_cont = BBasisConfiguration("cont.yaml")

    loss_spec = LossFunctionSpecification(kappa=0.1, w1_coeffs=0, )
    fit2 = PyACEFit(bBasisConfiguration_cont,
                    loss_spec=loss_spec,
                    executors_kw_args=dict(parallel_mode="serial"))
    all_params_cont = fit2.bbasis.all_coeffs
    assert all_params1 == all_params_cont

    fit2.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 20, "disp": True})
    print("fit2.last_loss=", fit2.last_loss)

    assert np.allclose(fit.last_loss, fit2.initial_loss)


def test_fit_predict_energy():
    path = "tests"
    fname = "Al.pbe.13.2.yaml"

    a_lst = np.linspace(2, 9, num=5)
    atoms_lst = [bulk("Al", "fcc", a=a) for a in a_lst]
    basis_conf = BBasisConfiguration(os.path.join(path, fname))
    bbasis = ACEBBasisSet(basis_conf)

    ae_lst = []
    for atoms in atoms_lst:
        ae = aseatoms_to_atomicenvironment(atoms)
        ae_lst.append(ae)

    df = pd.DataFrame({"a": a_lst, "atoms": atoms_lst, "atomic_env": ae_lst})
    fit = PyACEFit(basis_conf)

    ef_df = fit.predict_energy_forces(structures_dataframe=df)

    print(ef_df.shape, ef_df)
    assert ef_df.shape == (5, 2)


def test_fit_predict_bbasis_projections():
    path = "tests"
    fname = "Al.pbe.13.2.yaml"

    a_lst = np.linspace(2, 9, num=5)
    atoms_lst = [bulk("Al", "fcc", a=a) for a in a_lst]
    basis_conf = BBasisConfiguration(os.path.join(path, fname))
    bbasis = ACEBBasisSet(basis_conf)

    ae_lst = []
    for atoms in atoms_lst:
        ae = aseatoms_to_atomicenvironment(atoms)
        ae_lst.append(ae)

    df = pd.DataFrame({"a": a_lst, "atoms": atoms_lst, "atomic_env": ae_lst})
    fit = PyACEFit(basis_conf)
    proj_df = fit.predict_projections(structures_dataframe=df)

    proj_df = pd.DataFrame({"proj": proj_df})
    proj_df["single_proj"] = proj_df["proj"].map(lambda p: p[0])
    proj_df = proj_df.drop("proj", axis=1)
    block = basis_conf.funcspecs_blocks[0]
    funcs = block.funcspecs
    print("len funcs=", len(funcs))
    tot_df = pd.concat([df, proj_df, pd.DataFrame(np.vstack(proj_df["single_proj"]))], axis=1)
    tot_df.drop(["single_proj", "atoms", "atomic_env"], axis=1, inplace=True)

    print(tot_df.shape)
    assert tot_df.shape == (5, len(funcs) + 1)


def test_fit_predict_bbasis_config_projections():
    path = "tests"
    fname = "Al.pbe.13.2.yaml"

    a_lst = np.linspace(2, 9, num=5)
    atoms_lst = [bulk("Al", "fcc", a=a) for a in a_lst]
    basis_conf = BBasisConfiguration(os.path.join(path, fname))

    ae_lst = []
    for atoms in atoms_lst:
        ae = aseatoms_to_atomicenvironment(atoms)
        ae_lst.append(ae)

    df = pd.DataFrame({"a": a_lst, "atoms": atoms_lst, "atomic_env": ae_lst})
    fit = PyACEFit(basis_conf)
    proj_df = fit.predict_projections(params=basis_conf, structures_dataframe=df)

    proj_df = pd.DataFrame({"proj": proj_df})
    proj_df["single_proj"] = proj_df["proj"].map(lambda p: p[0])
    proj_df = proj_df.drop("proj", axis=1)
    block = basis_conf.funcspecs_blocks[0]
    funcs = block.funcspecs
    print("len funcs=", len(funcs))
    tot_df = pd.concat([df, proj_df, pd.DataFrame(np.vstack(proj_df["single_proj"]))], axis=1)
    tot_df.drop(["single_proj", "atoms", "atomic_env"], axis=1, inplace=True)

    print(tot_df.shape)
    assert tot_df.shape == (5, len(funcs) + 1)


# def test_fit_predict_cbasis_projections():
#     path = "tests"
#     fname = "Al.pbe.13.2.yaml"
#
#     a_lst = np.linspace(2, 9, num=5)
#     atoms_lst = [bulk("Al", "fcc", a=a) for a in a_lst]
#     basis_conf = BBasisConfiguration(os.path.join(path, fname))
#     bbasis = ACEBBasisSet(basis_conf)
#     cbasis = bbasis.to_ACECTildeBasisSet()
#
#     ae_lst = []
#     for atoms in atoms_lst:
#         ae = aseatoms_to_atomicenvironment(atoms)
#         ae_lst.append(ae)
#
#     df = pd.DataFrame({"a": a_lst, "atoms": atoms_lst, "atomic_env": ae_lst})
#     fit = PyACEFit(basis_conf)
#     proj_df = fit.predict_projections(structures_dataframe=df, params=cbasis)
#
#     proj_df = pd.DataFrame({"proj": proj_df})
#     proj_df["single_proj"] = proj_df["proj"]#.map(lambda p: p[0][::2])
#     proj_df = proj_df.drop("proj", axis=1)
#     block = basis_conf.funcspecs_blocks[0]
#     funcs = block.funcspecs
#     print("len funcs=", len(funcs))
#     tot_df = pd.concat([df, proj_df, pd.DataFrame(np.vstack(proj_df["single_proj"]))], axis=1)
#     tot_df.drop(["single_proj", "atoms", "atomic_env"], axis=1, inplace=True)
#
#     print(tot_df.shape)
#     assert tot_df.shape == (5, len(funcs) + 1)


def test_fit_predict_bbasis_config_multispecies_projections():
    path = "tests"
    fname = "Al-Ni_opt_all.yaml"

    a_lst = np.linspace(2, 9, num=5)
    atoms_lst = []
    for a in a_lst:
        at = bulk("Al", "fcc", a=a, cubic=True)
        at.set_chemical_symbols(["Al", "Al", "Ni", "Ni"])
        atoms_lst.append(at)

    basis_conf = BBasisConfiguration(os.path.join(path, fname))

    df = pd.DataFrame({"a": a_lst, "ase_atoms": atoms_lst})
    generate_atomic_env_column(df)

    fit = PyACEFit(basis_conf)
    projs_raw = fit.predict_projections(structures_dataframe=df)

    assert len(projs_raw) == len(df)
    assert (projs_raw.map(len) == df["ase_atoms"].map(len)).all()

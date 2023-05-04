import os
import pandas as pd
import numpy as np
import pytest
from pyace import *
from pyace.pyacefit import LossFunctionSpecification
from pyace.pyacefit import PyACEFit

from pyace.const import PYACE_EVAL
from pyace.preparedata import get_reference_dataset

TESTS_DF_PCKL = "tests/df_AlNi(murn).pckl.gzip"
COMPRESSION = "gzip"

# df_mod = pd.read_pickle(TESTS_DF_PCKL, compression="gzip")
df_mod = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
df_mod["energy_per_atom"] = df_mod["energy_corrected"] / df_mod["NUMBER_OF_ATOMS"]
df_mod["w_energy"] = 1.
df_mod["w_forces"] = df_mod["forces"].map(lambda f: np.ones(len(f)))
df_mod["species"] = df_mod['ase_atoms'].map(lambda at: tuple(sorted(set(at.get_chemical_symbols()))))
df_Al = df_mod[df_mod["species"] == ("Al",)].copy()
df_Ni = df_mod[df_mod["species"] == ("Ni",)].copy()
df_AlNi = df_mod[df_mod["species"] == ("Al", "Ni",)].copy()


def test_fit_multispecies():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")
    loss_spec = LossFunctionSpecification(kappa=0.1, w1_coeffs=0, )
    fit = PyACEFit(bBasisConfiguration,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"))

    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 5, "disp": True})

    print("fit._losses=", fit.last_loss)
    print("fit.best_params=", fit.best_params)
    assert np.allclose(fit.last_loss, 388402.1239189262)
    assert np.allclose(fit.best_params[:5], [1.00394737, 2.00789474, 1.00394737, 2.00789474, 1.00394737])


def test_fit_multispecies_fit_pair_blocks():
    # df = pd.read_pickle(TESTS_DF_PCKL, compression=COMPRESSION)
    df = get_reference_dataset(PYACE_EVAL, TESTS_DF_PCKL)
    print("train dataset shape:", df.shape)
    df["fweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.ones(n))
    df["eweights"] = df["NUMBER_OF_ATOMS"].map(lambda n: np.array([1.]).reshape(-1, ))
    df['w_forces'] = df["fweights"].copy()
    df['w_energy'] = df["eweights"].copy()

    bBasisConfiguration = BBasisConfiguration("tests/multispecies_AlNi.yaml")

    trainable_parameters = ["AlNi", "NiAl"]

    old_params_dict = {}

    for block_ind, block in enumerate(bBasisConfiguration.funcspecs_blocks):
        print("block.block_name: ", block.block_name)
        print("block.all_coeffs: ", block.get_all_coeffs())
        old_params_dict[block.block_name] = block.get_all_coeffs()


    loss_spec = LossFunctionSpecification(kappa=0.1, w1_coeffs=0, )
    fit = PyACEFit(bBasisConfiguration,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"),
                   trainable_parameters=trainable_parameters)

    fit.fit(structures_dataframe=df, method="Nelder-Mead", options={"maxiter": 5, "disp": True})

    print("Optimized parameter")
    bbasspec_opt = fit.bbasis_opt.to_BBasisConfiguration()
    new_params_dict = {}
    for block_ind, block in enumerate(bbasspec_opt.funcspecs_blocks):
        print("block.block_name: ", block.block_name)
        print("block.all_coeffs: ", block.get_all_coeffs())
        new_params_dict[block.block_name] = block.get_all_coeffs()

    assert not np.allclose(new_params_dict["Ni Al"], old_params_dict["Ni Al"])
    assert not np.allclose(new_params_dict["Al Ni"], old_params_dict["Al Ni"])

    assert np.allclose(new_params_dict["Al"], old_params_dict["Al"])
    assert np.allclose(new_params_dict["Ni"], old_params_dict["Ni"])


def test_fit_2comp_comp0_data0():
    tp_initial_loss = 225.6579298049239 #188076.07407006648
    potential_config = {
        # Step 0. define deltaSplineBins
        'deltaSplineBins': 0.001,

        # Step 1. specify all elements of the basis
        'elements': ['Al', 'Ni'],

        # Step 2. specify embeddings for all elements, using 'ALL' or elements name keywords
        'embeddings': {'ALL': {
            'ndensity': 1,
            # 'fs_parameters': [1, 1, 1, 0.5],
            'fs_parameters': [1, 1],

            'npot': 'FinnisSinclairShiftedScaled',

            'rho_core_cut': 200000.,
            'drho_core_cut': 250,
        },

        },

        # Step 3. specify bonds for all elements, using 'ALL', UNARY, BINARY or elements pairs
        'bonds': {'ALL': {
            'radbase': 'ChebPow',
            'radparameters': [2.0],
            'rcut': 5,
            'dcut': 0.01,
            'NameOfCutoffFunction': 'cos',
            'core-repulsion': [20000.0, 5.0],
        },
        },

        # Step 4. Specify BBasisFunctions list for each block using ALL, UNARY, BINARY, ..., QUINARY keywords
        # setup per-rank nradmax_by_orders and lmax_by_orders
        'functions': {
            'ALL': {
                'nradmax_by_orders': [2, 2],
                'lmax_by_orders': [0, 1]
            },
        }
    }

    np.random.seed(42)
    bbasisconf_Al = create_multispecies_basis_config(potential_config, func_coefs_initializer="random")
    df = df_Al
    loss_spec = LossFunctionSpecification(kappa=0.5)
    fit = PyACEFit(bbasisconf_Al,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"), )
    method = 'BFGS'
    fit.fit(structures_dataframe=df, method=method, options={"maxiter": 1, "disp": True})
    print("initial loss = ", fit.initial_loss)
    print("last_loss = ", fit.last_loss)
    c = fit.best_params
    print("c=", c)
    assert np.abs(fit.initial_loss - tp_initial_loss) < 5e-9


def test_fit_2comp_comp0_data0_smooth():
    tp_initial_loss = 229.23284317339986 #188076.07407006648

    potential_config = {
        # Step 0. define deltaSplineBins
        'deltaSplineBins': 0.001,

        # Step 1. specify all elements of the basis
        'elements': ['Al', 'Ni'],

        # Step 2. specify embeddings for all elements, using 'ALL' or elements name keywords
        'embeddings': {'ALL': {
            'ndensity': 1,
            # 'fs_parameters': [1, 1, 1, 0.5],
            'fs_parameters': [1, 1],

            'npot': 'FinnisSinclairShiftedScaled',

            'rho_core_cut': 200000.,
            'drho_core_cut': 250,
        },

        },

        # Step 3. specify bonds for all elements, using 'ALL', UNARY, BINARY or elements pairs
        'bonds': {'ALL': {
            'radbase': 'ChebPow',
            'radparameters': [2.0],
            'rcut': 5,
            'dcut': 0.01,
            'NameOfCutoffFunction': 'cos',
            'core-repulsion': [20000.0, 5.0],
        },
        },

        # Step 4. Specify BBasisFunctions list for each block using ALL, UNARY, BINARY, ..., QUINARY keywords
        # setup per-rank nradmax_by_orders and lmax_by_orders
        'functions': {
            'ALL': {
                'nradmax_by_orders': [2, 2],
                'lmax_by_orders': [0, 1]
            },
        }
    }

    np.random.seed(42)
    bbasisconf_Al = create_multispecies_basis_config(potential_config, func_coefs_initializer="random")
    # bbasisconf_Al.save('before.yaml')
    df = df_Al
    loss_spec = LossFunctionSpecification(kappa=0.5, w0_rad=1, w1_rad=1, w2_rad=1)
    fit = PyACEFit(bbasisconf_Al,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"), )

    # potential = ACE(bbasisconf_Al, compute_smoothness=True)
    # tp = TensorPotential(potential, loss_specs={
    #     LOSS_TYPE: 'per-atom',
    #     LOSS_FORCE_FACTOR: 0.5,
    #     LOSS_ENERGY_FACTOR: 0.5,
    #     L1_REG: 0.,
    #     L2_REG: 0.,
    #     AUX_LOSS_FACTOR: [1., 1., 1.]
    # })

    method = 'BFGS'
    fit.fit(structures_dataframe=df, method=method, options={"maxiter": 1, "disp": True})
    print("initial loss = ", fit.initial_loss)
    print("last_loss = ", fit.last_loss)
    c = fit.best_params
    print("c=", c)
    assert np.abs(fit.initial_loss - tp_initial_loss) < 3e-2


def test_fit_2comp_comp1_data1():
    tp_initial_loss = 457.3375354763678

    potential_config = {
        # Step 0. define deltaSplineBins
        'deltaSplineBins': 0.001,

        # Step 1. specify all elements of the basis
        'elements': ['Al', 'Ni'],

        # Step 2. specify embeddings for all elements, using 'ALL' or elements name keywords
        'embeddings': {'ALL': {
            'ndensity': 1,
            # 'fs_parameters': [1, 1, 1, 0.5],
            'fs_parameters': [1, 1],

            'npot': 'FinnisSinclairShiftedScaled',

            'rho_core_cut': 200000.,
            'drho_core_cut': 250,
        },

        },

        # Step 3. specify bonds for all elements, using 'ALL', UNARY, BINARY or elements pairs
        'bonds': {'ALL': {
            'radbase': 'ChebPow',
            'radparameters': [2.0],
            'rcut': 5,
            'dcut': 0.01,
            'NameOfCutoffFunction': 'cos',
            'core-repulsion': [20000.0, 5.0],
            'inner_cutoff_type':'density'
        },
        },

        # Step 4. Specify BBasisFunctions list for each block using ALL, UNARY, BINARY, ..., QUINARY keywords
        # setup per-rank nradmax_by_orders and lmax_by_orders
        'functions': {
            'ALL': {
                'nradmax_by_orders': [2, 2],
                'lmax_by_orders': [0, 1]
            },
        }
    }

    np.random.seed(42)
    bbasisconf_AlNi = create_multispecies_basis_config(potential_config, func_coefs_initializer="random")
    df = df_Ni
    # tp = TensorPotential(potential, loss_specs={
    #     LOSS_TYPE: 'per-atom',
    #     LOSS_FORCE_FACTOR: 0.5,
    #     LOSS_ENERGY_FACTOR: 0.5,
    #     L1_REG: 0.,
    #     L2_REG: 0.
    # })
    # tpf = FitTensorPotential(tp)
    # tpf.fit(df, niter=15, batch_size=40)

    loss_spec = LossFunctionSpecification(kappa=0.5)
    fit = PyACEFit(bbasisconf_AlNi,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"), )
    method = 'BFGS'
    fit.fit(structures_dataframe=df, method=method, options={"maxiter": 1, "disp": True})
    print("initial loss = ", fit.initial_loss)
    print("last_loss = ", fit.last_loss)
    c = fit.best_params
    print("c=", c)
    assert np.abs(fit.initial_loss - tp_initial_loss) < 1e-8


def test_fit_2comp_compall_data_all():
    tp_initial_loss = 1986.2212571535736
    potential_config = {
        # Step 0. define deltaSplineBins
        'deltaSplineBins': 0.001,

        # Step 1. specify all elements of the basis
        'elements': ['Al', 'Ni'],

        # Step 2. specify embeddings for all elements, using 'ALL' or elements name keywords
        'embeddings': {'ALL': {
            'ndensity': 1,
            # 'fs_parameters': [1, 1, 1, 0.5],
            'fs_parameters': [1, 1],

            'npot': 'FinnisSinclairShiftedScaled',

            'rho_core_cut': 200000.,
            'drho_core_cut': 250,
        },

        },

        # Step 3. specify bonds for all elements, using 'ALL', UNARY, BINARY or elements pairs
        'bonds': {'ALL': {
            'radbase': 'ChebPow',
            'radparameters': [2.0],
            'rcut': 5,
            'dcut': 0.01,
            'NameOfCutoffFunction': 'cos',
            'core-repulsion': [20000.0, 5.0],
            'inner_cutoff_type':'density'
        },
        },

        # Step 4. Specify BBasisFunctions list for each block using ALL, UNARY, BINARY, ..., QUINARY keywords
        # setup per-rank nradmax_by_orders and lmax_by_orders
        'functions': {
            'ALL': {
                'nradmax_by_orders': [2, 2],
                'lmax_by_orders': [0, 1]
            },
        }
    }

    np.random.seed(42)
    bbasisconf_AlNi = create_multispecies_basis_config(potential_config, func_coefs_initializer="random")
    df = df_mod

    # potential = ACE(bbasisconf_AlNi)
    # tp = TensorPotential(potential, loss_specs={
    #     LOSS_TYPE: 'per-atom',
    #     LOSS_FORCE_FACTOR: 0.5,
    #     LOSS_ENERGY_FACTOR: 0.5,
    #     L1_REG: 0.,
    #     L2_REG: 0.
    # })
    # tpf = FitTensorPotential(tp)
    # tpf.fit(df, niter=15, batch_size=40)

    loss_spec = LossFunctionSpecification(kappa=0.5)
    fit = PyACEFit(bbasisconf_AlNi,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"), )
    method = 'BFGS'
    fit.fit(structures_dataframe=df, method=method, options={"maxiter": 1, "disp": True})
    print("initial loss = ", fit.initial_loss)
    print("last_loss = ", fit.last_loss)
    c = fit.best_params
    print("c=", c)
    assert np.abs(fit.initial_loss - tp_initial_loss) < 1e-8


def test_fit_ace_with_l1_l2_reg():
    tp_initial_loss = 1986.2230679875538

    potential_config = {
        # Step 0. define deltaSplineBins
        'deltaSplineBins': 0.001,

        # Step 1. specify all elements of the basis
        'elements': ['Al', 'Ni'],

        # Step 2. specify embeddings for all elements, using 'ALL' or elements name keywords
        'embeddings': {'ALL': {
            'ndensity': 1,
            # 'fs_parameters': [1, 1, 1, 0.5],
            'fs_parameters': [1, 1],

            'npot': 'FinnisSinclairShiftedScaled',

            'rho_core_cut': 200000.,
            'drho_core_cut': 250,
        },

        },

        # Step 3. specify bonds for all elements, using 'ALL', UNARY, BINARY or elements pairs
        'bonds': {'ALL': {
            'radbase': 'ChebPow',
            'radparameters': [2.0],
            'rcut': 5,
            'dcut': 0.01,
            'NameOfCutoffFunction': 'cos',
            'core-repulsion': [20000.0, 5.0],
            'inner_cutoff_type':'density'
        },
        },

        # Step 4. Specify BBasisFunctions list for each block using ALL, UNARY, BINARY, ..., QUINARY keywords
        # setup per-rank nradmax_by_orders and lmax_by_orders
        'functions': {
            'ALL': {
                'nradmax_by_orders': [2, 2],
                'lmax_by_orders': [0, 1]
            },
        }
    }

    np.random.seed(42)
    bbasisconf_Al = create_multispecies_basis_config(potential_config, func_coefs_initializer="random")
    df = df_mod
    print("df.shape = ", df.shape)

    loss_spec = LossFunctionSpecification(kappa=0.5,
                                          L1_coeffs=0.1 * 5,
                                          L2_coeffs=0.05 * 5,
                                          w1_coeffs=1.0,
                                          w2_coeffs=1.0)

    fit = PyACEFit(bbasisconf_Al,
                   loss_spec=loss_spec,
                   executors_kw_args=dict(parallel_mode="serial"), )

    method = 'BFGS'
    fit.fit(structures_dataframe=df, method=method, options={"maxiter": 1, "disp": True})
    print("initial loss = ", fit.initial_loss)
    print("last_loss = ", fit.last_loss)
    c = fit.best_params
    print("c=", c)
    assert np.abs(fit.initial_loss - tp_initial_loss) < 5e-7

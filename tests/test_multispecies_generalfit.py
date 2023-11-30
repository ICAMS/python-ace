import pytest

import numpy as np
from pyace.basis import ACEBBasisSet
from pyace.generalfit import GeneralACEFit, BBasisConfiguration
from pyace.multispecies_basisextension import create_multispecies_basis_config, expand_trainable_parameters

test_fit_config_L1_L2 = {
    'optimizer': 'L-BFGS-B',
    'maxiter': 2,

    'loss': {
        'kappa': 0.5,
    },
}

simple_potential_config = {
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

test_Al_data_config = {'filename': 'tests/representative_df.pckl.gzip'}

TESTS_DF_PCKL = "tests/df_AlNi(murn).pckl.gzip"
test_AlNi_data_config = {'filename': TESTS_DF_PCKL}


def compare_coefficients(all_coeffs, all_coeffs_ref, abs_threshold=2e-7, rel_threshold=2e-7):
    all_coeffs = np.array(all_coeffs)
    all_coeffs_ref = np.array(all_coeffs_ref)
    print("all_coeffs=", list(all_coeffs))
    dcoeff = np.abs((all_coeffs - all_coeffs_ref))
    print("abs diff = ", dcoeff)
    print("abs diff max = ", np.max(dcoeff))
    print("abs diff norm = ", np.linalg.norm(dcoeff))

    rel_err = 2 * dcoeff / (all_coeffs + all_coeffs_ref)
    print("rel_err = ", rel_err)
    rel_norm = np.linalg.norm(rel_err)
    print("rel_err max= ", np.max(rel_err))
    print("rel_err norm= ", rel_norm)

    print("abs threshold = ", abs_threshold)
    print("rel threshold = ", rel_threshold)

    if abs_threshold is not None:
        assert np.linalg.norm(dcoeff) <= abs_threshold
    if rel_threshold is not None:
        assert rel_norm <= rel_threshold


def test_apply_gaussian_noise():
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["trainable_parameters"] = "BINARY"
    AlNi_initial_basis_conf = create_multispecies_basis_config(simple_potential_config)
    elements = ACEBBasisSet(AlNi_initial_basis_conf).elements_name

    trainable_parameters = fit_config.get("trainable_parameters", [])
    trainable_parameters_dict = expand_trainable_parameters(elements=elements,
                                                            trainable_parameters=trainable_parameters)
    old_params_dict = get_block_params_dict(AlNi_initial_basis_conf)

    AlNi_noisy_basis_conf = GeneralACEFit.apply_gaussian_noise(AlNi_initial_basis_conf, trainable_parameters_dict, 1.0,
                                                               0.)
    new_params_dict = get_block_params_dict(AlNi_noisy_basis_conf)

    assert not np.allclose(new_params_dict["Ni Al"]["radial"], old_params_dict["Ni Al"]["radial"])
    assert not np.allclose(new_params_dict["Ni Al"]["func"], old_params_dict["Ni Al"]["func"])
    assert not np.allclose(new_params_dict["Al Ni"]["radial"], old_params_dict["Al Ni"]["radial"])
    assert not np.allclose(new_params_dict["Al Ni"]["func"], old_params_dict["Al Ni"]["func"])

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])
    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])


def test_randomize_func_coeffs():
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["trainable_parameters"] = "BINARY"
    AlNi_initial_basis_conf = create_multispecies_basis_config(simple_potential_config)
    elements = ACEBBasisSet(AlNi_initial_basis_conf).elements_name

    trainable_parameters = fit_config.get("trainable_parameters", [])
    trainable_parameters_dict = expand_trainable_parameters(elements=elements,
                                                            trainable_parameters=trainable_parameters)
    old_params_dict = get_block_params_dict(AlNi_initial_basis_conf)

    AlNi_noisy_basis_conf = GeneralACEFit.randomize_func_coeffs(AlNi_initial_basis_conf, trainable_parameters_dict, 1.0)
    new_params_dict = get_block_params_dict(AlNi_noisy_basis_conf)

    assert np.allclose(new_params_dict["Ni Al"]["radial"], old_params_dict["Ni Al"]["radial"])
    assert not np.allclose(new_params_dict["Ni Al"]["func"], old_params_dict["Ni Al"]["func"])
    assert np.allclose(new_params_dict["Al Ni"]["radial"], old_params_dict["Al Ni"]["radial"])
    assert not np.allclose(new_params_dict["Al Ni"]["func"], old_params_dict["Al Ni"]["func"])

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])
    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])


def test_GeneralACEFit_pyace_fit():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}
    fit_config["maxiter"] = 0

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_Al_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()
    bbasisconfig.save("test_GeneralACEFit_pyace_fit.yaml")
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", all_coeffs)
    expected_all_coeffs = [
        # Al
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  #
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # AlNi
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  #
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Ni
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # NiAl
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(all_coeffs, expected_all_coeffs)


def test_GeneralACEFit_pyace_ladder_cycle_noise_fit_trainable_parameters():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    # fit_config["options"] = {"gtol": 1e6}
    fit_config["maxiter"] = 1
    fit_config["ladder_step"] = 8
    fit_config["fit_cycles"] = 2
    fit_config["noise_absolute_sigma"] = 1e-3
    fit_config["trainable_parameters"] = "Al"

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_AlNi_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()
    bbasisconfig.save("test_GeneralACEFit_pyace_fit.yaml")
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", list(all_coeffs))
    expected_all_coeffs = [1.0004287245996735, -0.0013925644732522332, 1.0014697913242803, 0.000300868222467599,
                           -0.000343492321824346, 0.9966664686692588, 0.00025095771235912975, 1.0009640893744245,
                           -0.007576147090710318, -0.02534333503174486, -0.006760044367975883, -0.0006985133695235871,
                           -0.021385602564057925, -0.002430240140991591, -0.06836254431991892, 0.0010978079438683837,
                           1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert np.allclose(all_coeffs, expected_all_coeffs)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_fit():
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}
    fit_config["maxiter"] = 0

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_Al_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()
    bbasisconfig.save("test_GeneralACEFit_tensorpot_fit.yaml")

    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", all_coeffs)
    expected_all_coeffs = [
        # Al
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  #
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # AlNi
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  #
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Ni
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # NiAl
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(all_coeffs, expected_all_coeffs)
    bbasisconfig = BBasisConfiguration("test_GeneralACEFit_tensorpot_fit.yaml")


def get_block_params_dict(basisconf):
    params_dict = {}
    for block_ind, block in enumerate(basisconf.funcspecs_blocks):
        all_coefs = block.get_all_coeffs()
        ncrad = np.prod(np.shape(block.radcoefficients))
        crad_coeffs = all_coefs[:ncrad]
        func_coefs = all_coefs[ncrad:]
        params_dict[block.block_name] = {"radial": crad_coeffs, "func": func_coefs}
        print("block[{}] = {}".format(block.block_name, params_dict[block.block_name]))
    return params_dict


def test_GeneralACEFit_pyace_Al_fit_func():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["maxiter"] = 5

    fit_config["trainable_parameters"] = "func"

    initial_basis_conf = create_multispecies_basis_config(simple_potential_config)

    old_params_dict = get_block_params_dict(initial_basis_conf)

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_Al_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    opt_bbasisconfig = fitace.fit()

    new_params_dict = get_block_params_dict(opt_bbasisconfig)

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert not np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])

    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])


def test_GeneralACEFit_pyace_AlNi_fit_trainable_binary():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["maxiter"] = 3

    fit_config["trainable_parameters"] = "BINARY"

    initial_basis_conf = create_multispecies_basis_config(simple_potential_config)

    old_params_dict = get_block_params_dict(initial_basis_conf)

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_AlNi_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    opt_bbasisconfig = fitace.fit()

    new_params_dict = get_block_params_dict(opt_bbasisconfig)

    assert not np.allclose(new_params_dict["Ni Al"]["radial"], old_params_dict["Ni Al"]["radial"])
    assert not np.allclose(new_params_dict["Ni Al"]["func"], old_params_dict["Ni Al"]["func"])
    assert not np.allclose(new_params_dict["Al Ni"]["radial"], old_params_dict["Al Ni"]["radial"])
    assert not np.allclose(new_params_dict["Al Ni"]["func"], old_params_dict["Al Ni"]["func"])

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])
    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])


def test_GeneralACEFit_pyace_AlNi_fit_trainable_AlNi_func():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["maxiter"] = 3

    fit_config["trainable_parameters"] = {"AlNi": "func"}

    initial_basis_conf = create_multispecies_basis_config(simple_potential_config)

    old_params_dict = get_block_params_dict(initial_basis_conf)

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_AlNi_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    opt_bbasisconfig = fitace.fit()

    new_params_dict = get_block_params_dict(opt_bbasisconfig)

    assert np.allclose(new_params_dict["Ni Al"]["radial"], old_params_dict["Ni Al"]["radial"])
    assert np.allclose(new_params_dict["Ni Al"]["func"], old_params_dict["Ni Al"]["func"])
    assert np.allclose(new_params_dict["Al Ni"]["radial"], old_params_dict["Al Ni"]["radial"])
    assert not np.allclose(new_params_dict["Al Ni"]["func"], old_params_dict["Al Ni"]["func"])

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])
    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_AlNi_fit_trainable_binary():
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["maxiter"] = 3

    fit_config["trainable_parameters"] = "BINARY"

    initial_basis_conf = create_multispecies_basis_config(simple_potential_config)

    old_params_dict = get_block_params_dict(initial_basis_conf)

    fitace = GeneralACEFit(potential_config=simple_potential_config,
                           data_config=test_AlNi_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    opt_bbasisconfig = fitace.fit()

    new_params_dict = get_block_params_dict(opt_bbasisconfig)

    assert not np.allclose(new_params_dict["Ni Al"]["radial"], old_params_dict["Ni Al"]["radial"])
    assert not np.allclose(new_params_dict["Ni Al"]["func"], old_params_dict["Ni Al"]["func"])
    assert not np.allclose(new_params_dict["Al Ni"]["radial"], old_params_dict["Al Ni"]["radial"])
    assert not np.allclose(new_params_dict["Al Ni"]["func"], old_params_dict["Al Ni"]["func"])

    assert np.allclose(new_params_dict["Al"]["radial"], old_params_dict["Al"]["radial"])
    assert np.allclose(new_params_dict["Al"]["func"], old_params_dict["Al"]["func"])
    assert np.allclose(new_params_dict["Ni"]["radial"], old_params_dict["Ni"]["radial"])
    assert np.allclose(new_params_dict["Ni"]["func"], old_params_dict["Ni"]["func"])
    opt_bbasisconfig.save("test_opt_bbasisconfig.yaml")
    opt_bbasisconfig = BBasisConfiguration("test_opt_bbasisconfig.yaml")

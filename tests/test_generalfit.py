import glob
import os

import numpy as np
import pytest
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='-1'
from pyace import *
from pyace.generalfit import GeneralACEFit, active_import
from pyace.basisextension import construct_bbasisconfiguration
import pandas as pd

TENSORPOTENTIAL_IMPORTED = True
# TENSORPOTENTIAL_IMPORTED = False

test_fit_config_L1_L2 = {
    'optimizer': 'L-BFGS-B',  # Nelder-Mead #BFGS
    'maxiter': 5,

    'loss': {
        'kappa': 0.5,
        'L1_coeffs': 5e-2,
        'L2_coeffs': 5e-2,
        'w1_coeffs': 1,
        'w2_coeffs': 1,
        'w0_rad': 0,
        'w1_rad': 0,
        'w2_rad': 0,
    },
}

test_data_config = {'filename': 'tests/representative_df.pckl.gzip'}

test_fit_config_w_rad = {
    'optimizer': 'L-BFGS-B',  # 'L-BFGS-B',  # Nelder-Mead #BFGS
    'maxiter': 5,

    'loss': {
        'kappa': 0.5,

        'L1_coeffs': 0,
        'L2_coeffs': 0,

        'w1_coeffs': 1,
        'w2_coeffs': 1,

        'w0_rad': 1e-1,
        'w1_rad': 2e-1,
        'w2_rad': 3e-1,
    },
}

fitcyles_potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1, 1, 0.5],
    "npot": "FinnisSinclairShiftedScaled",
    "NameOfCutoffFunction": "cos",
    # "rankmax": 1,
    "nradmax_by_orders": [2],
    "lmax_by_orders": [0],
    "ndensity": 2,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
}

func_coefs_random_potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1, 1, 0.5],
    "npot": "FinnisSinclairShiftedScaled",
    "NameOfCutoffFunction": "cos",
    "rankmax": 1,
    # "nradmax_by_orders": [2],
    # "lmax_by_orders": [0],
    "nradmax": [2],
    "lmax": [0],
    "ndensity": 2,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
    "func_coefs_init": "random"
}

test_potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1, 1, 0.5],
    "npot": "FinnisSinclairShiftedScaled",
    "NameOfCutoffFunction": "cos",
    # "rankmax": 2,
    "nradmax_by_orders": [4, 3],
    "lmax_by_orders": [0, 1],
    "ndensity": 2,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",

    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
}

all_coeffs_ref_L1_L2 = np.array(
    [1.0018251007755958, -4.1253112073243596e-05, 0.00014407142339193204, 0.0009068770358774021, 0.9992726100949606,
     -4.13259661999816e-05, -0.00017630705959434005, -0.0003830962785025632, -0.0005154906879633648, 0.9999695212061385,
     -0.00013976779272491084, -0.0003195121591480847, -0.00011907907090764517, 0.9999978653303362,
     -1.7643155458789115e-05, -5.408171499482168e-05, -0.00136185568345254, -8.202601657756907e-05, 0.9996260764520081,
     -0.0008506335381837143, -0.0003682686247714983, -4.007865025889646e-06, 0.9999512806806697, -0.0001643737892630971,
     -0.007446305527441, -0.006626531801662597, 0.0003178422095423563, -9.480242468406922e-07, -0.0002787050358002696,
     -0.0009355121130896069, -0.0034691168309473113, -0.003445195988518582, -0.0274420217071527, -0.03143593665206571,
     -0.02168378024001206, -0.0024502245681668603, 0.010731828287725245, 0.0045285190124818185, -0.007815100758348204,
     -0.0037161436343309375, 0.021330957119601306, 0.006383302099100587, -0.021505727564229957, -0.009390823019485213,
     0.003355190617433722, 0.0017035344491587064, -0.0016477972002598286, -0.0008746819281863965, 0.009177192742107034,
     0.004581127854813982, -0.004855538914570914, -0.0025661430034491554, 0.02402951109823248, 0.011559903509051287,
     -0.01403089611519704, -0.00722872163379242])


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


def test_GeneralACEFit_pyace_fit_options():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}

    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]

    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", block.radcoefficients)

    compare_coefficients(all_coeffs,
                         [0, 0, 0, 0],
                         abs_threshold=2e-5,
                         rel_threshold=None)


def test_GeneralACEFit_pyace_fit_func_coefs_init_random():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}

    fitace = GeneralACEFit(potential_config=func_coefs_random_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()
    block = bbasisconfig.funcspecs_blocks[0]

    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", block.radcoefficients)

    compare_coefficients(all_coeffs,
                         [4.967141530112327e-05, -1.3826430117118467e-05, 6.476885381006925e-05,
                          0.00015230298564080253],
                         abs_threshold=2e-6,
                         rel_threshold=None)


def test_GeneralACEFit_pyace_core_rep():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}
    potential_config = fitcyles_potential_config.copy()
    potential_config["core-repulsion"] = [500, 10]
    potential_config["rho_core_cut"] = 50
    potential_config["drho_core_cut"] = 20

    fitace = GeneralACEFit(potential_config=potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    block = bbasisconfig.funcspecs_blocks[0]
    print("block.core_rep_parameters=", block.core_rep_parameters)
    print("block.rho_core_cut=", block.rho_cut)
    print("block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20

    fitace.save_optimized_potential("test_core_rep.yaml")
    bbasisconfig = BBasisConfiguration("test_core_rep.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("loaded block.core_rep_parameters=", block.core_rep_parameters)
    print("loaded block.rho_core_cut=", block.rho_cut)
    print("loaded block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20

    fitace2 = GeneralACEFit(potential_config="test_core_rep.yaml",
                            data_config=test_data_config,
                            fit_config=fit_config,
                            backend_config=backend_config, seed=42)
    bbasisconfig2 = fitace2.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block2 = bbasisconfig2.funcspecs_blocks[0]
    print("block2.core_rep_parameters=", block2.core_rep_parameters)
    print("block2.rho_core_cut=", block2.rho_cut)
    print("block2.drho_core_cut=", block2.drho_cut)
    assert block2.rho_cut == 50
    assert block2.drho_cut == 20


def test_GeneralACEFit_pyace_continue_with_core_rep():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}
    potential_config = fitcyles_potential_config.copy()
    bbasis_config = construct_bbasisconfiguration(potential_config)

    bbasis_config.save("test_no_core_rep.yaml")

    potential_config["initial_potential"] = "test_no_core_rep.yaml"
    potential_config["nradmax_by_orders"] = [3]
    potential_config["core-repulsion"] = [500, 10]
    potential_config["rho_core_cut"] = 50
    potential_config["drho_core_cut"] = 20

    fitace = GeneralACEFit(potential_config=potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    block = bbasisconfig.funcspecs_blocks[0]
    print("block.core_rep_parameters=", block.core_rep_parameters)
    print("block.rho_core_cut=", block.rho_cut)
    print("block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20

    fitace.save_optimized_potential("test_core_rep.yaml")
    bbasisconfig = BBasisConfiguration("test_core_rep.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("loaded block.core_rep_parameters=", block.core_rep_parameters)
    print("loaded block.rho_core_cut=", block.rho_cut)
    print("loaded block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20

    fitace2 = GeneralACEFit(potential_config="test_core_rep.yaml",
                            data_config=test_data_config,
                            fit_config=fit_config,
                            backend_config=backend_config, seed=42)
    bbasisconfig2 = fitace2.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block2 = bbasisconfig2.funcspecs_blocks[0]
    print("block2.core_rep_parameters=", block2.core_rep_parameters)
    print("block2.rho_core_cut=", block2.rho_cut)
    print("block2.drho_core_cut=", block2.drho_cut)
    assert block2.rho_cut == 50
    assert block2.drho_cut == 20


def test_GeneralACEFit_save_fitting_data_info():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    expected_filename = "fitting_data_info.pckl.gzip"
    if os.path.isfile(expected_filename):
        os.remove(expected_filename)
    assert not os.path.isfile(expected_filename)

    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)

    fitace.fit()
    assert os.path.isfile(expected_filename)
    fit_df = pd.read_pickle(expected_filename, compression="gzip")
    columns = sorted(fit_df.columns)

    assert "ase_atoms" in columns
    assert "energy" in columns
    assert "energy_corrected" in columns
    assert "energy_corrected_per_atom" in columns
    assert "forces" in columns

    os.remove(expected_filename)


def test_GeneralACEFit_fit_cycles_relative_noise():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    fit_config_cycles["fit_cycles"] = 2
    fit_config_cycles["noise_relative_sigma"] = 0.1
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("crad=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    # assert int(bbasisconfig.metadata["_fit_cycles"]) == 2
    compare_coefficients(all_coeffs,
                         [-0.06745176623506423, -1.1261920199354432, 1.5536919177018216, 0.2528407212067861],
                         abs_threshold=2e-5,
                         rel_threshold=None)


def test_GeneralACEFit_fit_cycles_absolute_noise():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    fit_config_cycles["fit_cycles"] = 2
    fit_config_cycles["noise_absolute_sigma"] = 0.1
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("crad=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    # assert int(bbasisconfig.metadata["_fit_cycles"]) == 2
    compare_coefficients(all_coeffs,
                         [-0.03742081810260262, -0.9032276344882844, 1.2840935490690006, 0.34987568968784133],
                         abs_threshold=2e-5,
                         rel_threshold=None)


def test_GeneralACEFit_dataset_with_weights_column():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    # fit_config_cycles["fit_cycles"] = 2
    # fit_config_cycles["noise_absolute_sigma"] = 0.1
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config={"filename": "tests/df_weights.pckl.gzip"},
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("crad=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    compare_coefficients(all_coeffs,
                         [-0.27669969651794296, -0.47842325717047246, 2.3316516904371483, 1.9139344049851845],
                         abs_threshold=5e-6,
                         rel_threshold=None)


def test_GeneralACEFit_dataset_with_weights_column_force_weights_recalc():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    data_config = {"filename": "tests/df_weights.pckl.gzip", "ignore_weights": True}
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=data_config,
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("crad=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    compare_coefficients(all_coeffs,
                         [-0.011837195208307498, -0.2551255903213622, 0.11393075424512798, 0.02827651434980819],
                         abs_threshold=1e-6,
                         rel_threshold=None)


def test_GeneralACEFit_pyace_L1_L2():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }

    fitace = GeneralACEFit(potential_config=test_potential_config, data_config=test_data_config,
                           fit_config=test_fit_config_L1_L2,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("test_GeneralACEFit_pyace_L1_L2:crad=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    compare_coefficients(all_coeffs, all_coeffs_ref_L1_L2, abs_threshold=2e-6, rel_threshold=1e-2)


all_coeffs_rad_smoothness = np.array(
    [1.0019169656619304, -4.647047170258803e-05, 0.00014034280495604616, 0.0009583564389256882, 0.9989287434336567,
     -6.200445621404138e-05, -0.00027220425985198396, -0.0005788038038901483, -0.0008259408296999081,
     0.9999268660800964, -0.0002535510431383997, -0.0004927811018673341, -0.00036058978283052634, 0.999959387379982,
     -0.0001117104695747868, -0.00018144431737907173, -0.0018175904349284641, -0.00012417967577225576,
     0.9994617646940869, -0.0011353382325416962, -0.000656176579733758, -3.582605059275514e-05, 0.9998359760995102,
     -0.0003347322829450886, -0.007928538500798294, -0.006841249595933891, 0.00029786760697641916,
     -5.7460132367418244e-05, -0.00035100203428952917, -0.0010001480494823871, -0.003778707408784413,
     -0.0035848036562923526, -0.02930579902132576, -0.031227653857346792, -0.022062769863560497, -0.003182125438847701,
     0.011189831983510106, 0.004847572373215616, -0.008099217345682697, -0.003894375421558507, 0.022058726015781967,
     0.007292296393542844, -0.022242843640854285, -0.00992428116873616, 0.0035894323335002257, 0.0018610830689793426,
     -0.0017960799591859307, -0.0009839527216469414, 0.009673756300408058, 0.004895115958719443, -0.00514442708024614,
     -0.002753293747719123, 0.02503918938840678, 0.012191732176742989, -0.014695291955885448, -0.007635408163554351]
)


def test_GeneralACEFit_pyace_loss_rad():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }

    fitace = GeneralACEFit(potential_config=test_potential_config,
                           fit_config=test_fit_config_w_rad,
                           data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_pyace_pot2.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("block.radcoefficients=", block.radcoefficients)
    all_coeffs = fitace.target_bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_rad_smoothness, abs_threshold=2e-6, rel_threshold=1e-2)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_core_rep():
    backend_config = {
        'evaluator': 'tensorpot',
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}
    potential_config = fitcyles_potential_config.copy()
    potential_config["core-repulsion"] = [500, 10]
    potential_config["rho_core_cut"] = 50
    potential_config["drho_core_cut"] = 20

    fitace = GeneralACEFit(potential_config=potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("block.core_rep_parameters=", block.core_rep_parameters)
    print("block.rho_core_cut=", block.rho_cut)
    print("block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20

    fitace.save_optimized_potential("test_core_rep.yaml")
    bbasisconfig = BBasisConfiguration("test_core_rep.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("loaded block.core_rep_parameters=", block.core_rep_parameters)
    print("loaded block.rho_core_cut=", block.rho_cut)
    print("loaded block.drho_core_cut=", block.drho_cut)
    assert block.rho_cut == 50
    assert block.drho_cut == 20


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_fit_options():
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }
    fit_config = test_fit_config_L1_L2.copy()
    fit_config["options"] = {"gtol": 1e6}

    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config=test_data_config,
                           fit_config=fit_config,
                           backend_config=backend_config, seed=42)
    bbasisconfig = fitace.fit()

    # bbasisconfig = fitace.target_bbasisconfig
    # bbasisconfig.save("test_pyace_pot.yaml")
    block = bbasisconfig.funcspecs_blocks[0]

    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    print("all_coeffs=", all_coeffs)
    # assert int(bbasisconfig.metadata["_fit_cycles"]) == 2
    compare_coefficients(all_coeffs,
                         [0.0, 0.0, 0.0, 0.0],
                         abs_threshold=2e-5,
                         rel_threshold=None)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_L1_L2():
    if not TENSORPOTENTIAL_IMPORTED:
        pytest.fail()
    # Fit config
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }

    fitace = GeneralACEFit(potential_config=test_potential_config, data_config=test_data_config,
                           fit_config=test_fit_config_L1_L2,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_tf_pot.yaml")
    all_coeffs = fitace.target_bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_ref_L1_L2, abs_threshold=0.005, rel_threshold=None)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_loss_rad():
    if not TENSORPOTENTIAL_IMPORTED:
        pytest.fail()
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }

    fitace = GeneralACEFit(potential_config=test_potential_config,
                           fit_config=test_fit_config_w_rad,
                           data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_tp_pot2.yaml")
    block = bbasisconfig.funcspecs_blocks[0]
    print("block.radcoefficients=", block.radcoefficients)
    all_coeffs = np.array(bbasisconfig.get_all_coeffs())
    compare_coefficients(all_coeffs, all_coeffs_rad_smoothness, abs_threshold=0.005, rel_threshold=None)


all_coeffs_bbasis_yaml = [1.0156768899086037, 7.0045191654986985, 0.18102546004421222, 7.530623283167457, 3.0, 9.0, 4.0,
                          10.0, 5.0, 11.0, 6.0, 12.0, 0.9999957696902209, 0.9999985307940878, 2.000386331451278,
                          3.0157732868376046, 3.622085508003881]


def test_GeneralACEFit_pyace_BBasis_yaml():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif_2.yaml")
    fitace = GeneralACEFit(potential_config=bbasis, fit_config=test_fit_config_L1_L2, data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_pyace_pot3.yaml")
    all_coeffs = bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_bbasis_yaml, abs_threshold=7e-6, rel_threshold=2e-5)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_BBasis_yaml():
    if not TENSORPOTENTIAL_IMPORTED:
        pytest.fail()
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
        'batch_size': 2
    }
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif_2.yaml")
    fitace = GeneralACEFit(potential_config=bbasis, fit_config=test_fit_config_L1_L2, data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    bbasisconfig.save("test_tensorpot_pot3.yaml")
    all_coeffs = bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_bbasis_yaml, abs_threshold=2e-6, rel_threshold=6e-6)


test_fit_config_query = {
    'optimizer': 'L-BFGS-B',  # Nelder-Mead #BFGS
    'maxiter': 5,

    'loss': {
        'kappa': 0.5,

        'L1_coeffs': 0,
        'L2_coeffs': 0,

        'w1_coeffs': 1,
        'w2_coeffs': 1,

        'w0_rad': 0,
        'w1_rad': 0,
        'w2_rad': 0,
    },
    'weighting': {
        'type': 'EnergyBasedWeightingPolicy',
        'nfit': 10,
        'cutoff': 10,
        'DElow': 0.1,
        'DEup': 10.0,
        'DE': 1.0,
        'wlow': 0.99,
        'seed': 42
    }
}

test_data_config_query = {'config': {
    "element": "Al",
    "calculator": 'FHI-aims/PBE/tight',
    'cutoff': 10,
    'seed': 42,
    "datapath": "tests",
    'overwrite_original_file': False,
    'save_fit_df': False,
}, 'datapath': 'tests'
}

all_coeffs_ref_query = [1.0253271106342696, 7.0057278638726315, 0.2828847846460406, 7.6512483395083075, 3.0, 9.0, 4.0,
                        10.0, 5.0, 11.0, 6.0, 12.0, 0.9999867378951081, 0.999996530192606, 2.0006053956850787,
                        3.0218397754095827, 3.67733950790798]

test_fit_config_weighting = {
    'optimizer': 'L-BFGS-B',  # Nelder-Mead #BFGS
    'maxiter': 5,

    'loss': {
        'kappa': 0.5,

        'L1_coeffs': 0,
        'L2_coeffs': 0,

        'w1_coeffs': 1,
        'w2_coeffs': 1,

        'w0_rad': 0,
        'w1_rad': 0,
        'w2_rad': 0,
    },

    'weighting': {
        'type': 'EnergyBasedWeightingPolicy',
        'nfit': 100,
        'cutoff': 10,
        'DElow': 1.0,
        'DEup': 10.0,
        'DE': 1.0,
        'wlow': 0.75,
        'seed': 42
    }
}

all_coeffs_ref_weighting = [1.0209968271570045, 7.005161495282142, 0.24169719971414913, 7.6050327797763755, 3.0, 9.0,
                            4.0, 10.0, 5.0, 11.0, 6.0, 12.0, 0.999989824116624, 0.9999971572487281, 2.000534764767421,
                            3.0190596629161126, 3.656235168348817]


def test_GeneralACEFit_pyace_weighting():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif_2.yaml")
    fitace = GeneralACEFit(potential_config=bbasis,
                           fit_config=test_fit_config_weighting,
                           data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    all_coeffs = bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_ref_weighting, abs_threshold=3e-6, rel_threshold=6e-6)


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_weighting():
    if not TENSORPOTENTIAL_IMPORTED:
        pytest.fail()

    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif_2.yaml")
    fitace = GeneralACEFit(potential_config=bbasis,
                           fit_config=test_fit_config_weighting,
                           data_config=test_data_config,
                           backend_config=backend_config)
    fitace.fit()

    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    all_coeffs = bbasisconfig.get_all_coeffs()
    compare_coefficients(all_coeffs, all_coeffs_ref_weighting, abs_threshold=2e-6, rel_threshold=5e-6)


ladder_potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",
    # "rankmax": 2,
    "nradmax_by_orders": [2, 1],
    "lmax_by_orders": [0, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
}

ladder_fit_config = {
    'optimizer': 'L-BFGS-B',  # Nelder-Mead #BFGS
    'maxiter': 5,

    'loss': {
        'kappa': 0.5,
        'L1_coeffs': 5e-2,
        'L2_coeffs': 5e-2,
        'w1_coeffs': 1,
        'w2_coeffs': 1,
        'w0_rad': 0,
        'w1_rad': 0,
        'w2_rad': 0,
    },

    'ladder_step': 0.9,
    'fit_cycles': 2
}


def test_GeneralACEFit_pyace_ladderscheme():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
    }

    # remove and check that there are no interim potentials
    for i in range(3):
        fname = "interim_potential_ladder_step_{i}.yaml".format(i=i)
        if os.path.isfile(fname):
            os.remove(fname)
        assert not os.path.isfile(fname)

    fitace = GeneralACEFit(potential_config=ladder_potential_config,
                           data_config=test_data_config,
                           fit_config=ladder_fit_config,
                           backend_config=backend_config)
    fitace.fit()
    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    all_coeffs = bbasisconfig.get_all_coeffs()
    print("all_coeffs=", all_coeffs)
    compare_coefficients(all_coeffs,
                         [0.9998508552498461, -0.0038456640969905514, 0.9997419540364952, 0.005679640585532174,
                          -0.6537803633506181, 3.1399397329259173, -0.0009856261099147837, -0.01723476703936692],
                         abs_threshold=6e-3, rel_threshold=None)

    # check that the interim potentials after each ladder step is saved
    for i in range(3):
        fname = "interim_potential_ladder_step_{i}.yaml".format(i=i)
        assert os.path.isfile(fname)


def test_GeneralACEFit_pyace_ladderscheme_initial_potential():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
    }

    ladder_potential_config_init = ladder_potential_config.copy()
    ladder_potential_config_init["initial_potential"] = "tests/Al-r1l0.yaml"

    fitace = GeneralACEFit(potential_config=ladder_potential_config_init,
                           data_config=test_data_config,
                           fit_config=ladder_fit_config,
                           backend_config=backend_config)
    fitace.fit()
    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    all_coeffs = bbasisconfig.get_all_coeffs()
    print("all_coeffs=", all_coeffs)
    compare_coefficients(all_coeffs,
                         [0.9998687311259148, -0.002074779584948978, 0.9997998833532301, 0.0035680103476770156,
                          -0.8729670926048945, 2.4480651973606204, -4.1657079213402396e-05, -0.012264438891335432],
                         abs_threshold=6e-3,
                         rel_threshold=None,

                         )


@pytest.mark.tensorpot
def test_GeneralACEFit_tensorpot_ladderscheme_initial_potential():
    backend_config = {
        'evaluator': 'tensorpot',  # pyace, tensorpot
    }

    ladder_potential_config_init = ladder_potential_config.copy()
    ladder_potential_config_init["initial_potential"] = "tests/Al-r1l0.yaml"

    fitace = GeneralACEFit(potential_config=ladder_potential_config_init,
                           data_config=test_data_config,
                           fit_config=ladder_fit_config,
                           backend_config=backend_config)
    fitace.fit()
    bbasisconfig = fitace.target_bbasisconfig
    print(bbasisconfig)
    all_coeffs = bbasisconfig.get_all_coeffs()
    print("all_coeffs=", all_coeffs)
    compare_coefficients(all_coeffs,
                         [0.9998687311259148, -0.002074779584948978, 0.9997998833532301, 0.0035680103476770156,
                          -0.8729670926048945, 2.4480651973606204, -4.1657079213402396e-05, -0.012264438891335432],
                         abs_threshold=6e-3,
                         rel_threshold=None,

                         )


def test_GeneralACEFit_save_interim_potentials_callback():
    backend_config = {
        'evaluator': 'pyace',  # pyace, tensorpot
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()
    for interim_pots in glob.glob("interim_potential*.yaml"):
        os.remove(interim_pots)
    interim_pots_before_fit = glob.glob("interim_potential*.yaml")
    assert len(interim_pots_before_fit) == 0
    fit_config_cycles["fit_cycles"] = 2
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config={"filename": "tests/df_weights.pckl.gzip"},
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42)
    fitace.fit()
    interim_pots_after_fit = glob.glob("interim_potential*.yaml")
    print("interim_pots_after_fit=", interim_pots_after_fit)
    assert len(interim_pots_after_fit) > 0
    assert 'interim_potential_1.yaml' in interim_pots_after_fit
    assert 'interim_potential_best_cycle.yaml' in interim_pots_after_fit


def test_GeneralACEFit_callback():
    backend_config = {
        'evaluator': 'pyace',
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()

    history = []

    def test_callback(basis_config: BBasisConfiguration, current_fit_iteration: int, current_fit_cycle: int,
                      current_ladder_step: int):
        history.append((current_fit_iteration, current_fit_cycle, current_ladder_step))

    fit_config_cycles["fit_cycles"] = 2
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config={"filename": "tests/df_weights.pckl.gzip"},
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42, callbacks=[test_callback])
    fitace.fit()

    print("History: ", history)
    assert history == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
                       (3, 1, 0), (4, 1, 0)]


def test_GeneralACEFit_external_callback():
    backend_config = {
        'evaluator': 'pyace',
        "parallel_mode": "serial",
    }
    fit_config_cycles = test_fit_config_L1_L2.copy()

    history = []

    fit_config_cycles["fit_cycles"] = 2
    print("cwd: ", os.getcwd())
    cb_name = "pyace.generalfit.save_interim_potential_callback"
    cb = active_import(cb_name)
    print("cb=", cb)
    fitace = GeneralACEFit(potential_config=fitcyles_potential_config,
                           data_config={"filename": "tests/df_weights.pckl.gzip"},
                           fit_config=fit_config_cycles,
                           backend_config=backend_config, seed=42, callbacks=[cb_name])
    fitace.fit()

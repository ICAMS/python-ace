import numpy as np
import pytest

from pyace import *
from pyace.basisextension import construct_bbasisconfiguration
from pyace.multispecies_basisextension import single_to_multispecies_converter, expand_trainable_parameters, \
    compute_bbasisset_train_mask, extend_multispecies_basis


def test_create_multispecies_basis_config_1():
    potential_config = {
        'deltaSplineBins': 0.001,
        'elements': ['Al', 'Ni', 'Cu'],

        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclairShiftedScaled',
                               'rho_core_cut': 200000}},

        'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                          'core-repulsion': [10000.0, 5.0],
                          'dcut': 0.01,
                          'radbase': 'ChebPow',
                          'radparameters': [2.0],
                          'lmax': 2,
                          'rcut': 3.9}
                  },

        'functions': {
            'UNARY': {
                'nradmax_by_orders': [5, 2],
                'lmax_by_orders': [0, 0]
            },

            'Al': {
                'nradmax_by_orders': [5, 2],
                'lmax_by_orders': [0, 0]
            },

            'BINARY': {
                'nradmax_by_orders': [5, 2, 2],
                'lmax_by_orders': [0, 2, 2],
            },

            ('Ni', 'Al'): {
                'nradmax_by_orders': [4, 2, 1],
                'lmax_by_orders': [0, 2, 1],
            },

            'TERNARY': {
                'nradmax_by_orders': [3, 1, 1],
                'lmax_by_orders': [0, 1, 1],
            }
        }
    }

    new_basis_conf = create_multispecies_basis_config(potential_config)
    assert len(new_basis_conf.funcspecs_blocks) == 12
    block_names = [b.block_name for b in new_basis_conf.funcspecs_blocks]
    block_names_ref = ['Al',
                       'Cu',
                       'Ni',
                       'Al Cu',
                       'Al Ni',
                       'Cu Al',
                       'Cu Ni',
                       'Ni Al',
                       'Ni Cu',
                       'Al Cu Ni',
                       'Cu Al Ni',
                       'Ni Al Cu']

    assert block_names == block_names_ref

    num_block_funcs = [len(b.funcspecs) for b in new_basis_conf.funcspecs_blocks]
    print("num_block_funcs=", num_block_funcs)
    num_block_funcs_ref = [8, 8, 8, 160, 160, 160, 160, 33, 160, 12, 12, 12]
    assert num_block_funcs == num_block_funcs_ref


def test_create_multispecies_basis_config_2():
    potential_config = {
        'deltaSplineBins': 0.001,
        'elements': ['Al', 'H'],

        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclair',
                               'rho_core_cut': 200000},
                       'Al': {'drho_core_cut': 250,
                              'fs_parameters': [1, 1, 1, 0.5],
                              'ndensity': 2,
                              'npot': 'FinnisSinclairShiftedScaled',
                              'rho_core_cut': 200000}

                       },

        'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                          'core-repulsion': [10000.0, 5.0],
                          'dcut': 0.01,
                          'radbase': 'ChebPow',
                          'radparameters': [2.0],
                          'rcut': 3.9},

                  ('Al', 'H'): {'NameOfCutoffFunction': 'cos',
                                'core-repulsion': [10000.0, 5.0],
                                'dcut': 0.01,
                                'radbase': 'ChebExpCos',
                                'radparameters': [2.0],
                                'rcut': 3.5},
                  },

        'functions': {
            'UNARY': {
                'nradmax_by_orders': [5, 2, 2],
                'lmax_by_orders': [0, 1, 1]
            },

            'BINARY': {
                'nradmax_by_orders': [5, 1, 1],
                'lmax_by_orders': [0, 1, 1],
            },
        }
    }

    new_basis_conf = create_multispecies_basis_config(potential_config)
    assert len(new_basis_conf.funcspecs_blocks) == 4
    block_names = [b.block_name for b in new_basis_conf.funcspecs_blocks]
    block_names_ref = ['Al',
                       'H',
                       'Al H',
                       'H Al']

    assert block_names == block_names_ref

    num_block_funcs = [len(b.funcspecs) for b in new_basis_conf.funcspecs_blocks]
    num_block_funcs_ref = [21, 21, 17, 17]
    assert num_block_funcs == num_block_funcs_ref

    assert [b.radbase for b in new_basis_conf.funcspecs_blocks] == ["ChebPow", "ChebPow", "ChebExpCos", "ChebExpCos"]


def test_create_10elements_binary_only():
    elements = ['Ag', 'Al', 'Co', 'Cu', 'Fe', 'Mg', 'Nb', 'Ni', 'Ti', 'V']
    potential_config = {
        'deltaSplineBins': 0.001,
        'elements': elements,
        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclair',
                               'rho_core_cut': 200000},
                       },

        'bonds': {'UNARY': {'NameOfCutoffFunction': 'cos',
                            'core-repulsion': [10000.0, 5.0],
                            'dcut': 0.01,
                            'radbase': 'ChebPow',
                            'radparameters': [2.0],
                            'rcut': 3.9},

                  'BINARY': {'NameOfCutoffFunction': 'cos',
                             'core-repulsion': [10000.0, 5.0],
                             'dcut': 0.01,
                             'radbase': 'ChebExpCos',
                             'radparameters': [2.0],
                             'rcut': 3.5},
                  },

        'functions': {
            'UNARY': {
                'nradmax_by_orders': [5, 1, 1],
                'lmax_by_orders': [0, 0, 0]
            },
            'BINARY': {
                'nradmax_by_orders': [5, 1, 1],
                'lmax_by_orders': [0, 0, 0],
            },
        }
    }

    new_basis_conf = create_multispecies_basis_config(potential_config)
    block_names = [b.block_name for b in new_basis_conf.funcspecs_blocks]
    print('num of blocks=', len(block_names))
    print('block_names=', block_names)

    assert len(block_names) == 100

    block_names_ref = ['Ag', 'Al', 'Co', 'Cu', 'Fe', 'Mg', 'Nb', 'Ni', 'Ti', 'V', 'Ag Al', 'Ag Co', 'Ag Cu', 'Ag Fe',
                       'Ag Mg', 'Ag Nb', 'Ag Ni', 'Ag Ti', 'Ag V', 'Al Ag', 'Al Co', 'Al Cu', 'Al Fe', 'Al Mg', 'Al Nb',
                       'Al Ni', 'Al Ti', 'Al V', 'Co Ag', 'Co Al', 'Co Cu', 'Co Fe', 'Co Mg', 'Co Nb', 'Co Ni', 'Co Ti',
                       'Co V', 'Cu Ag', 'Cu Al', 'Cu Co', 'Cu Fe', 'Cu Mg', 'Cu Nb', 'Cu Ni', 'Cu Ti', 'Cu V', 'Fe Ag',
                       'Fe Al', 'Fe Co', 'Fe Cu', 'Fe Mg', 'Fe Nb', 'Fe Ni', 'Fe Ti', 'Fe V', 'Mg Ag', 'Mg Al', 'Mg Co',
                       'Mg Cu', 'Mg Fe', 'Mg Nb', 'Mg Ni', 'Mg Ti', 'Mg V', 'Nb Ag', 'Nb Al', 'Nb Co', 'Nb Cu', 'Nb Fe',
                       'Nb Mg', 'Nb Ni', 'Nb Ti', 'Nb V', 'Ni Ag', 'Ni Al', 'Ni Co', 'Ni Cu', 'Ni Fe', 'Ni Mg', 'Ni Nb',
                       'Ni Ti', 'Ni V', 'Ti Ag', 'Ti Al', 'Ti Co', 'Ti Cu', 'Ti Fe', 'Ti Mg', 'Ti Nb', 'Ti Ni', 'Ti V',
                       'V Ag', 'V Al', 'V Co', 'V Cu', 'V Fe', 'V Mg', 'V Nb', 'V Ni', 'V Ti']

    assert block_names == block_names_ref

    num_block_funcs = [len(b.funcspecs) for b in new_basis_conf.funcspecs_blocks]
    print("num_block_funcs=", num_block_funcs)
    num_block_funcs_ref = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                           10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                           10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                           10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                           10, 10, 10, 10, 10]

    assert num_block_funcs == num_block_funcs_ref


def test_single_to_multispecies_converter():
    potential_config = {
        "deltaSplineBins": 0.001,
        "element": "Al",
        "fs_parameters": [1, 1, 1, 0.5],
        "npot": "FinnisSinclairShiftedScaled",
        "ndensity": 2,

        "NameOfCutoffFunction": "cos",

        "rankmax": 2,
        'nradmax_by_orders': [5, 2],
        'lmax_by_orders': [0, 1],
        #     "nradmax": [2],
        #     "lmax": [0],
        "rcut": 8.7,
        "dcut": 0.01,
        "radparameters": [5.25],
        "radbase": "ChebExpCos",

        'rho_core_cut': 200000,
        'drho_core_cut': 250,

        'core-repulsion': [10000.0, 5.0],
        "func_coefs_init": "random"
    }

    multispecies_potential_config = {
        'deltaSplineBins': 0.001,
        'elements': ['Al'],

        'embeddings':
            {'Al':
                 {'npot': 'FinnisSinclairShiftedScaled',
                  'fs_parameters': [1, 1, 1, 0.5],
                  'ndensity': 2,
                  'rho_core_cut': 200000,
                  'drho_core_cut': 250}
             },

        'bonds':
            {'Al': {'NameOfCutoffFunction': 'cos',
                    'core-repulsion': [10000.0, 5.0],
                    'dcut': 0.01,
                    'rcut': 8.7,
                    'radbase': 'ChebExpCos',
                    'radparameters': [5.25]}
             },

        'functions':
            {'Al':
                 {'nradmax_by_orders': [5, 2],
                  'lmax_by_orders': [0, 1],
                  'coefs_init': 'random'
                  }
             }

    }

    new_multi_species_potential_config = single_to_multispecies_converter(potential_config)

    assert new_multi_species_potential_config == multispecies_potential_config


@pytest.mark.parametrize("trainable_parameters, elements, expected_trainable_dict", [
    ([], ["Al"], {('Al',): ['func', 'radial']}),
    (["ALL"], ["Al"], {('Al',): ['func', 'radial']}),
    (["UNARY"], ["Al"], {('Al',): ['func', 'radial']}),
    (["UNARY"], ["Al", "Cu"], {('Al',): ['func', 'radial'], ('Cu',): ['func', 'radial']}),
    (["BINARY"], ["Al", "Cu"], {('Al', 'Cu'): ['func', 'radial'], ('Cu', 'Al'): ['func', 'radial']}),
    ([("Al", "Cu"), ("Cu", "Al")], ["Al", "Cu"], {('Al', 'Cu'): ['func', 'radial'], ('Cu', 'Al'): ['func', 'radial']}),
    ([("Al", "Cu", "Zn")], ["Al", "Zn", "Cu"], {('Al', 'Cu', 'Zn'): ['func']}),

    # with string in list
    (["AlCu", "CuAl"], ["Al", "Cu"], {('Al', 'Cu'): ['func', 'radial'], ('Cu', 'Al'): ['func', 'radial']}),
    (["AlCuZn"], ["Al", "Zn", "Cu"], {('Al', 'Cu', 'Zn'): ['func']}),

    # with string only
    ("ALL", ["Al"], {('Al',): ['func', 'radial']}),
    ("UNARY", ["Al"], {('Al',): ['func', 'radial']}),
    ("BINARY", ["Al", "Cu"], {('Al', 'Cu'): ['func', 'radial'], ('Cu', 'Al'): ['func', 'radial']}),
    ("AlCuZn", ["Al", "Zn", "Cu"], {('Al', 'Cu', 'Zn'): ['func']}),
])
def test_expand_trainable_parameters_as_list(trainable_parameters, elements, expected_trainable_dict):
    trainable_dict = expand_trainable_parameters(elements, trainable_parameters)
    print("elements=", elements)
    print("trainable_parameters=", trainable_parameters)
    print("trainable_dict=", trainable_dict)
    assert trainable_dict == expected_trainable_dict


@pytest.mark.parametrize("trainable_parameters, elements, expected_trainable_dict", [
    ({"UNARY": ["func"]}, ["Al"], {('Al',): ['func']}),
    ({"UNARY": "func"}, ["Al"], {('Al',): ['func']}),
    ({"UNARY": "func"}, ["Al", "Cu"], {('Al',): ['func'], ('Cu',): ['func']}),

    ({"UNARY": "all"}, ["Al"], {('Al',): ['func', 'radial']}),
    ({"UNARY": "all"}, ["Al", "Cu"], {('Al',): ['func', 'radial'], ('Cu',): ['func', 'radial']}),

    ({"UNARY": ["func", "radial"]},
     ["Al", "Cu"],
     {('Al',): ['func', 'radial'], ('Cu',): ['func', 'radial']}
     ),
    # 6
    ({"ALL": "func"}, ["Al"], {('Al',): ['func']}),
    # 7
    ({"ALL": "func"}, ["Al", "Cu"],
     {('Al',): ['func'], ('Cu',): ['func'], ('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),
    # 8
    ({"ALL": "func"}, ["Al", "Cu", "B"],
     {('Al',): ['func'], ('B',): ['func'], ('Cu',): ['func'], ('Al', 'B'): ['func'], ('Al', 'Cu'): ['func'],
      ('B', 'Al'): ['func'], ('B', 'Cu'): ['func'], ('Cu', 'Al'): ['func'], ('Cu', 'B'): ['func'],
      ('Al', 'B', 'Cu'): ['func'], ('B', 'Al', 'Cu'): ['func'], ('Cu', 'Al', 'B'): ['func']}
     ),

    # 9
    ({"BINARY": "func"}, ["Al"], {}),
    # 10
    ({"BINARY": "func"}, ["Al", "Cu"], {('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),
    # 11
    ({"BINARY": "func"}, ["Al", "Cu", "Zn"],
     {('Al', 'Cu'): ['func'], ('Al', 'Zn'): ['func'], ('Cu', 'Al'): ['func'], ('Cu', 'Zn'): ['func'],
      ('Zn', 'Al'): ['func'], ('Zn', 'Cu'): ['func']}
     ),

    ({"TERNARY": "func"}, ["Al"], {}),
    ({"TERNARY": "func"}, ["Al", "Cu"], {}),
    ({"TERNARY": "func"}, ["Al", "Cu", "Zn"],
     {('Al', 'Cu', 'Zn'): ['func'], ('Cu', 'Al', 'Zn'): ['func'], ('Zn', 'Al', 'Cu'): ['func']}),
    # 15
    ({("Al", "Cu"): "func", ("Cu", "Al"): ["func"]}, ["Al", "Cu"],
     {('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),
    # 16
    ({("Al", "Cu"): "func", ("Cu", "Al"): ["func"]}, ["Al", "Cu", "Zn"],
     {('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),
    # 16
    ({("Al", "Cu", "Zn"): "func"}, ["Al", "Cu", "Zn"], {('Al', 'Cu', 'Zn'): ['func']}),

    # as string
    # 18
    ({"AlCu": "func", "CuAl": "func"}, ["Al", "Cu"],
     {('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),

    # 19
    ({"AlCu": "func", "CuAl": ["func"]}, ["Al", "Cu", "Zn"],
     {('Al', 'Cu'): ['func'], ('Cu', 'Al'): ['func']}),

    # 20
    ({"AlCuZn": "func"}, ["Al", "Cu", "Zn"], {('Al', 'Cu', 'Zn'): ['func']}),

])
def test_expand_trainable_parameters_as_dict(trainable_parameters, elements, expected_trainable_dict):
    trainable_dict = expand_trainable_parameters(elements, trainable_parameters)
    print("elements=", elements)
    print("trainable_parameters=", trainable_parameters)
    print("trainable_dict=", trainable_dict)
    assert trainable_dict == expected_trainable_dict


def test_compute_bbasisset_train_mask_binary():
    bbasisconf = BBasisConfiguration("tests/ternary_AlCuZn.yaml")
    trainable_parameter_dict = {('Al', 'Cu'): ['func', 'radial'],
                                ('Al', 'Zn'): ['func', 'radial'],
                                ('Cu', 'Al'): ['func', 'radial'],
                                ('Cu', 'Zn'): ['func', 'radial'],
                                ('Zn', 'Al'): ['func', 'radial'],
                                ('Zn', 'Cu'): ['func', 'radial']}

    total_train_mask = compute_bbasisset_train_mask(bbasisconf, trainable_parameter_dict)

    expected_total_train_mask = np.array([False, False, False, False, False, False, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          False, False, False, False, False, False, True, True, True,
                                          True, True, True, False, False, False, False, False, False,
                                          False, False, False, False, False, False, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          False, False, False, False, True, True, True, True, True,
                                          True, True, True, True, True, True, True, False, False,
                                          False, False, True, True, True, True, False, False, False,
                                          False, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          False, False, False, True, True, True, True, True, True,
                                          True, False, False, False, False, True, True, True, True,
                                          True, True, True, True, True, True, True, True, False,
                                          False, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, False, False, False,
                                          True, True, False, False, True, True, True, True, True,
                                          True, False, False, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, False, False])

    assert np.all(total_train_mask == expected_total_train_mask)


def test_compute_bbasisset_train_mask_binary_func():
    bbasisconf = BBasisConfiguration("tests/ternary_AlCuZn.yaml")
    trainable_parameter_dict = {('Al', 'Cu'): ['func'],
                                ('Al', 'Zn'): ['func'],
                                ('Cu', 'Al'): ['func'],
                                ('Cu', 'Zn'): ['func'],
                                ('Zn', 'Al'): ['func'],
                                ('Zn', 'Cu'): ['func']}

    total_train_mask = compute_bbasisset_train_mask(bbasisconf, trainable_parameter_dict)

    expected_total_train_mask = np.array([False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          False, False, False, False, True, True, True, True, True,
                                          True, True, True, True, True, True, True, False, False,
                                          False, False, True, True, True, True, False, False, False,
                                          False, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          False, False, False, True, True, True, True, True, True,
                                          True, False, False, False, False, True, True, True, True,
                                          True, True, True, True, True, True, True, True, False,
                                          False, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, False, False, False,
                                          True, True, False, False, True, True, True, True, True,
                                          True, False, False, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True,
                                          True, False, False])

    assert np.all(total_train_mask == expected_total_train_mask)


def test_compute_bbasisset_train_mask_ternary():
    bbasisconf = BBasisConfiguration("tests/ternary_AlCuZn.yaml")
    trainable_parameter_dict = {('Al', 'Cu', 'Zn'): ['func'],
                                ('Cu', 'Al', 'Zn'): ['func'],
                                ('Zn', 'Al', 'Cu'): ['func']}

    total_train_mask = compute_bbasisset_train_mask(bbasisconf, trainable_parameter_dict)

    expected_total_train_mask = np.array([False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, True, True,
                                          True, True, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, True, True, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, True, True, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False, False, False, False, False, False, False,
                                          False, False, False])

    assert np.all(total_train_mask == expected_total_train_mask)


@pytest.mark.parametrize("num_funcs", [i for i in range(0, 20, 4)])
def test_extend_unary_config(num_funcs):
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe'],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_conf = extend_multispecies_basis(initial_config, final_config, num_funcs=num_funcs)
    print("ext_conf=", ext_conf)
    print("ext_conf.total_number_of_functions=", ext_conf.total_number_of_functions)
    assert ext_conf.total_number_of_functions == len(ext_conf.funcspecs_blocks) * num_funcs


@pytest.mark.parametrize("num_funcs", [i for i in range(0, 20, 4)])
def test_extend_binary_config(num_funcs):
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe', 'H', ],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_conf = extend_multispecies_basis(initial_config, final_config, num_funcs=num_funcs)
    print("ext_conf=", ext_conf)
    print("ext_conf.total_number_of_functions=", ext_conf.total_number_of_functions)
    assert ext_conf.total_number_of_functions == len(ext_conf.funcspecs_blocks) * num_funcs


@pytest.mark.parametrize("num_funcs", [i for i in range(0, 20, 4)])
def test_extend_ternary_config(num_funcs):
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe', 'H', 'Cr'],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_conf = extend_multispecies_basis(initial_config, final_config, num_funcs=num_funcs)
    print("ext_conf=", ext_conf)
    print("ext_conf.total_number_of_functions=", ext_conf.total_number_of_functions)
    assert ext_conf.total_number_of_functions == len(ext_conf.funcspecs_blocks) * num_funcs


@pytest.mark.parametrize("num_funcs", [i for i in range(0, 11, 4)])
def test_extend_quaternary_config(num_funcs):
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe', 'H', 'Cr', "Zn"],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_conf = extend_multispecies_basis(initial_config, final_config, num_funcs=num_funcs)

    print("ext_conf=", ext_conf)
    print("ext_conf.total_number_of_functions=", ext_conf.total_number_of_functions)
    ext_conf.save("ext_quaternary.yaml")
    assert ext_conf.total_number_of_functions == len(ext_conf.funcspecs_blocks) * num_funcs


@pytest.mark.parametrize("num_funcs", [i for i in range(0, 11, 4)])
def test_extend_quinary_config(num_funcs):
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe', 'H', 'Cr', "Zn", "W"],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_conf = extend_multispecies_basis(initial_config, final_config, num_funcs=num_funcs)
    ext_conf.save("ext_quinary.yaml")
    print("ext_conf=", ext_conf)
    print("ext_conf.total_number_of_functions=", ext_conf.total_number_of_functions)
    assert ext_conf.total_number_of_functions == len(ext_conf.funcspecs_blocks) * num_funcs


def test_extend_ternary_config_crad_preserving():
    pot_config_init = {'deltaSplineBins': 0.001,
                       'elements': ['Fe', 'H', 'Cr'],
                       'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                              'fs_parameters': [1, 1, 1, 0.5],
                                              'ndensity': 2,
                                              'rho_core_cut': 200000,
                                              'drho_core_cut': 250}},
                       'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                         'core-repulsion': [0.0, 5.0],
                                         'dcut': 0.01,
                                         'rcut': 6.2,
                                         'radbase': 'ChebExpCos',
                                         'radparameters': [5.25]}},
                       'functions': {
                           'ALL': {'nradmax_by_orders': [2],
                                   'lmax_by_orders': [0],
                                   'coefs_init': 'zero'},
                           'TERNARY': {'nradmax_by_orders': [2, 1],
                                       'lmax_by_orders': [0, 1],
                                       'coefs_init': 'zero'}}
                       }

    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Fe', 'H', 'Cr'],
                  'embeddings': {'ALL': {'npot': 'FinnisSinclairShiftedScaled',
                                         'fs_parameters': [1, 1, 1, 0.5],
                                         'ndensity': 2,
                                         'rho_core_cut': 200000,
                                         'drho_core_cut': 250}},
                  'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                                    'core-repulsion': [0.0, 5.0],
                                    'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos',
                                    'radparameters': [5.25]}},
                  'functions': {'ALL': {'nradmax_by_orders': [5, 2, 2, 1],
                                        'lmax_by_orders': [0, 2, 2, 1],
                                        'coefs_init': 'zero'}}}
    initial_config = create_multispecies_basis_config(pot_config_init)
    final_config = create_multispecies_basis_config(pot_config)

    print("initial_config=", initial_config)
    print("initial_config.total_number_of_functions=", initial_config.total_number_of_functions)
    new_coefs = 42 * np.ones_like(initial_config.get_radial_coeffs())
    print("new_coefs=", new_coefs)
    initial_config.set_radial_coeffs(new_coefs)
    i_crad = initial_config.get_radial_coeffs()
    print("i_crad=", i_crad)
    initial_config.save("te_init_config.yaml")
    for num_funcs in range(1, 10, 1):
        ext_config = extend_multispecies_basis(initial_config, final_config, num_funcs=1)

        ext_config.save("te_ext_config.yaml")
        crad = ext_config.get_radial_coeffs()
        print("e_crad=", crad)
        assert 42 in crad


def test_ZrNb_basis_extension():
    pot_config = {
        "deltaSplineBins": 0.001,
        "elements": ["Zr", "Nb"],

        "embeddings": {
            "ALL": {
                "npot": 'FinnisSinclairShiftedScaled',
                "fs_parameters": [1, 1, 1, 0.5],
                "ndensity": 2,
                "rho_core_cut": 200000,
                "drho_core_cut": 250
            }
        },

        "bonds": {"ALL": {
            "NameOfCutoffFunction": "cos",
            "core-repulsion": [0.0, 5.0],
            "dcut": 0.01,
            "rcut": 6.2,
            "radbase": "ChebExpCos",
            "radparameters": [5.25],
        }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": [5, 2],
                "lmax_by_orders": [0, 2],
                "coefs_init": "zero"
            }

        }
    }

    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_basis = extend_multispecies_basis(initial_config, final_config, num_funcs=1)
    ext_basis.save("test_ZrNb_ext_basis.yaml")
    loaded_basis = BBasisConfiguration("test_ZrNb_ext_basis.yaml")


def test_FeHCr_basis_extension():
    pot_config = {
        "deltaSplineBins": 0.001,
        "elements": ["Fe", "H", "Cr"],

        "embeddings": {
            "ALL": {
                "npot": 'FinnisSinclairShiftedScaled',
                "fs_parameters": [1, 1, 1, 0.5],
                "ndensity": 2,
                "rho_core_cut": 200000,
                "drho_core_cut": 250
            }
        },

        "bonds": {"ALL": {
            "NameOfCutoffFunction": "cos",
            "core-repulsion": [0.0, 5.0],
            "dcut": 0.01,
            "rcut": 6.2,
            "radbase": "ChebExpCos",
            "radparameters": [5.25],
        }
        },

        "functions": {
            "ALL": {
                "nradmax_by_orders": [10, 3, 2, 1, 1],
                "lmax_by_orders": [0, 2, 1, 1, 1],
                "coefs_init": "zero"
            }
        }
    }

    initial_config = BBasisConfiguration()
    final_config = create_multispecies_basis_config(pot_config)
    ext_basis = extend_multispecies_basis(initial_config, final_config, num_funcs=1)
    ext_basis.save("test_FeHCr_ext_basis.yaml")
    loaded_basis = BBasisConfiguration("test_FeHCr_ext_basis.yaml")


def test_merge_Fe_Cr():
    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    fecr_ladder0_pot = fe_pot + cr_pot

    fecr_ladder0_pot.save("CrFe_ladder0_initial.yaml")
    fecr_ladder0_pot.validate(True)
    fecr_ladder0_pot_loaded = BBasisConfiguration("CrFe_ladder0_initial.yaml")


def test_merge_Fe_Cr_extend_basis():
    pot_config = {'deltaSplineBins': 0.001, 'elements': ['Cr', 'Fe'],
                  'embeddings': {
                      'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5], 'ndensity': 2,
                              'rho_core_cut': 200000, 'drho_core_cut': 250}},
                  'bonds': {
                      'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01, 'rcut': 6.2,
                              'radbase': 'ChebExpCos', 'radparameters': [5.25], "inner_cutoff_type": "density"}},
                  'functions': {'ALL': {'nradmax_by_orders': [2, 3, 2, 1, 1], 'lmax_by_orders': [0, 2, 2, 1, 1],
                                        'coefs_init': 'zero'}},
                  'initial_potential': 'CrFe_ladder0_initial.yaml'}

    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    initial_config = fe_pot + cr_pot
    initial_config.validate(True)

    final_config = create_multispecies_basis_config(pot_config)
    ext_basis = extend_multispecies_basis(initial_config, final_config, num_funcs=1)
    print("total_number_of_functions = ", ext_basis.total_number_of_functions)
    ext_basis.save("CrFe_extended.yaml")
    assert ext_basis.total_number_of_functions == 34


def test_extend_FeCr_with_initial_Fe_Cr():
    binary_only_pot_config = {'deltaSplineBins': 0.001,
                              'elements': ['Fe', 'Cr'],
                              'bonds': {'BINARY': {'NameOfCutoffFunction': 'cos',
                                                   'core-repulsion': [0.0, 5.0],
                                                   'dcut': 0.01,
                                                   'rcut': 6.2,
                                                   'radbase': 'ChebExpCos',
                                                   "inner_cutoff_type": "density",
                                                   'radparameters': [5.25]}},
                              'functions': {'BINARY': {'nradmax_by_orders': [5, 2, 2, 1],
                                                       'lmax_by_orders': [0, 2, 2, 1],
                                                       'coefs_init': 'zero'}}}
    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    initial_config = fe_pot + cr_pot
    initial_config.validate(True)

    binary_pot_conf = create_multispecies_basis_config(binary_only_pot_config, initial_basisconfig=initial_config)
    print(binary_pot_conf)


def test_merge_Fe_Cr_extend_H():
    pot_config = {'deltaSplineBins': 0.001, 'elements': ['Cr', 'Fe'], 'embeddings': {
        'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5], 'ndensity': 2,
                'rho_core_cut': 200000, 'drho_core_cut': 250}},

                  'bonds': {
                      'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01, 'rcut': 6.2,
                              'radbase': 'ChebExpCos', 'radparameters': [5.25], "inner_cutoff_type": "density"}},
                  'functions': {
                      'ALL': {'nradmax_by_orders': [2, 3, 2, 1, 1], 'lmax_by_orders': [0, 2, 2, 1, 1],
                              'coefs_init': 'zero'}},
                  'initial_potential': 'CrFe_ladder0_initial.yaml'}

    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    initial_config = fe_pot + cr_pot
    initial_config.validate(True)

    final_config = create_multispecies_basis_config(pot_config)
    ext_basis = extend_multispecies_basis(initial_config, final_config, num_funcs=3)
    new_radial_coeffs = np.ones_like(ext_basis.get_radial_coeffs()) * 2
    ext_basis.set_radial_coeffs(new_radial_coeffs)

    print("total_number_of_functions = ", ext_basis.total_number_of_functions)
    ext_basis.save("CrFe_extended1.yaml")
    assert ext_basis.total_number_of_functions == 42

    pot_config_ternary = {'deltaSplineBins': 0.001, 'elements': ['Cr', 'Fe', 'H'], 'embeddings': {
        'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5], 'ndensity': 2,
                'rho_core_cut': 200000, 'drho_core_cut': 250}},
                          'bonds': {
                              'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01,
                                      'rcut': 6.2,
                                      'radbase': 'ChebExpCos', 'radparameters': [5.25],
                                      'inner_cutoff_type': "density"}}, 'functions': {
            'ALL': {'nradmax_by_orders': [2, 3, 2, 1, 1], 'lmax_by_orders': [0, 2, 2, 1, 1], 'coefs_init': 'zero'}},
                          'initial_potential': 'CrFe_ladder0_initial.yaml'}

    final_config_ternary = create_multispecies_basis_config(pot_config_ternary)
    ext_basis_2 = extend_multispecies_basis(ext_basis, final_config_ternary, num_funcs=1)
    crad2 = ext_basis_2.get_radial_coeffs()
    print("crad2=", crad2)
    crad2_expected = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    ext_basis_2.save("CrFe_extended2.yaml")
    assert ext_basis_2.total_number_of_functions == 54
    assert np.allclose(crad2, crad2_expected)


def test_merge_Fe_Cr_overwrite_blocks_from_initial_bbasis():
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Cr', 'Fe'],
                  'embeddings': {
                      'ALL': {
                          'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5], 'ndensity': 2,
                          'rho_core_cut': 200000, 'drho_core_cut': 250
                      }
                  },

                  'bonds': {
                      'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01, 'rcut': 6.2,
                              'radbase': 'ChebExpCos', 'radparameters': [5.25], "inner_cutoff_type": "density"}},
                  'functions': {
                      'ALL': {'nradmax_by_orders': [2, 3, 2, 1, 1], 'lmax_by_orders': [0, 2, 2, 1, 1],
                              'coefs_init': 'zero'}},
                  }

    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    initial_config = fe_pot + cr_pot
    initial_config.validate(True)

    final_config = create_multispecies_basis_config(pot_config,
                                                    initial_basisconfig=initial_config,
                                                    overwrite_blocks_from_initial_bbasis=False
                                                    )
    print(final_config.total_number_of_functions)

    assert final_config.total_number_of_functions == 554

    num_funcs = [len(bl.funcspecs) for bl in final_config.funcspecs_blocks]
    print("num_funcs=", num_funcs)  # [52,52,225,225]
    assert num_funcs == [52, 52, 225, 225]
    final_config_overwrite_blocks = create_multispecies_basis_config(pot_config,
                                                                     initial_basisconfig=initial_config,
                                                                     overwrite_blocks_from_initial_bbasis=True
                                                                     )
    print(final_config_overwrite_blocks.total_number_of_functions)

    assert final_config_overwrite_blocks.total_number_of_functions == 480

    num_funcs_overwrite = [len(bl.funcspecs) for bl in final_config_overwrite_blocks.funcspecs_blocks]
    print("num_funcs_overwrite=", num_funcs_overwrite)
    assert num_funcs_overwrite == [15, 15, 225, 225]
    assert len(final_config.funcspecs_blocks[0].funcspecs) == 52
    assert len(final_config_overwrite_blocks.funcspecs_blocks[0].funcspecs) == 15

def test_merge_Fe_Cr_extend_H_overwrite_blocks_from_initial_bbasis():
    pot_config = {'deltaSplineBins': 0.001,
                  'elements': ['Cr', 'Fe', 'H'],
                  'embeddings': {
                      'ALL': {
                          'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5], 'ndensity': 2,
                          'rho_core_cut': 200000, 'drho_core_cut': 250
                      }
                  },

                  'bonds': {
                      'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01, 'rcut': 6.2,
                              'radbase': 'ChebExpCos', 'radparameters': [5.25], "inner_cutoff_type": "density"}},
                  'functions': {
                      'ALL': {'nradmax_by_orders': [2, 3, 2, 1, 1], 'lmax_by_orders': [0, 2, 2, 1, 1],
                              'coefs_init': 'zero'}},
                  }

    cr_pot = BBasisConfiguration("tests/Cr_ladder_0.yaml")
    fe_pot = BBasisConfiguration("tests/Fe_ladder_0.yaml")
    initial_config = fe_pot + cr_pot
    initial_config.validate(True)

    final_config = create_multispecies_basis_config(pot_config,
                                                    initial_basisconfig=initial_config,
                                                    overwrite_blocks_from_initial_bbasis=False
                                                    )
    print(final_config.total_number_of_functions)
    assert final_config.total_number_of_functions == 2541

    num_funcs = [len(bl.funcspecs) for bl in final_config.funcspecs_blocks]
    print("num_funcs=", num_funcs)  # [52,52,225,225]
    assert num_funcs == [52, 52, 52, 225, 225, 225, 225, 225, 225, 345, 345, 345]

    final_config_overwrite_blocks = create_multispecies_basis_config(pot_config,
                                                                     initial_basisconfig=initial_config,
                                                                     overwrite_blocks_from_initial_bbasis=True
                                                                     )
    print(final_config_overwrite_blocks.total_number_of_functions)

    assert final_config_overwrite_blocks.total_number_of_functions == 2467

    num_funcs_overwrite = [len(bl.funcspecs) for bl in final_config_overwrite_blocks.funcspecs_blocks]
    print("num_funcs_overwrite=", num_funcs_overwrite)
    assert num_funcs_overwrite ==[15, 15, 52, 225, 225, 225, 225, 225, 225, 345, 345, 345]
    assert len(final_config.funcspecs_blocks[0].funcspecs) == 52
    assert len(final_config_overwrite_blocks.funcspecs_blocks[0].funcspecs) == 15

def test_ZrNb_empty_initial_potentail():
    potential_config = {'deltaSplineBins': 0.001, 'elements': ['Zr', 'Nb'],
                        'embeddings': {
                            'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5],
                                    'ndensity': 2,
                                    'rho_core_cut': 200000, 'drho_core_cut': 250}},
                        'bonds': {
                            'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01,
                                    'rcut': 6.2,
                                    'radbase': 'ChebExpCos', 'radparameters': [5.25]}},
                        'functions': {
                            'ALL': {'nradmax_by_orders': [12, 5, 3, 2, 1], 'lmax_by_orders': [0, 3, 2, 2, 1],
                                    'coefs_init': 'zero'}}}
    initial_bbasisconfig = construct_bbasisconfiguration(potential_config)
    for block in initial_bbasisconfig.funcspecs_blocks:
        block.lmaxi = 0
        block.nradmaxi = 0
        block.nradbaseij = 0
        block.radcoefficients = []
        block.funcspecs = []
    target_bbasisconfig = construct_bbasisconfiguration(potential_config,
                                                        initial_basisconfig=initial_bbasisconfig)


def test_inner_cutoff_basis_generation_default_distance():
    potential_config = {'deltaSplineBins': 0.001, 'elements': ['Zr', 'Nb', "Al"],
                        'embeddings': {
                            'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5],
                                    'ndensity': 2,
                                    'rho_core_cut': 200000, 'drho_core_cut': 250}},
                        'bonds': {
                            'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01,
                                    'rcut': 6.2, 'radbase': 'ChebExpCos', 'radparameters': [5.25],
                                    "inner_cutoff_type": "density",
                                    "r_in": 2.0, "delta_in": 1.0}},
                        'functions': {
                            'ALL': {'nradmax_by_orders': [5, 2, 1, 1, 1], 'lmax_by_orders': [0, 1, 1, 1, 1],
                                    'coefs_init': 'zero'}}}

    bbasisconfig = construct_bbasisconfiguration(potential_config)
    bbasisconfig.save("saved_inner_cutoff_pot.yaml")
    loaded_bbasisconfig = BBasisConfiguration("saved_inner_cutoff_pot.yaml")
    for block in loaded_bbasisconfig.funcspecs_blocks:
        # print(block)
        if block.number_of_species <= 2:
            assert block.r_in == 2.0
            assert block.delta_in == 1.0
            assert block.inner_cutoff_type == "density"
        else:
            assert block.r_in == 0.0
            assert block.delta_in == 0.0
            assert block.inner_cutoff_type == "density"  # density is new default


def test_inner_cutoff_basis_generation():
    potential_config = {'deltaSplineBins': 0.001, 'elements': ['Zr', 'Nb', "Al"],
                        'embeddings': {
                            'ALL': {'npot': 'FinnisSinclairShiftedScaled', 'fs_parameters': [1, 1, 1, 0.5],
                                    'ndensity': 2,
                                    'rho_core_cut': 200000, 'drho_core_cut': 250}},
                        'bonds': {
                            'ALL': {'NameOfCutoffFunction': 'cos', 'core-repulsion': [0.0, 5.0], 'dcut': 0.01,
                                    'rcut': 6.2, 'radbase': 'ChebExpCos', 'radparameters': [5.25],
                                    "r_in": 2.0, "delta_in": 1.0, "inner_cutoff_type": "density"}},
                        'functions': {
                            'ALL': {'nradmax_by_orders': [5, 2, 1, 1, 1], 'lmax_by_orders': [0, 1, 1, 1, 1],
                                    'coefs_init': 'zero'}}}

    bbasisconfig = construct_bbasisconfiguration(potential_config)
    bbasisconfig.save("saved_inner_cutoff_pot.yaml")
    loaded_bbasisconfig = BBasisConfiguration("saved_inner_cutoff_pot.yaml")
    for block in loaded_bbasisconfig.funcspecs_blocks:
        # print(block)
        if block.number_of_species <= 2:
            assert block.r_in == 2.0
            assert block.delta_in == 1.0
            assert block.inner_cutoff_type == "density"
        else:
            assert block.r_in == 0.0
            assert block.delta_in == 0.0
            assert block.inner_cutoff_type == "density"  # density is new default

def test_number_of_functions_per_element():
    potential_config = {
        'deltaSplineBins': 0.001,
        'elements': ['Al', 'Mg'],

        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclair',
                               'rho_core_cut': 200000},
                       },

        'bonds': {'ALL': {'NameOfCutoffFunction': 'cos',
                          'core-repulsion': [10000.0, 5.0],
                          'dcut': 0.01,
                          'radbase': 'ChebPow',
                          'radparameters': [2.0],
                          'rcut': 6},
                  },

        'functions': {
            'ALL': {
                'nradmax_by_orders': [10, 4, 3, 2, 1],
                'lmax_by_orders':    [0,  3, 2, 1, 1]
            },
            'number_of_functions_per_element': 200
        }
    }

    bconf = create_multispecies_basis_config(potential_config=potential_config)
    print("total_number_of_functions=",bconf.total_number_of_functions)
    assert bconf.total_number_of_functions == 400
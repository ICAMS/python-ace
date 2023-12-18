import pytest
from pytest import fail

from pyace import *
from pyace.basisextension import *

test_potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],  # , 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 2,
    "nradmax_by_orders": [4, 3],
    "lmax_by_orders": [0, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
}


def print_basisconfig(basis):
    print("LEN:", len(basis.funcspecs_blocks[0].funcspecs))
    print(".crad.shape=", np.shape(basis.funcspecs_blocks[0].radcoefficients))
    print(".crad=", basis.funcspecs_blocks[0].radcoefficients)
    print(".funcspecs=")
    for func in basis.funcspecs_blocks[0].funcspecs:
        print("\t-", func)


@pytest.mark.parametrize("pot_file_name, step", [
    ("tests/Al-r1l0.yaml", 1),
    ("tests/Al-r1l0.yaml", 2),
    ("tests/Al-r1l0.yaml", 10),
    ("tests/Al-r1l0.yaml", 15),
    ("tests/Al-r1l0.yaml", 16),
    ("tests/Al-r1l0.yaml", 20),
])
def test_extend_basis_grow(pot_file_name, step):
    old_basis = BBasisConfiguration(pot_file_name)
    new_basis = construct_bbasisconfiguration(test_potential_config)

    old_num_of_funcs = len(old_basis.funcspecs_blocks[0].funcspecs)

    all_new_funcs = new_basis.funcspecs_blocks[0].funcspecs
    new_num_of_funcs = len(all_new_funcs)

    print("old_basis=")
    print_basisconfig(old_basis)

    print("new_basis=")
    print_basisconfig(new_basis)

    assert new_num_of_funcs > old_num_of_funcs

    print("=" * 20)
    ext_basis = old_basis
    for _ in range((new_num_of_funcs - old_num_of_funcs) // step + 1):
        ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
        ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
        print("curr_ext_basis=")
        print_basisconfig(ext_basis)
        new_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
        new_ext_num_of_funcs = len(new_ext_funcs)
        assert new_ext_num_of_funcs == min(ext_num_of_funcs + step, new_num_of_funcs)
        if all_new_funcs[:new_ext_num_of_funcs] != new_ext_funcs:
            print("ERROR: expected:")
            for f in all_new_funcs[:new_ext_num_of_funcs]:
                print("\t- ", f)
            print("got:")
            for f in new_ext_funcs:
                print("\t- ", f)
            assert all_new_funcs[:new_ext_num_of_funcs] != new_ext_funcs

# @pytest.mark.xfail
# @pytest.mark.parametrize("pot_file_name, step", [
#     ("tests/Al-r1l0-hole.yaml", 1),
#     ("tests/Al-r1l0-hole.yaml", 2),
#     ("tests/Al-r1l0-hole.yaml", 10),
#     ("tests/Al-r1l0-hole.yaml", 15),
#     ("tests/Al-r1l0-hole.yaml", 16),
#     ("tests/Al-r1l0-hole.yaml", 20),
# ])
# def test_extend_basis_grow_with_hole(pot_file_name, step):
#     old_basis = BBasisConfiguration(pot_file_name)
#     new_basis = construct_bbasisconfiguration(test_potential_config)
#
#     old_num_of_funcs = len(old_basis.funcspecs_blocks[0].funcspecs)
#     all_new_funcs = new_basis.funcspecs_blocks[0].funcspecs[1:]  # skip first
#     new_num_of_funcs = len(all_new_funcs)
#
#     print("old_basis=")
#     print_basisconfig(old_basis)
#     print("new_basis=")
#     print_basisconfig(new_basis)
#     assert new_num_of_funcs > old_num_of_funcs
#     print("=" * 20)
#     ext_basis = old_basis
#     for _ in range((new_num_of_funcs - old_num_of_funcs) // step + 1):
#         ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
#         print("curr_ext_basis=")
#         print_basisconfig(ext_basis)
#         new_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
#         new_ext_num_of_funcs = len(new_ext_funcs)
#         assert new_ext_num_of_funcs == min(ext_num_of_funcs + step, new_num_of_funcs)
#         if all_new_funcs[:new_ext_num_of_funcs] != new_ext_funcs:
#             print("ERROR: expected:")
#             for f in all_new_funcs[:new_ext_num_of_funcs]:
#                 print("\t- ", f)
#             print("got:")
#             for f in new_ext_funcs:
#                 print("\t- ", f)
#             assert all_new_funcs[:new_ext_num_of_funcs] != new_ext_funcs


# ("tests/Al-r234.yaml", 1),
# ("tests/Al-r234-2.yaml", 1),
# ("tests/Al-r1234l12_crad_dif.yaml", 1),
# ("tests/Al-r1234l12_crad_dif_2.yaml", 1),
# @pytest.mark.xfail
# def test_extend_basis_grow_shrink(pot_file_name="tests/Al-r1234.yaml", step=1):
#     initial_basis = BBasisConfiguration(pot_file_name)
#     target_basis = construct_bbasisconfiguration(test_potential_config)
#
#     old_num_of_funcs = len(initial_basis.funcspecs_blocks[0].funcspecs)
#     all_new_funcs = target_basis.funcspecs_blocks[0].funcspecs  # skip first
#     new_num_of_funcs = len(all_new_funcs)
#
#     print("Initial basis=")
#     print_basisconfig(initial_basis)
#     print("Target basis=")
#     print_basisconfig(target_basis)
#     # assert new_num_of_funcs > old_num_of_funcs
#
#     ext_basis = initial_basis
#     it = 0
#     break_reached = False
#     for _ in range((new_num_of_funcs - old_num_of_funcs) // step + 3):
#         print("=" * 20, "it = ", it)
#         old_num_of_funcs = ext_basis.total_number_of_functions
#         ext_basis, is_extended = extend_basis(ext_basis, target_basis, 'body_order', step, return_is_extended = True)
#         new_num_of_funcs = ext_basis.total_number_of_functions
#         print("curr_ext_basis=")
#         print_basisconfig(ext_basis)
#         it += 1
#         print("target_basis.total_number_of_functions=", target_basis.total_number_of_functions)
#         print("ext_basis.total_number_of_functions=", ext_basis.total_number_of_functions)
#         if not is_extended:
#             print("BREAK!!!")
#             break_reached = True
#             break
#
#     assert break_reached
#     all_coeffs = ext_basis.get_all_coeffs()
#     print("all_coeffs = ", all_coeffs)
#     assert all_coeffs == [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
#                           0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# @pytest.mark.xfail
# def test_extend_basis_shrink(step=1):
#     initial_basis = construct_bbasisconfiguration(test_potential_config)
#     target_basis = BBasisConfiguration("tests/Al-r1l0.yaml")
#     print("Initial basis=")
#     print_basisconfig(initial_basis)
#     print("Target basis=")
#     print_basisconfig(target_basis)
#
#     it = 0
#
#     current_bbasis = initial_basis
#     while True:
#         print("=" * 20, "it = ", it)
#         current_bbasis, is_extended = extend_basis(current_bbasis, target_basis, 'body_order', step, return_is_extended = True)
#         print("curr_ext_basis=")
#         print_basisconfig(current_bbasis)
#         if not is_extended:
#             break
#
#     all_coeffs = current_bbasis.get_all_coeffs()
#     print("all_coeffs = ", all_coeffs)
#
#     assert all_coeffs == [0.0]


def test_extend_basis_equal(step=1):
    initial_basis = BBasisConfiguration("tests/Al-r234.yaml")
    target_basis = BBasisConfiguration("tests/Al-r234.yaml")
    print("Initial basis=")
    print_basisconfig(initial_basis)
    print("Target basis=")
    print_basisconfig(target_basis)

    it = 0

    current_bbasis = initial_basis
    while True:
        print("=" * 20, "it = ", it)
        current_bbasis, is_extended = extend_basis(current_bbasis, target_basis, 'body_order', step, return_is_extended = True)
        print("curr_ext_basis=")
        print_basisconfig(current_bbasis)
        if not is_extended:
            break

    all_coeffs = current_bbasis.get_all_coeffs()
    print("all_coeffs = ", all_coeffs)

    assert all_coeffs == [1.0, 1, 1, 1, 1, 1]


def test_extend_basis_equal_power_order(step=1):
    initial_basis = BBasisConfiguration("tests/Al-r234.yaml")
    target_basis = BBasisConfiguration("tests/Al-r234.yaml")
    print("Initial basis=")
    print_basisconfig(initial_basis)
    print("Target basis=")
    print_basisconfig(target_basis)

    it = 0

    current_bbasis = initial_basis
    while True:
        print("=" * 20, "it = ", it)
        current_bbasis, is_extended = extend_basis(current_bbasis, target_basis, 'power_order', step, return_is_extended = True)
        print("curr_ext_basis=")
        print_basisconfig(current_bbasis)
        if not is_extended:
            break

    all_coeffs = current_bbasis.get_all_coeffs()
    print("all_coeffs = ", all_coeffs)

    assert all_coeffs == [1.0, 1, 1, 1, 1, 1]


test_potential_config1 = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],  # 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 3,
    "nradmax_by_orders": [4, 3, 2],
    "lmax_by_orders": [0, 1, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
}

# @pytest.mark.xfail
# def test_extend_basis_with_holes():
#     old_basis = BBasisConfiguration("tests/Al_r3_test_exp_hole.yaml")
#     new_basis = construct_bbasisconfiguration(test_potential_config1)
#
#     old_num_of_funcs = len(old_basis.funcspecs_blocks[0].funcspecs)
#     new_num_of_funcs = len(new_basis.funcspecs_blocks[0].funcspecs)
#     # print("old_basis=")
#     # print_basisconfig(old_basis)
#     print("new_basis=")
#     print_basisconfig(new_basis)
#
#     assert new_num_of_funcs > old_num_of_funcs
#
#     step = 1
#
#     print("=" * 20)
#     ext_basis = old_basis
#     for _ in range(6):
#         ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
#         print("curr_ext_basis=")
#         print_basisconfig(ext_basis)
#         new_ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         print(new_ext_num_of_funcs)
#         print("=" * 20)
#     assert new_ext_num_of_funcs == 23
#

test_potential_config2 = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],  # 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 4,
    "nradmax_by_orders": [4, 3, 2, 1],
    "lmax_by_orders": [0, 2, 1, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': './data/pyace_bbasisfunc_df_rho2.pckl',
}


# @pytest.mark.xfail
# def test_extend_basis_with_holes_r4():
#     old_basis = BBasisConfiguration("tests/Al_r4_test_exp_hole.yaml")
#     new_basis = construct_bbasisconfiguration(test_potential_config2)
#
#     old_num_of_funcs = len(old_basis.funcspecs_blocks[0].funcspecs)
#     new_num_of_funcs = len(new_basis.funcspecs_blocks[0].funcspecs)
#     print("old_basis=")
#     print_basisconfig(old_basis)
#     print("new_basis=")
#     print_basisconfig(new_basis)
#
#     assert new_num_of_funcs > old_num_of_funcs
#
#     step = 1
#
#     print("=" * 20)
#     ext_basis = old_basis
#     for _ in range(6):
#         ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         ext_basis,is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
#         print("curr_ext_basis=")
#         print_basisconfig(ext_basis)
#         new_ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         print(new_ext_num_of_funcs)
#         print("=" * 20)
#     assert new_ext_num_of_funcs == 33
#

test_potential_config3 = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],  # 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 3,
    "nradmax_by_orders": [3, 2, 2],
    "lmax_by_orders": [0, 2, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': './data/pyace_bbasisfunc_df_rho2.pckl',
}

# @pytest.mark.xfail
# def test_shrink_basis_with_holes_r4():
#     old_basis = BBasisConfiguration("tests/Al_r4_test_exp_hole.yaml")
#     new_basis = construct_bbasisconfiguration(test_potential_config3)
#     crad_new = np.array(
#         [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
#     print('Old shape: ', (np.shape(old_basis.funcspecs_blocks[0].radcoefficients)))
#     print('NEW shape: ', crad_new.shape)
#     old_num_of_funcs = len(old_basis.funcspecs_blocks[0].funcspecs)
#     new_num_of_funcs = len(new_basis.funcspecs_blocks[0].funcspecs)
#     print("old_basis=")
#     print_basisconfig(old_basis)
#     print("new_basis=")
#     print_basisconfig(new_basis)
#
#     assert new_num_of_funcs < old_num_of_funcs
#
#     step = 1
#
#     print("=" * 20)
#     ext_basis = old_basis
#     for _ in range(6):
#         ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
#         print("curr_ext_basis=")
#         print_basisconfig(ext_basis)
#         new_ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#         print(new_ext_num_of_funcs)
#         print("=" * 20)
#     assert new_ext_num_of_funcs == 20
#     assert ext_basis.funcspecs_blocks[0].radcoefficients == [[[1.0, 13.0, 12.0], [1.0, 3.14, 0.0], [1.0, 0.0, 0.0]],
#                                                              [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]
#     assert np.shape(ext_basis.funcspecs_blocks[0].radcoefficients) == crad_new.shape


test_potential_config_large = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1],  # 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 6,
    "nradmax_by_orders": [15, 5, 4, 1, 1, 1],
    "lmax_by_orders": [0, 6, 4, 2, 1, 1],
    "ndensity": 1,
    "rcut": 8.7,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    # 'basisdf': 'data/pyace_selected_bbasis_funcspec.pckl.gzip',
}


# @pytest.mark.xfail
def test_extend_large_basis_grow():
    step = 100
    old_basis = BBasisConfiguration("tests/Al-r1l0.yaml")
    old_num_of_funcs = 1
    new_basis = construct_bbasisconfiguration(test_potential_config_large)

    true_bconf = construct_bbasisconfiguration(test_potential_config_large) #BBasisConfiguration("tests/Cu.pbe.in.yaml")
    all_new_funcs = new_basis.funcspecs_blocks[0].funcspecs
    all_true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
    new_num_of_funcs = len(all_new_funcs)

    # print("old_basis=")
    # print_basisconfig(old_basis)
    #
    # print("new_basis=")
    # print_basisconfig(new_basis)

    assert new_num_of_funcs > old_num_of_funcs

    print("=" * 20)
    ext_basis = old_basis
    while True:
        ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
        ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'body_order', step, return_is_extended = True)
        # print("curr_ext_basis=")
        # print_basisconfig(ext_basis)
        new_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
        new_ext_num_of_funcs = len(new_ext_funcs)
        assert new_ext_num_of_funcs == min(ext_num_of_funcs + step, new_num_of_funcs)

        # print("expected:")
        # for f in all_new_funcs[:new_ext_num_of_funcs]:
        #     print("\t- ", f)
        # print("got:")
        # for f in new_ext_funcs:
        #     print("\t- ", f)
        if not is_extended:
            break

    assert new_ext_num_of_funcs == new_num_of_funcs
    assert len(ext_basis.funcspecs_blocks)==1

    all_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
    ext_num_of_funcs = len(all_ext_funcs)
    assert ext_num_of_funcs == new_num_of_funcs

    all_ext_set = set()
    for new_func in all_ext_funcs:
        all_ext_set.add((tuple(new_func.ns), tuple(new_func.ls), tuple(new_func.LS)))

    all_true_set = set()
    for new_func in all_true_funcs:
        all_true_set.add((tuple(new_func.ns), tuple(new_func.ls), tuple(new_func.LS)))

    print("Number of  expected function:", len(all_true_set))
    print("Number of actually extended functions:", len(all_ext_funcs))

    diff1 = sorted(all_ext_set - all_true_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))
    diff2 = sorted(all_true_set - all_ext_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))

    print("SHOULD NOT BE BUT EXISTS")
    for comb in diff1:
        print(comb, " alternatives in TRUE found:",
              list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_true_set)))

    print("\n" * 2)
    print("SHOULD BE BUT MISSED")
    for comb in diff2:
        print(comb, " alternatives in PROPOSED found:",
              list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_ext_set)))

    assert len(all_true_set) == 684
    assert len(all_ext_funcs) == len(all_true_set)

    ext_funcspecs = ext_basis.funcspecs_blocks[0].funcspecs

    new_df = pd.DataFrame({"spec": ext_funcspecs})

    new_df["ns"] = new_df["spec"].map(lambda f: tuple(f.ns))
    new_df["ls"] = new_df["spec"].map(lambda f: tuple(f.ls))
    new_df["LS"] = new_df["spec"].map(lambda f: tuple(f.LS))

    true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
    true_df = pd.DataFrame({"spec": true_funcs})
    true_df["ns"] = true_df["spec"].map(lambda f: tuple(f.ns))
    true_df["ls"] = true_df["spec"].map(lambda f: tuple(f.ls))
    true_df["LS"] = true_df["spec"].map(lambda f: tuple(f.LS))

    assert (true_df[["ns", "ls", "LS"]] == new_df[["ns", "ls", "LS"]]).all().all(), "Basis inconsistent"


# @pytest.mark.xfail
def test_extend_large_basis_grow_power_order():
    step = 100
    old_basis = BBasisConfiguration("tests/Al-r1l0.yaml")
    old_num_of_funcs = 1
    new_basis = construct_bbasisconfiguration(test_potential_config_large)

    true_bconf = construct_bbasisconfiguration(test_potential_config_large) # BBasisConfiguration("tests/Cu.pbe.in.yaml")
    all_new_funcs = new_basis.funcspecs_blocks[0].funcspecs
    all_true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
    new_num_of_funcs = len(all_new_funcs)

    # print("old_basis=")
    # print_basisconfig(old_basis)
    #
    # print("new_basis=")
    # print_basisconfig(new_basis)

    assert new_num_of_funcs > old_num_of_funcs

    print("=" * 20)
    ext_basis = old_basis
    while True:
        ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
        ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'power_order', step, return_is_extended = True)
        # print("curr_ext_basis=")
        # print_basisconfig(ext_basis)
        new_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
        new_ext_num_of_funcs = len(new_ext_funcs)
        assert new_ext_num_of_funcs == min(ext_num_of_funcs + step, new_num_of_funcs)

        # print("expected:")
        # for f in all_new_funcs[:new_ext_num_of_funcs]:
        #     print("\t- ", f)
        # print("got:")
        # for f in new_ext_funcs:
        #     print("\t- ", f)
        if not is_extended:
            break

    assert new_ext_num_of_funcs == new_num_of_funcs

    all_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
    ext_num_of_funcs = len(all_ext_funcs)
    assert ext_num_of_funcs == new_num_of_funcs

    all_ext_set = set()
    for new_func in all_ext_funcs:
        all_ext_set.add((tuple(new_func.ns), tuple(new_func.ls), tuple(new_func.LS)))

    all_true_set = set()
    for new_func in all_true_funcs:
        all_true_set.add((tuple(new_func.ns), tuple(new_func.ls), tuple(new_func.LS)))

    print("Number of  expected function:", len(all_true_set))
    print("Number of actually extended functions:", len(all_ext_funcs))

    diff1 = sorted(all_ext_set - all_true_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))
    diff2 = sorted(all_true_set - all_ext_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))

    print("SHOULD NOT BE BUT EXISTS")
    for comb in diff1:
        print(comb, " alternatives in TRUE found:",
              list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_true_set)))

    print("\n" * 2)
    print("SHOULD BE BUT MISSED")
    for comb in diff2:
        print(comb, " alternatives in PROPOSED found:",
              list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_ext_set)))

    assert len(all_true_set) == 684
    assert len(all_ext_funcs) == len(all_true_set)

    ext_funcspecs = ext_basis.funcspecs_blocks[0].funcspecs

    new_df = pd.DataFrame({"spec": ext_funcspecs})

    new_df["ns"] = new_df["spec"].map(lambda f: tuple(f.ns))
    new_df["ls"] = new_df["spec"].map(lambda f: tuple(f.ls))
    new_df["LS"] = new_df["spec"].map(lambda f: tuple(f.LS))

    true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
    true_df = pd.DataFrame({"spec": true_funcs})
    true_df["ns"] = true_df["spec"].map(lambda f: tuple(f.ns))
    true_df["ls"] = true_df["spec"].map(lambda f: tuple(f.ls))
    true_df["LS"] = true_df["spec"].map(lambda f: tuple(f.LS))

    assert (true_df[["ns", "ls", "LS"]] == new_df[["ns", "ls", "LS"]]).all().all(), "Basis inconsistent"


test_potential_config_power_grow = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1, 1, 0.5],
    "npot": "FinnisSinclair",
    "NameOfCutoffFunction": "cos",

    # "rankmax": 4,
    "nradmax_by_orders": [12, 7, 4, 2],
    "lmax_by_orders": [0, 4, 2, 2],
    "ndensity": 2,
    "rcut": 6.0,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    'basisdf': 'data/pyace_selected_bbasis_funcspec.pckl.gzip',
}


# @pytest.mark.xfail
# def test_extend_large_basis_grow_power_order_step():
#     step = 30
#     old_basis = BBasisConfiguration("tests/power_order_r1.yaml")
#     old_num_of_funcs = 1
#     new_basis = construct_bbasisconfiguration(test_potential_config_power_grow)
#
#     true_bconf = construct_bbasisconfiguration(test_potential_config_power_grow) #BBasisConfiguration("tests/power_order_growth30_true.yaml")
#     all_new_funcs = new_basis.funcspecs_blocks[0].funcspecs
#     all_true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
#     new_num_of_funcs = len(all_true_funcs)
#
#     assert new_num_of_funcs > old_num_of_funcs
#
#     print("=" * 20)
#     ext_basis = old_basis
#
#     ext_num_of_funcs = len(ext_basis.funcspecs_blocks[0].funcspecs)
#     ext_basis, is_extended = extend_basis(ext_basis, new_basis, 'power_order', step, return_is_extended = True)
#     # print("curr_ext_basis=")
#     # print_basisconfig(ext_basis)
#     ext_basis.save("extended_basis.yaml")
#     new_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
#     new_ext_num_of_funcs = len(new_ext_funcs)
#     assert new_ext_num_of_funcs == min(ext_num_of_funcs + step, new_num_of_funcs)
#
#     assert new_ext_num_of_funcs == new_num_of_funcs
#
#     all_ext_funcs = ext_basis.funcspecs_blocks[0].funcspecs
#     ext_num_of_funcs = len(all_ext_funcs)
#     assert ext_num_of_funcs == new_num_of_funcs
#
#     all_ext_set = set()
#     for new_func in all_ext_funcs:
#         all_ext_set.add((tuple(new_func.ns), tuple(new_func.ls), tuple(new_func.LS)))
#
#     all_true_set = set()
#     for new_func in all_true_funcs:
#         # if len(new_func.ns)>1:
#         nl_combs = sorted([(n,l) for n,l in zip(new_func.ns,new_func.ls)])
#         # print("nl_combs=",nl_combs)
#         ns, ls = [v for v in zip(*nl_combs)]
#         # else:
#         #     ns = new_func.ns
#         #     ls = new_func.ls
#         all_true_set.add((tuple(ns), tuple(ls), tuple(new_func.LS)))
#
#     print("Number of  expected function:", len(all_true_set))
#     print("Number of actually extended functions:", len(all_ext_funcs))
#
#     diff1 = sorted(all_ext_set - all_true_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))
#     diff2 = sorted(all_true_set - all_ext_set, key=lambda d: (len(d[0]), d[0], d[1], d[2]))
#
#     print("SHOULD NOT BE BUT EXISTS")
#     for comb in diff1:
#         print(comb, " alternatives in TRUE found:",
#               list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_true_set)))
#
#     print("\n" * 2)
#     print("SHOULD BE BUT MISSED")
#     for comb in diff2:
#         print(comb, " alternatives in PROPOSED found:",
#               list(filter(lambda d: (d[0] == comb[0] and d[1] == comb[1]), all_ext_set)))
#
#     assert len(all_true_set) == 42
#     assert len(all_ext_funcs) == len(all_true_set)
#
#     ext_funcspecs = ext_basis.funcspecs_blocks[0].funcspecs
#
#     new_df = pd.DataFrame({"spec": ext_funcspecs})
#
#     new_df["ns"] = new_df["spec"].map(lambda f: tuple(f.ns))
#     new_df["ls"] = new_df["spec"].map(lambda f: tuple(f.ls))
#     new_df["LS"] = new_df["spec"].map(lambda f: tuple(f.LS))
#
#     true_funcs = true_bconf.funcspecs_blocks[0].funcspecs
#     true_df = pd.DataFrame({"spec": true_funcs})
#     true_df["ns"] = true_df["spec"].map(lambda f: tuple(f.ns))
#     true_df["ls"] = true_df["spec"].map(lambda f: tuple(f.ls))
#     true_df["LS"] = true_df["spec"].map(lambda f: tuple(f.LS))
#
#     assert (true_df[["ns", "ls", "LS"]] == new_df[["ns", "ls", "LS"]]).all().all(), "Basis inconsistent"


def test_prepare_bbasisfuncspecifications_non_default_element():
    pot_config = {
        "deltaSplineBins": 0.001,
        "element": "Cu",
        "fs_parameters": [1, 1, 1, 0.5],
        "npot": "FinnisSinclairShiftedScaled",
        "NameOfCutoffFunction": "cos",
        "nradmax_by_orders": [5, 2, 1],
        "lmax_by_orders": [0, 2, 1],
        "ndensity": 2,
        "rcut": 5,
        "dcut": 0.01,
        "radparameters": [5.25],
        "radbase": "ChebExpCos",
        # 'basisdf': 'data/pyace_bbasisfunc_df_rho2.pckl',
    }
    basisconf = construct_bbasisconfiguration(pot_config)
    basis_spec_list = basisconf.funcspecs_blocks[0].funcspecs
    spec = basis_spec_list[0]
    assert spec.elements == ["Cu", "Cu"]


def test_construct_bbasisconfiguration_SBessel():
    pot_config = {
        "deltaSplineBins": 0.001,
        "element": "Cu",
        "fs_parameters": [1, 1, 1, 0.5],
        "npot": "FinnisSinclairShiftedScaled",
        "NameOfCutoffFunction": "cos",
        "nradmax_by_orders": [5, 2, 1],
        "lmax_by_orders": [0, 2, 1],
        "ndensity": 2,
        "rcut": 5,
        "dcut": 0.01,
        "radparameters": [5.25],
        "radbase": "TEST_SBessel"
    }
    basisconf = construct_bbasisconfiguration(pot_config)
    basis_spec_list = basisconf.funcspecs_blocks[0].funcspecs
    spec = basis_spec_list[0]
    assert spec.elements == ["Cu", "Cu"]
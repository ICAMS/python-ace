import pytest

from pyace.coupling import ACECouplingTree
from pyace.coupling import generate_ms_cg_list, MsCgPair
from pyace.coupling import validate_ls_LS, is_valid_ls_LS, expand_ls_LS


# from pyace.coupling import expand_ls_LS_wrapper

def test_ACECouplingTree():
    ct = ACECouplingTree(3)
    print("ct.tree_indices_array=", ct.tree_indices_array)
    assert ct.tree_indices_array == [0, 1, 3, 3, 2, -1]
    ct.tree_indices_array.append(3)
    assert ct.tree_indices_array == [0, 1, 3, 3, 2, -1]


def test_MsCgPair():
    mcp = MsCgPair()
    assert mcp.gen_cg == 0
    assert mcp.ms == []


def test_generate_ms_cg_list():
    res = generate_ms_cg_list([1, 1, 0], [0], half_basis=False)
    print(res)
    assert len(res) == 3
    ms_cg = res[0]
    assert abs(ms_cg.gen_cg - 0.5773502691896257) < 1e-9
    assert ms_cg.ms == [1, -1, 0]


def test_generate_ms_cg_list_noLS():
    res = generate_ms_cg_list([1, 1, 0], half_basis=False)
    print(res)
    assert len(res) == 3
    ms_cg = res[0]
    assert abs(ms_cg.gen_cg - 0.5773502691896257) < 1e-9
    assert ms_cg.ms == [1, -1, 0]


def test_generate_ms_cg_list_def_half_basis():
    res = generate_ms_cg_list([1, 1, 0], [0])
    assert len(res) == 2
    ms_cg = res[0]
    assert abs(ms_cg.gen_cg - 2 * 0.5773502691896257) < 1e-9
    assert ms_cg.ms == [1, -1, 0]


def test_compare_and_noLS():
    res1 = generate_ms_cg_list([1, 1, 0, 0], [0, 0], half_basis=False)
    assert len(res1) == 3
    res2 = generate_ms_cg_list([1, 1, 0, 0], [0], half_basis=False)
    assert len(res2) == 3
    assert res1[-1].ms == res2[-1].ms
    assert res1[0] == res2[0]
    assert res1[0] != res2[1]
    assert res1 == res2


def test_expand_ls_LS():
    ls = []
    LS = []
    ls, LS = expand_ls_LS(1, ls, LS)
    assert ls == [0]
    assert LS == []

    ls = [1]
    LS = []
    ls, LS = expand_ls_LS(2, ls, LS)
    assert ls == [1, 1]
    assert LS == []

    ls = [0, 1, 2]
    LS = []
    ls, LS = expand_ls_LS(3, ls, LS)
    assert ls == [0, 1, 2]
    assert LS == [2]


def test_validate_ls_LS():
    assert is_valid_ls_LS([], []) == True
    assert is_valid_ls_LS([0], []) == True
    assert is_valid_ls_LS([1], []) == False
    assert is_valid_ls_LS([1, 1], []) == True
    assert is_valid_ls_LS([0, 1, 1], []) == False
    assert is_valid_ls_LS([0, 1, 1], [1]) == True

    with pytest.raises(ValueError):
        validate_ls_LS([1], [])
    with pytest.raises(ValueError):
        validate_ls_LS([0, 1, 1], [])
    with pytest.raises(ValueError):
        validate_ls_LS([0, 1, 2], [2])


def test_generate_equivariant_ms_cg_list():
    res = generate_ms_cg_list([1, 1], L=1, M=0, half_basis=False, check_is_even=False)
    print(res)
    assert len(res) == 2
    ms_cg = res[0]
    assert abs(ms_cg.gen_cg - 0.707107) < 1e-5
    assert ms_cg.ms == [1, -1]

import pytest

from pyace import ACEBBasisSet
from pyace.basisextension import construct_bbasisconfiguration
from pyace.generalfit import get_maximal_rcut


def build_potential_config(bonds):
    return {
        'deltaSplineBins': 0.001,
        'elements': ['Al', 'Ni'],

        'embeddings': {'ALL': {'drho_core_cut': 250,
                               'fs_parameters': [1, 1],
                               'ndensity': 1,
                               'npot': 'FinnisSinclairShiftedScaled',
                               'rho_core_cut': 200000}},

        'bonds': bonds,

        'functions': {
            'UNARY': {
                'nradmax_by_orders': [5, 2],
                'lmax_by_orders': [0, 0]
            },
            'BINARY': {
                'nradmax_by_orders': [5, 2, 2],
                'lmax_by_orders': [0, 2, 2],
            },
        }
    }


DEFAULT_BOND = {'NameOfCutoffFunction': 'cos',
                'core-repulsion': [10000.0, 5.0],
                'dcut': 0.01,
                'radbase': 'ChebPow',
                'radparameters': [2.0],
                'lmax': 2,
                'rcut': 3.9}


def test_get_maximal_rcut_uniform():
    config = construct_bbasisconfiguration(build_potential_config({'ALL': DEFAULT_BOND}))
    assert get_maximal_rcut(config) == pytest.approx(3.9)


def test_get_maximal_rcut_per_bond():
    bonds = {'ALL': DEFAULT_BOND,
             'AlNi': dict(DEFAULT_BOND, rcut=5.5)}
    config = construct_bbasisconfiguration(build_potential_config(bonds))
    assert get_maximal_rcut(config) == pytest.approx(5.5)


@pytest.mark.parametrize("binary_rcut", [3.9, 5.5])
def test_get_maximal_rcut_equals_cutoffmax(binary_rcut):
    """
    The cutoff deduced for the neighbour lists must coincide with `ACEBBasisSet.cutoffmax`,
    which is what the ACE calculators (ASE, LAMMPS) use in production. `get_maximal_rcut`
    only inspects one- and two-species blocks, while `cutoffmax` is a maximum over all
    blocks - these agree because the radial basis (`rcut` included) is copied from the
    pair block into the higher-rank blocks.
    """
    bonds = {'ALL': DEFAULT_BOND,
             'AlNi': dict(DEFAULT_BOND, rcut=binary_rcut)}
    config = construct_bbasisconfiguration(build_potential_config(bonds))
    assert get_maximal_rcut(config) == pytest.approx(ACEBBasisSet(config).cutoffmax)

import numpy as np

from ase.build import bulk
from pyace.activeexploration import ActiveExploration

MULTISPECIES_TESTS_DF_PCKL = "tests/df_AlNi(murn).pckl.gzip"
TESTS_DF_PCKL = "tests/representative_df.pckl.gzip"
COMPRESSION = "gzip"


def test_active_exploration():
    atoms = bulk("Al", cubic=True)

    print(atoms)
    print(atoms.get_scaled_positions())

    potname = "tests/multispecies_AlNi.yaml"
    asiname = "tests/multispecies_AlNi.asi"

    ae = ActiveExploration(potname, asiname)
    ae_atoms = ae.active_exploration(atoms, initial_moving_atom_index=0, n_iter=1)

    print(ae_atoms)
    sp = ae_atoms.get_scaled_positions()
    print(ae_atoms.get_scaled_positions())
    assert len(ae.extrapolative_structures) == 2
    assert np.allclose(sp[0], [0.01042486, 0.99573258, 0.00799615])


def test_active_exploration_gamma_lo_None():
    atoms = bulk("Al", cubic=True)

    print(atoms)
    print(atoms.get_scaled_positions())

    potname = "tests/multispecies_AlNi.yaml"
    asiname = "tests/multispecies_AlNi.asi"

    ae = ActiveExploration(potname, asiname)
    ae_atoms = ae.active_exploration(atoms, initial_moving_atom_index=0, n_iter=4, gamma_lo=None)

    print(ae_atoms)
    sp = ae_atoms.get_scaled_positions()
    print(sp)
    assert len(ae.extrapolative_structures) == 5


def test_active_exploration_movable_atoms():
    atoms = bulk("Al", cubic=True)

    print(atoms)
    sp0 = atoms.get_scaled_positions()
    print(sp0)

    potname = "tests/multispecies_AlNi.yaml"
    asiname = "tests/multispecies_AlNi.asi"

    ae = ActiveExploration(potname, asiname)
    movable_atoms_indices = [0, 1]
    ae_atoms = ae.active_exploration(atoms, movable_atoms_indices=movable_atoms_indices, gamma_lo=None)

    print(ae_atoms)
    sp = ae_atoms.get_scaled_positions()
    print(sp)
    assert len(ae.extrapolative_structures) == 3
    assert not np.allclose(sp0[movable_atoms_indices], sp[movable_atoms_indices])
    assert np.allclose(sp0[[2, 3]], sp[[2, 3]])

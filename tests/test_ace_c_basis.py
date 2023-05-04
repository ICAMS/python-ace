import numpy as np
import os
import pickle

import pytest

from pyace.evaluator import ACECTildeEvaluator
import pyace.atomicenvironment as pe
from pyace.calculator import ACECalculator
from pyace.basis import ACECTildeBasisSet


def test_cbasis():
    pp = ACECTildeBasisSet()
    pp.load("tests/Al.pbe.in-rank1.ace")
    assert pp.lmax == 0
    assert pp.nradbase == 12
    assert pp.nradmax == 0
    assert pp.nelements == 1
    # assert pp.rankmax == 1
    assert pp.ndensitymax == 2
    assert pp.cutoffmax == 8.7


def test_ACECTildeBasisSet_pickle():
    cbasis = ACECTildeBasisSet()
    cbasis.load("tests/Al.pbe.in-rank1.ace")

    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    energy = ace.energy
    forces = ace.forces
    print("Energy: ", energy)
    print("forces: ", forces)

    pickled_cbasis = pickle.dumps(cbasis)
    print("pickled_cbasis =", pickled_cbasis)
    unpickled_cbasis = pickle.loads(pickled_cbasis)

    new_evaluator = ACECTildeEvaluator()
    new_evaluator.set_basis(unpickled_cbasis)
    new_ace = ACECalculator()
    new_ace.set_evaluator(new_evaluator)
    new_atoms = pe.create_linear_chain(2)
    new_ace.compute(new_atoms)

    new_energy = new_ace.energy
    new_forces = new_ace.forces
    print("new energy: ", new_energy)
    print("new forces: ", new_forces)

    assert energy == new_energy
    assert forces == new_forces


def test_ACECTildeBasisSet_all_coefs():
    cbasis = ACECTildeBasisSet()
    cbasis.load("tests/Al.pbe.in-rank1.ace")
    all_coeffs = cbasis.all_coeffs
    print("all_coeffs=", all_coeffs)
    assert np.allclose(all_coeffs,
                       [0.37281243865961977, -1.17905321855563, 1527.5284707464255, -13408.315757728185,
                        -165.62272197809557, 6694.716760764875, -997.4820842451243, 67.56069396765703,
                        1422.3154526839494, -3870.5973287276875, -1210.6170483560968, 4227.430047642065,
                        752.2398165154448, -2707.0395878945324, -355.7548629547101, 1111.213341542384,
                        125.10396203207958, -234.50867148171997, -30.04276887026525, -19.557847735831277,
                        3.8207609041419146, 22.654371715550976, 1.73644e-13, -2.808363682801332])
    cbasis.all_coeffs = all_coeffs

    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    energy = ace.energy
    forces = ace.forces
    print("Energy: ", energy)
    print("forces: ", forces)

    assert np.allclose(energy, 175.59324909105746)
    assert np.allclose(forces, [[0.0, 0.0, -665.9729936394472], [0.0, 0.0, 665.9729936394472]])

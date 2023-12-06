import numpy as np
import os
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from pyace.asecalc import PyACECalculator, PyACEEnsembleCalculator
from pyace.atomicenvironment import aseatoms_to_atomicenvironment, create_cube, create_linear_chain
from pyace.basis import ACECTildeBasisSet, ACEBBasisSet, FexpShiftedScaled
from pyace.calculator import ACECalculator
from pyace.evaluator import ACECTildeEvaluator, ACEBEvaluator, ACERecursiveEvaluator, get_ace_evaluator_version


def test_calculator_energy():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.in-rank1.ace")
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 9.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)
    assert abs(calculator.energy + 271.755644394495) < 1E-5


def test_calculator_forces():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.in-rank1.ace")
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 9.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)
    f = calculator.forces
    print("calculator.forces=", f)

    assert len(f) == atoms.n_atoms_real


def test_calculator_energy_rank3_corerep():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe-in.rank3-core-rep.ace")
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 3.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)

    e0 = 5451.5152094384885
    f0 = -367.12320042084258

    # e0 =  5451.5414550234818
    # f0 = -367.13574172876906

    assert abs(calculator.energy - (e0)) < 1E-7
    f = calculator.forces
    assert abs(f[0][0] - (f0)) < 1E-7
    assert abs(f[1][1] - (f0)) < 1E-7
    assert abs(f[7][2] - (-f0)) < 1E-7


def test_calculator_energy_rhocore():
    fcc_at = bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3)
    print("fcc_at=", fcc_at)
    fcc_ae = aseatoms_to_atomicenvironment(fcc_at)
    print("fcc_ae=", fcc_ae)
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.rhocore.ace")
    print("basis loaded")
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(basis)
    print("evaluator.set_basis(basis) done")
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    print("calculator.set_evaluator(evaluator) done")

    print("call calculator.compute")
    calculator.compute(fcc_ae)
    print(calculator.energy)
    e0 = -482.2811898327966  # -18.21569651045736 / 4 * fcc_ae.n_atoms_real
    print("expected e0=", e0)
    f0 = 0
    f1 = 0
    f2 = 0

    assert abs(calculator.energy - (e0)) < 1E-7
    f = calculator.forces
    print(f)

    assert abs(f[0][0] - (f0)) < 1E-7
    assert abs(f[1][1] - (f1)) < 1E-7
    assert abs(f[-1][2] - (f2)) < 1E-7


# def test_calculator_cbasis_projections():
#     basis = ACECTildeBasisSet()
#     basis.load("tests/Al.pbe.rhocore.ace")
#     evaluator = ACECTildeEvaluator()
#     evaluator.set_basis(basis)
#     chain_z = create_linear_chain(3)
#     chain_x = create_linear_chain(3, axis=0)
#
#     calculator = ACECalculator()
#     calculator.set_evaluator(evaluator)
#
#     calculator.compute(chain_z)
#     z_projs = np.array(calculator.projections)
#
#     calculator.compute(chain_x)
#     x_projs = np.array(calculator.projections)
#
#     print()
#     print("z: z_projs=", z_projs)
#
#     print("x: x_projs=", x_projs)
#
#     assert chain_z.n_atoms_real == len(z_projs)
#     assert len(z_projs[0]) == 117+12
#
#     assert np.allclose(z_projs, x_projs)


def test_calculator_bbasis_projections():
    basis = ACEBBasisSet()
    basis.load("tests/Al.pbe.13.2.yaml")
    evaluator = ACEBEvaluator()
    evaluator.set_basis(basis)
    chain_z = create_linear_chain(3)
    chain_x = create_linear_chain(3, axis=0)

    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)

    calculator.compute(chain_z, compute_projections=True)
    z_projs = np.array(calculator.projections)

    z_rhos = np.array(calculator.rhos)
    z_dF_drhos = np.array(calculator.dF_drhos)

    calculator.compute(chain_x)
    x_projs = np.array(calculator.projections)

    x_rhos = np.array(calculator.rhos)
    x_dF_drhos = np.array(calculator.dF_drhos)

    print()
    print("z: z_projs=", z_projs)

    print("x: x_projs=", x_projs)



    print("x: x_projs=", x_projs)

    print("x_rhos=", x_rhos)
    print("z_rhos=", z_rhos)

    print("x_dF_drhos=", x_dF_drhos)
    print("z_dF_drhos=", z_dF_drhos)

    assert chain_z.n_atoms_real == len(z_projs)

    assert len(z_projs[0]) == 187 + 10

    assert chain_z.n_atoms_real == len(z_rhos)
    assert chain_z.n_atoms_real == len(z_dF_drhos)

    assert np.allclose(z_projs, x_projs)

    assert np.allclose(z_rhos, x_rhos)
    assert np.allclose(z_dF_drhos, x_dF_drhos)


@pytest.mark.parametrize("potname", ["tests/Al.pbe.13.2.yaml",
                                     "tests/Al-Ni_opt_all.yaml",
                                     ])
def test_ASE_calculator_bbasis_projections(potname):
    calculator = PyACECalculator(potname)
    atoms = bulk("Al", "fcc")
    calculator.calculate(atoms)
    projections = calculator.projections
    print("projections=", projections)
    assert len(projections) == len(atoms)
    bbasis = calculator.basis
    # Al - 0
    nfunc = len(bbasis.basis_rank1[0]) + len(bbasis.basis[0])
    ndens = bbasis.map_embedding_specifications[0].ndensity

    print("ndens = ", ndens)
    print("nfunc = ", nfunc)
    assert len(projections[0]) == nfunc


def test_ASE_calculator_Voigt_stress_order():
    potname = "tests/Al.pbe.rhocore.ace"
    calculator = PyACECalculator(potname)

    test_Al = bulk("Al", "fcc", a=4.05, cubic=True)
    eps = 1e-2
    deform_matrix = np.array([[1 - eps, eps, eps],
                              [-eps, 1 + eps, -eps],
                              [eps, eps, 1 - eps]])
    cell = test_Al.get_cell()
    new_cell = np.dot(cell, deform_matrix)
    test_Al.set_cell(new_cell, scale_atoms=True)

    test_Al.set_calculator(calculator)
    stresses = test_Al.get_stress()
    print("stresses=", stresses)
    stresses_ref = [1.39710511e-01, 1.53093581e-01, 1.39710511e-01, -9.67787622e-05,
                    2.42101868e-03, -9.67787622e-05]
    assert np.allclose(stresses, stresses_ref)


@pytest.mark.parametrize("potname", ["tests/Al-Ni_opt_all.yaml", "tests/Al-Ni_opt_all.yace"])
def test_ASE_calculator_multispecies_AlNi(potname):
    calculator = PyACECalculator(potname)
    test_Al = bulk("Al", "fcc", a=4.05, cubic=True)
    chemsymb = test_Al.get_chemical_symbols()
    chemsymb[0] = "Ni"
    test_Al.set_chemical_symbols(chemsymb)

    test_Al.set_calculator(calculator)

    e = test_Al.get_potential_energy()
    ee = test_Al.get_potential_energies()
    f = test_Al.get_forces()
    s = test_Al.get_stress()

    print("e=", e)
    print("ee=", ee)
    print("f=", f)
    print("s=", s)

    eref = 306.0646090745618
    eeref = np.array([341.38851822, -11.77463638, -11.77463638, -11.77463638])
    fref = np.zeros((4, 3))
    s0 = -4.82378191e+00
    sref = np.array([s0, s0, s0, 0, 0, 0])

    assert np.allclose(e, eref)
    assert np.allclose(ee, eeref)
    assert np.allclose(f, fref)
    assert np.allclose(s, sref)


def test_Ni_2dens_FexpShiftedScaled_manual_e_consistency():
    calc = PyACECalculator("tests/Ni_2dens_bug_case.yaml")
    print("calc.basis.basis_rank1=", calc.basis.basis_rank1)
    print("calc.basis.basis=", calc.basis.basis)
    f1 = calc.basis.basis_rank1[0][0]
    f2 = calc.basis.basis[0][0]

    print("f1.coeffs=", f1.coeffs)
    print("f2.coeffs=", f2.coeffs)

    trimer = Atoms('Ni3', positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], pbc=0)
    trimer.set_calculator(calc)

    e = trimer.get_potential_energy()
    proj = calc.projections
    p1 = proj[0, 0]
    p2 = proj[0, 1]
    print("Projections=", p1, p2)

    rho1 = p1 * f1.coeffs[0] + p2 * f2.coeffs[0]
    rho2 = p1 * f1.coeffs[1] + p2 * f2.coeffs[1]

    print("rho1, rho2 = ", rho1, rho2)
    manual_e = 3 * (rho1 + FexpShiftedScaled(rho2, 0.5)[0])

    print("manual_e=", manual_e)
    print("e = ", e)
    assert np.allclose(manual_e, e)


def test_calculator_energy_recursive():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.in-rank1.ace")
    evaluator = ACERecursiveEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 9.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)
    assert abs(calculator.energy + 271.755644394495) < 1E-5


def test_calculator_forces_recursive():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.in-rank1.ace")
    evaluator = ACERecursiveEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 9.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)
    f = calculator.forces
    print("calculator.forces=", f)

    assert len(f) == atoms.n_atoms_real


def test_calculator_energy_rank3_corerep_recursive():
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe-in.rank3-core-rep.ace")
    evaluator = ACERecursiveEvaluator()
    evaluator.set_basis(basis)
    atoms = create_cube(3., 3.)
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    calculator.compute(atoms, False)

    e0 = 5451.5152094384885
    f0 = -367.12320042084258
    assert abs(calculator.energy - (e0)) < 1E-7
    f = calculator.forces
    assert abs(f[0][0] - (f0)) < 1E-7
    assert abs(f[1][1] - (f0)) < 1E-7
    assert abs(f[7][2] - (-f0)) < 1E-7


def test_calculator_energy_rhocore_recursive():
    fcc_at = bulk("Al", "fcc", a=4.05, cubic=True) * (3, 3, 3)
    print("fcc_at=", fcc_at)
    fcc_ae = aseatoms_to_atomicenvironment(fcc_at)
    print("fcc_ae=", fcc_ae)
    basis = ACECTildeBasisSet()
    basis.load("tests/Al.pbe.rhocore.ace")
    print("basis loaded")
    evaluator = ACERecursiveEvaluator()
    evaluator.set_basis(basis)
    print("evaluator.set_basis(basis) done")
    calculator = ACECalculator()
    calculator.set_evaluator(evaluator)
    print("calculator.set_evaluator(evaluator) done")

    print("call calculator.compute")
    calculator.compute(fcc_ae)
    print(calculator.energy)
    e0 = -482.2811898327966  # -18.21569651045736 / 4 * fcc_ae.n_atoms_real
    print("expected e0=", e0)
    f0 = 0
    f1 = 0
    f2 = 0

    assert abs(calculator.energy - (e0)) < 1E-7
    f = calculator.forces
    print(f)

    assert abs(f[0][0] - (f0)) < 1E-7
    assert abs(f[1][1] - (f1)) < 1E-7
    assert abs(f[-1][2] - (f2)) < 1E-7


def test_get_ace_evaluator_version():
    version = get_ace_evaluator_version()
    print("version=", version)
    assert version is not None


def test_PyACEEnsembleCalculator():
    fcc_at = bulk("Al", "fcc", a=4.05, cubic=True)
    fnames = ["tests/Al.pbe.rhocore.ace", "tests/Al.pbe.rhocore.ace"]
    calc = PyACEEnsembleCalculator(fnames)
    fcc_at.set_calculator(calc)

    e = fcc_at.get_potential_energy()
    print("e=", e)
    assert np.allclose(e, -17.862266290103605)
    results = calc.results
    print("results=", results)
    print("results.keys=", results.keys())
    expected_keys = ['energy', 'free_energy', 'forces', 'energies', 'energy_std', 'free_energy_std', 'forces_std',
                     'energies_std', 'energy_dev', 'energies_dev', 'forces_dev', 'stress', 'stress_std', 'stress_dev']

    for exp_key in expected_keys:
        assert exp_key in results

    assert np.allclose(results["energy_dev"], 0)
    assert np.allclose(results["energies_dev"], [0, 0, 0, 0])
    assert np.allclose(results["forces_dev"], [0, 0, 0, 0])
    assert np.allclose(results["stress_dev"], [0, 0, 0, 0, 0, 0])


def test_PyACECalculator_active_set():
    atoms = bulk("Ag", cubic=True)
    atoms.set_chemical_symbols(["Ag", "Cu", "Ag", "Cu"])
    pos = atoms.get_positions()
    np.random.seed(42)
    pos += np.random.randn(*pos.shape)
    atoms.set_positions(pos)

    asecalc = PyACECalculator("tests/DFT10B-AgCu.yaml")
    asecalc.set_active_set("tests/DFT10B-AgCu.asi")

    atoms.set_calculator(asecalc)
    energy = atoms.get_potential_energy()
    gamma = asecalc.results["gamma"]
    print("energy = ", energy)
    print("gamma = ", gamma)
    energy_expected = -778.3043076364492
    gamma_expected = [5691014.320236206, 1711.20627784729, 5691197.948196411, 1647.2230472564697]

    assert np.allclose(energy, energy_expected)
    assert np.allclose(gamma, gamma_expected)


def test_PyACECalculator_active_set_dump_extrapolation():
    atoms = bulk("Ag", cubic=True)
    atoms.set_chemical_symbols(["Ag", "Cu", "Ag", "Cu"])
    pos = atoms.get_positions()
    np.random.seed(42)
    pos += np.random.randn(*pos.shape)
    atoms.set_positions(pos)

    asecalc = PyACECalculator("tests/DFT10B-AgCu.yaml",
                              dump_extrapolative_structures=True,
                              keep_extrapolative_structures=True,
                              stop_at_large_extrapolation=True)
    asecalc.set_active_set("tests/DFT10B-AgCu.asi")

    atoms.set_calculator(asecalc)

    expected_file_name = "extrapolation_0_gamma=5691197.948196411.cfg"
    if os.path.isfile(expected_file_name):
        os.remove(expected_file_name)
    assert not os.path.isfile(expected_file_name)
    with pytest.raises(RuntimeError) as excinfo:
        energy = atoms.get_potential_energy()
    print("Exception raised: ", excinfo)
    assert os.path.isfile(expected_file_name)

    # second config
    expected_file_name_1 = "extrapolation_1_gamma=5691197.948196411.cfg"
    if os.path.isfile(expected_file_name_1):
        os.remove(expected_file_name_1)
    assert not os.path.isfile(expected_file_name_1)
    asecalc.reset()
    with pytest.raises(RuntimeError) as excinfo:
        energy = atoms.get_potential_energy()
    print("Exception raised: ", excinfo)
    assert os.path.isfile(expected_file_name_1)

    os.remove(expected_file_name)
    os.remove(expected_file_name_1)

    assert len(asecalc.extrapolative_structures_gamma) == 2
    assert len(asecalc.extrapolative_structures_list) == 2

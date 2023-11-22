import numpy as np
import pytest

from ase.build import bulk
from ase.atoms import Atoms

from pyace import *


def create_dimer(x):
    return Atoms(["Al"] * 2, positions=[[0, 0, 0], [0, 0, x]], pbc=False)


def create_trimer(x):
    return Atoms(["Al"] * 3, positions=[[0, 0, 0], [0, 0, x], [0, 0, 2 * x]], pbc=False)


def test_setup():
    block = BBasisFunctionsSpecificationBlock()

    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 0
    block.npoti = "FinnisSinclair"
    block.fs_parameters = [1, 1]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1
    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [[[1]]]

    block.funcspecs = [
        BBasisFunctionSpecification(["Al", "Al"], ns=[1], ls=[0], LS=[], coeffs=[1.]),
        BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[0, 0], LS=[], coeffs=[2])
    ]

    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]

    a = create_dimer(1)
    print(a)
    calc = PyACECalculator(basis_set=basisConfiguration)
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)


def test_load_YAML():
    a = create_dimer(1)
    print(a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.13.2.yaml")
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)


def test_load_ace():
    a = create_dimer(1)
    print(a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.rhocore.ace")
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)

    e0 = 89.15966867577228
    f0 = np.array([[0., 0., -193.82052454], [0., 0., 193.82052454]])

    assert np.allclose(e0, e1)
    assert np.allclose(f0, f1)


def test_load_ace_recursive():
    a = create_dimer(1)
    print(a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.rhocore.ace", recursive_evaluator=True, recursive=True)
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)

    e0 = 89.15966867577228
    f0 = np.array([[0., 0., -193.82052454], [0., 0., 193.82052454]])

    assert np.allclose(e0, e1)
    assert np.allclose(f0, f1)


def test_dimer_r1_energy_forces():
    a = create_dimer(1)
    print(a)
    calc = PyACECalculator(basis_set="tests/Al-r1l0.yaml")
    a.set_calculator(calc)
    energy = a.get_potential_energy()
    forces = a.get_forces()
    print(energy)
    print(forces)
    assert (energy - 1.9355078359256011) < 5e-10
    assert (forces[0][2] + 0.12757969575334369) < 1e-11
    assert (forces[1][2] - 0.12757969575334369) < 1e-11


def test_trimer_r234_energy_forces():
    a = create_trimer(1)
    print(a)
    calculator = PyACECalculator(basis_set="tests/Al-r234.yaml")
    a.set_calculator(calculator)
    energy = a.get_potential_energy()
    forces = a.get_forces()
    print(energy)
    print(forces)
    assert (energy - 28.457404084390994) < 5e-10
    assert (forces[0][2] + 13.052474776412932) < 1e-11
    assert (forces[1][2] - 0.0000000000000000) < 1e-11


def test_load_YAML_pbc_symmetry_cubic():
    a = bulk("Al", "sc", a=1, cubic=True)
    print("a=", a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.13.2.yaml")
    calc.cutoff = np.sqrt(2) * 1
    a.set_calculator(calc)
    e1 = a.get_potential_energy()
    f1 = a.get_forces()
    print("ae=", calc.ae)
    print("ae.x=", calc.ae.x)
    print("ae.species_type=", calc.ae.species_type)
    print("ae.neighbour_list=", calc.ae.neighbour_list)
    print("len=", list(map(len, calc.ae.neighbour_list)))
    print("cutoff=", calc.cutoff)
    print(e1)
    print(f1)
    assert np.max(f1) < 1e-10


def test_load_YAML_pbc_symmetry_fcc_supercell():
    a = bulk("Al", "fcc", cubic=True) * (1, 2, 3)
    print("a=", a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.13.2.yaml")
    a.set_calculator(calc)
    e1 = a.get_potential_energy()
    f1 = a.get_forces()
    print("ae=", calc.ae)
    print("ae.x=", calc.ae.x)
    print("ae.species_type=", calc.ae.species_type)
    print("ae.neighbour_list=", calc.ae.neighbour_list)
    print("len=", list(map(len, calc.ae.neighbour_list)))
    print("cutoff=", calc.cutoff)
    print(e1)
    print(f1)
    assert np.max(f1) < 1e-10


def test_non_supported_element():
    a = bulk("C", "sc", a=1, cubic=True)
    print("a=", a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.13.2.yaml")
    with pytest.raises(ValueError):
        a.set_calculator(calc)
        e1 = a.get_potential_energy()


def test_fcc_stress():
    a = bulk("Al", "fcc", a=4.03, cubic=True) * (1, 1, 1)
    print("a=", a)
    calc = PyACECalculator(basis_set="tests/Al.pbe.in-rank1.ace")
    a.set_calculator(calc)
    e1 = a.get_potential_energy()
    f1 = a.get_forces()
    s1 = a.get_stress()
    print(e1)
    print(f1)
    print(s1)
    e_ref = -15.75993931278226
    s_ref = -4.00944358e-01
    assert np.max(f1) < 1e-10
    assert abs(e1 - e_ref) < 1e-7
    assert abs(s1[0] - s_ref) < 1e-7


def test_relaxation():
    from ase.constraints import UnitCellFilter
    from ase.optimize import QuasiNewton

    calc = PyACECalculator(basis_set="tests/Al.pbe.rhocore-v2.ace")
    fcc = bulk("Al", cubic=True)
    fcc.set_calculator(calc)
    print("Atoms before = ", fcc)
    e0 = fcc.get_potential_energy()
    f0 = fcc.get_forces()
    s0 = fcc.get_stress()

    print("e_pot=", e0)
    print("forces=", f0)
    print("stress=", s0)

    ucf = UnitCellFilter(fcc)
    qn = QuasiNewton(ucf)
    qn.run(fmax=0.005)

    print("Atoms after = ", fcc)
    e1 = fcc.get_potential_energy()
    f1 = fcc.get_forces()
    s1 = fcc.get_stress()

    print("e_pot=", e1)
    print("forces=", f1)
    print("stress=", s1)

    assert e1 < e0
    assert np.linalg.norm(s1) < np.linalg.norm(s0)
    assert np.allclose(np.linalg.norm(f1), np.linalg.norm(f0))


def test_PyACEEnsembleCalculator():
    basis_sets = ["tests/Al.pbe.13.2.yaml", "tests/Al.pbe.rhocore.ace"]
    calc = PyACEEnsembleCalculator(basis_set=basis_sets)
    a = create_dimer(1)
    a.set_calculator(calc)
    e1 = (a.get_potential_energy())
    f1 = a.get_forces()
    print(e1)
    print(f1)


def test_ZBL_analytical_derivative():
    calc = PyACECalculator("lib/ace/test/fitting/potentials/ZBL_rep.yaml")

    def check(at, msg):
        at.set_calculator(calc)
        num_forces = calc.calculate_numerical_forces(at, d=1e-6)
        an_forces = at.get_forces()
        print(f"{num_forces=}, {an_forces=}")
        assert np.allclose(num_forces, an_forces), msg+f": {num_forces=}, {an_forces=}"

    at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 4]])  # ACE
    check(at, "ACE forces are inconsistent")

    at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.5]])  # ACE-ZBL
    check(at, "ACE-ZBL forces are inconsistent")

    at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])  # ZBL
    check(at, "ZBL forces are inconsistent")

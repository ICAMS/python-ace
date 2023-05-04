from sys import stdout

import pytest
import pickle

import numpy as np
from pyace.evaluator import ACEBEvaluator, ACECTildeEvaluator
import pyace.atomicenvironment as pe
from pyace.calculator import ACECalculator
from pyace.basis import (ACECTildeBasisSet,
                         ACECTildeBasisFunction,
                         ACEBBasisSet,
                         BBasisConfiguration,
                         BBasisFunctionSpecification,
                         BBasisFunctionsSpecificationBlock, FexpShiftedScaled)

from ase import Atoms


def test_Fexp():
    xs = -np.arange(-1, 12)
    xs = np.power(10., xs)

    xs = np.sort(np.concatenate([xs, -xs]))
    print(xs)
    FS = []
    DFS = []
    for x in xs:
        F, DF = FexpShiftedScaled(x, 0.5)
        FS.append(F)
        DFS.append(DF)
        print("X: {}\tF: {}\t DF: {}".format(x, F, DF))
    print("FS=", FS)
    print("DFS=", DFS)

    F_ref = [-3.1622567547927845, -0.8610338964462405, -0.11872869271732522, -0.012431936894573337,
             -0.0012493131967951099, -0.0001249931256978587, -1.2499931250686824e-05, -1.2499993124470699e-06,
             -1.249999931340895e-07, -1.2499999924031613e-08, -1.24999993689201e-09, -1.2500001034254637e-10,
             -1.2500001034254637e-11, 1.2500001034254637e-11, 1.2500001034254637e-10, 1.24999993689201e-09,
             1.2499999924031613e-08, 1.249999931340895e-07, 1.2499993124470699e-06, 1.2499931250686824e-05,
             0.0001249931256978587, 0.0012493131967951099, 0.012431936894573337, 0.11872869271732522,
             0.8610338964462405, 3.1622567547927845]
    DF_ref = [0.15813469865510948, 0.6184148713968093, 1.1298188364982225, 1.2364549872082238, 1.2486270892661513,
              1.2498625209330063, 1.2499862502093706, 1.2499986250020938, 1.249999862500021, 1.2499999862500002,
              1.249999998625, 1.2499999998625, 1.24999999998625, 1.24999999998625, 1.2499999998625, 1.249999998625,
              1.2499999862500002, 1.249999862500021, 1.2499986250020938, 1.2499862502093706, 1.2498625209330063,
              1.2486270892661513, 1.2364549872082238, 1.1298188364982225, 0.6184148713968093, 0.15813469865510948]

    assert np.allclose(FS, F_ref)
    assert np.allclose(DFS, DF_ref)


def test_dimer_r1_energy_forces():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r1l0.yaml")
    ace = ACECalculator()

    evaln = ACEBEvaluator(basis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    assert abs(ace.energy - 1.9355078359256011) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 0.12757969575334369) < 1e-11
    assert abs(forces[1][2] - 0.12757969575334369) < 1e-11


def test_trimer_r234_energy_forces():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r234.yaml")
    ace = ACECalculator()

    evaln = ACEBEvaluator(basis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert abs(ace.energy - 28.457404084390994) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 13.052474776412932) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11


def test_constructor_load_yaml():
    basis = ACEBBasisSet("tests/Al-r234.yaml")
    ace = ACECalculator()

    evaln = ACEBEvaluator(basis)

    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert abs(ace.energy - 28.457404084390994) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 13.052474776412932) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11


def test_trimer_r1_initilize_basis():
    bfunc = BBasisFunctionSpecification(["Al", "Al"], [1], [0], [], [1.])
    print(bfunc)
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
    block.funcspecs = [bfunc]
    print(block)
    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]
    print(basisConfiguration)

    # basis = ACEBBasisSet(basisConfiguration)
    evaluator = ACEBEvaluator(basisConfiguration)

    ace = ACECalculator(evaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert np.abs((ace.energy - 5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11


def test_trimer_r1_initilize_basis_default_parameters():
    bfunc = BBasisFunctionSpecification(["Al", "Al"], [1], [1.])
    print(bfunc)
    block = BBasisFunctionsSpecificationBlock()
    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 0
    block.npoti = "FinnisSinclair"
    block.fs_parameters = [1, 1, ]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1

    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [[[1]]]
    block.funcspecs = [bfunc]
    print(block)
    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]
    print(basisConfiguration)
    basis = ACEBBasisSet()

    basis.initialize_basis(basisConfiguration)

    evaluator = ACEBEvaluator(basis)

    ace = ACECalculator(evaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert np.abs((ace.energy - 5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11


def test_constructor_BBasisConfiguration():
    bfunc = BBasisFunctionSpecification(["Al", "Al"], [1], [1.])
    print(bfunc)
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
    block.funcspecs = [bfunc]
    print(block)
    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]
    print(basisConfiguration)
    basis = ACEBBasisSet(basisConfiguration)

    evaluator = ACEBEvaluator(basis)

    ace = ACECalculator()
    ace.set_evaluator(evaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert np.abs((ace.energy - 5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11


def test_dimer_construction_basis():
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

    basis = ACEBBasisSet()
    basis.initialize_basis(basisConfiguration)
    evaluator = ACEBEvaluator(basis)

    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    print("Energy: ", ace.energy)
    forces = ace.forces
    print("forces: ", forces)

    assert np.abs((ace.energy - 5.681698418855005)) < 5e-10
    assert abs(forces[0][2] - (-0.6214426974245455)) < 1e-11
    assert abs(forces[1][2] - 0.6214426974245455) < 1e-11

    def get_energy(x):
        ase_atoms = Atoms(positions=[[0.0, 0.0, 0], [0, 0, x]], symbols=["W"] * 2)
        ae = pe.aseatoms_to_atomicenvironment(ase_atoms)
        ace.compute(ae, False)
        return ace.energy

    xs = np.linspace(1, 10, 10)
    ens = [get_energy(xx) for xx in xs]
    assert len(ens) == 10


def test_dimer_r1_initilize_bbasis_conv_to_ctilde_save_load():
    bfunc = BBasisFunctionSpecification(["Al", "Al"], [1], [0], [], [1.])
    block = BBasisFunctionsSpecificationBlock()
    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 0
    block.npoti = "FinnisSinclair"
    block.fs_parameters = [1, 1, ]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1

    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [[[1]]]
    block.funcspecs = [bfunc]
    print(block)
    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]
    print(basisConfiguration)
    basis = ACEBBasisSet()

    basis.initialize_basis(basisConfiguration)

    evaluator = ACEBEvaluator(basis)

    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    trimer = pe.create_linear_chain(3)
    ace.compute(trimer)

    assert np.abs((ace.energy - 5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11


def test_trimer_r234_dens2_initilize_basis():
    block = BBasisFunctionsSpecificationBlock()

    block.block_name = "Al"
    block.nradmaxi = 1
    block.lmaxi = 2
    block.nradbaseij = 1
    block.npoti = "FinnisSinclairShiftedScaled"
    block.fs_parameters = [1, 1, 1, 0.5]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"

    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [[[1], [1], [1]]]

    block.funcspecs = [

        BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[1, 1], LS=[], coeffs=[0.5, 0.5]),
        BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[1, 1, 0], LS=[0], coeffs=[0.5, 0.5]),
        BBasisFunctionSpecification(["Al", "Al", "Al", "Al", "Al"], ns=[1, 1, 1, 1], ls=[1, 1, 1, 1], LS=[2, 2],
                                    coeffs=[0.5, 0.5])
    ]

    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]

    basis = ACEBBasisSet()
    basis.initialize_basis(basisConfiguration)
    bevaluator = ACEBEvaluator(basis)

    ace = ACECalculator()
    ace.set_evaluator(bevaluator)
    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    print("Energy: ", ace.energy)
    forces = ace.forces
    print("forces: ", forces)

    assert np.abs((ace.energy - 19.562513088532768)) < 5e-10
    assert abs(forces[0][2] - (-7.752016046734244)) < 1e-11
    assert abs(forces[1][2] - (0)) < 1e-11
    assert abs(forces[2][2] - (7.752016046734244)) < 1e-11

    # TODO
    cevaluator = ACECTildeEvaluator()
    cbasis = basis.to_ACECTildeBasisSet()

    cevaluator.set_basis(cbasis)
    ace.set_evaluator(cevaluator)

    ace.compute(atoms)

    print("Energy: ", ace.energy)
    forces = ace.forces
    print("forces: ", forces)

    assert np.abs((ace.energy - 19.562513088532768)) < 5e-10
    assert abs(forces[0][2] - (-7.752016046734244)) < 1e-11
    assert abs(forces[1][2] - (0)) < 1e-11
    assert abs(forces[2][2] - (7.752016046734244)) < 1e-11


def test_chain_constructor():
    basisConfiguration = create_BBasisConfiguration()

    cbasis = ACEBBasisSet(basisConfiguration).to_ACECTildeBasisSet()
    ace = ACECalculator()

    cevaluator = ACECTildeEvaluator(cbasis)
    ace.set_evaluator(cevaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    print("Energy: ", ace.energy)
    forces = ace.forces
    print("forces: ", forces)

    assert np.abs((ace.energy - 19.562513088532768)) < 5e-10
    assert abs(forces[0][2] - (-7.752016046734244)) < 1e-11
    assert abs(forces[1][2] - (0)) < 1e-11
    assert abs(forces[2][2] - (7.752016046734244)) < 1e-11


def create_BBasisConfiguration(elm="Cu"):
    block = BBasisFunctionsSpecificationBlock()
    block.block_name = elm
    block.nradmaxi = 1
    block.lmaxi = 2
    block.npoti = "FinnisSinclairShiftedScaled"
    block.fs_parameters = [1, 1, 1, 0.5]
    block.rcutij = 8.7
    block.dcutij = 0.01
    block.NameOfCutoffFunctionij = "cos"
    block.nradbaseij = 1
    block.radbase = "ChebExpCos"
    block.radparameters = [3.0]
    block.radcoefficients = [[[1], [1], [1]]]
    block.funcspecs = [
        BBasisFunctionSpecification([elm] * 3, ns=[1, 1], ls=[1, 1], LS=[], coeffs=[0.5, 0.5]),
        BBasisFunctionSpecification([elm] * 4, ns=[1, 1, 1], ls=[1, 1, 0], LS=[0], coeffs=[0.5, 0.5]),
        BBasisFunctionSpecification([elm] * 5, ns=[1, 1, 1, 1], ls=[1, 1, 1, 1], LS=[2, 2],
                                    coeffs=[0.5, 0.5])
    ]
    basisConfiguration = BBasisConfiguration()
    basisConfiguration.deltaSplineBins = 0.001
    basisConfiguration.funcspecs_blocks = [block]
    basisConfiguration.validate(True)
    return basisConfiguration


def test_get_set_state_BBasisFunctionSpecification():
    spec1 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[1, 1, 0], coeffs=[0.5, 0.5])
    print(spec1)
    dump1 = pickle.dumps(spec1)
    spec2 = pickle.loads(dump1)
    print(spec2)

    assert spec1.elements == spec2.elements
    assert spec1.ns == spec2.ns
    assert spec1.ls == spec2.ls
    assert spec1.LS == spec2.LS
    assert spec1.coeffs == spec2.coeffs


def test_get_set_all_coeffs():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r234-2.yaml")

    all_coeffs = basis.all_coeffs
    print("all_coeffs before=", all_coeffs)
    assert all_coeffs == [1, 2, 3] + [2, 3, 4]

    basis.all_coeffs = [1, 1, 1] + [1, 1, 1]

    assert basis.all_coeffs == [1, 1, 1] + [1, 1, 1]
    print("all_coeffs after=", basis.all_coeffs)

    ace = ACECalculator()
    evaln = ACEBEvaluator(basis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert abs(ace.energy - 28.457404084390994) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 13.052474776412932) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11


def test_get_set_crad_basis_coeffs():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r234-2.yaml")
    crad_coeffs = basis.crad_coeffs
    basis_coeffs = basis.basis_coeffs
    assert crad_coeffs == [1, 2, 3]
    assert basis_coeffs == [2, 3, 4]

    basis.crad_coeffs = [1, 1, 1]
    basis.basis_coeffs = [1, 1, 1]
    all_coeffs = basis.all_coeffs
    print("all_coeffs after=", all_coeffs)
    ace = ACECalculator()

    evaln = ACEBEvaluator(basis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert abs(ace.energy - 28.457404084390994) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 13.052474776412932) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11


def test_to_BBasisConfiguration():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r1234l12_crad_dif.yaml")
    radfuncs = basis.radial_functions
    crad = np.array(radfuncs.crad)
    print("crad=", crad)

    bBasisConfig = basis.to_BBasisConfiguration()
    # print(bBasisConfig)
    block = bBasisConfig.funcspecs_blocks[0]
    print("block.radcoefficients=", block.radcoefficients)
    assert block.radcoefficients == [[[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]], [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]]]
    print("crad.shape=", crad.shape)
    crad = np.transpose(crad, (0, 1, 4, 3, 2))
    print("crad.T=", crad)


def test_BBasisConfiguration_load_save():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    block = bBasisConfig.funcspecs_blocks[0]
    print(" block.radcoefficients =", block.radcoefficients)
    assert block.radcoefficients == [[[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]], [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]]]

    bBasisConfig.save("tmp_config.yaml")
    bBasisConfig2 = BBasisConfiguration()
    bBasisConfig2.load("tmp_config.yaml")
    block2 = bBasisConfig2.funcspecs_blocks[0]
    assert block2.radcoefficients == [[[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]], [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]]]


def test_BBasisConfiguration_get_all_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_all_coeffs()
    print("coeffs=", coeffs)
    assert coeffs == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0, 1.0, 2.0, 3.0, 4.0]


def test_BBasisConfiguration_get_func_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_func_coeffs()
    print("coeffs=", coeffs)
    assert coeffs == [1.0, 2.0, 3.0, 4.0]


def test_BBasisConfiguration_get_radial_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_radial_coeffs()
    print("coeffs=", coeffs)
    assert coeffs == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0]


def test_BBasisConfiguration_set_func_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_func_coeffs()
    print("coeffs=", coeffs)
    new_coeffs = np.ones_like(coeffs) * 12
    bBasisConfig.set_func_coeffs(new_coeffs)
    coeffs2 = bBasisConfig.get_func_coeffs()
    print("coeffs2=", coeffs2)
    assert np.allclose(coeffs2, new_coeffs)

    acoeffs = bBasisConfig.get_all_coeffs()
    print("acoeffs=", acoeffs)
    assert acoeffs == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0,
                       12.0, 12.0, 12.0, 12.0]


def test_BBasisConfiguration_set_radial_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_radial_coeffs()
    print("coeffs=", coeffs)
    new_coeffs = np.ones_like(coeffs) * 12
    bBasisConfig.set_radial_coeffs(new_coeffs)
    coeffs2 = bBasisConfig.get_radial_coeffs()
    print("coeffs2=", coeffs2)
    assert np.allclose(coeffs2, new_coeffs)

    acoeffs = bBasisConfig.get_all_coeffs()
    print("acoeffs=", acoeffs)
    assert acoeffs == [12.0, 12, 12, 12, 12, 12, 12, 12, 12.0, 12.0, 12.0, 12.0,
                       1.0, 2.0, 3.0, 4.0]


def test_BBasisConfiguration_mult_get_set_radial_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/multispecies_AlNiCu.yaml")
    coeffs = bBasisConfig.get_radial_coeffs()
    print("coeffs=", coeffs)
    new_coeffs = np.ones_like(coeffs) * 12
    bBasisConfig.set_radial_coeffs(new_coeffs)
    coeffs2 = bBasisConfig.get_radial_coeffs()
    print("coeffs2=", coeffs2)
    assert np.allclose(coeffs2, new_coeffs)

    acoeffs = bBasisConfig.get_all_coeffs()
    print("acoeffs=", acoeffs)
    assert acoeffs == [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 2.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 1.0, 2.0, 1.0,
                       2.0, 1.0, 2.0, 1.0, 2.0, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0,
                       1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 12.0, 1.0, 2.0, 1.0,
                       2.0, 1.0, 2.0, 1.0, 2.0, 12.0, 12.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                       1.0, 2.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_BBasisConfiguration_mult_get_set_func_coeffs():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/multispecies_AlNiCu.yaml")
    coeffs = bBasisConfig.get_func_coeffs()
    print("coeffs=", coeffs)
    new_coeffs = np.ones_like(coeffs) * 12
    bBasisConfig.set_func_coeffs(new_coeffs)
    coeffs2 = bBasisConfig.get_func_coeffs()
    print("coeffs2=", coeffs2)
    assert np.allclose(coeffs2, new_coeffs)

    acoeffs = bBasisConfig.get_all_coeffs()
    print("acoeffs=", acoeffs)
    assert acoeffs == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 2.0, 3.0, 12.0, 12.0, 12.0,
                       12.0, 12.0, 12.0, 12.0, 12.0, 3.0, 2.0, 1.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 2.0, 3.0, 12.0,
                       12.0, 12.0, 12.0, 1.0, 1.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 2.0, 3.0, 12.0,
                       12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                       12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 12.0, 12.0,
                       12.0, 12.0, 12.0, 12.0]


def test_BBasisConfiguration_TEST_radial_function_name():
    bBasisConfig = BBasisConfiguration()
    bBasisConfig.load("tests/Al-Ni_test_radial_func_name.yaml")

    radbases = [block.radbase for block in bBasisConfig.funcspecs_blocks]
    print("radbases =", radbases)
    assert radbases == ['TEST_BesselFirst', 'TEST_BesselFirst', 'TEST_BesselSecond', 'TEST_BesselFirst']

    bBasisConfig.save("tmp_config.yaml")

    bbasisSet = ACEBBasisSet(bBasisConfig);
    print("bbasisSet=", bbasisSet)


def test_element_mapping():
    basis = ACEBBasisSet("tests/Al-r234.yaml")
    ace = ACECalculator()
    evaln = ACEBEvaluator(basis)
    evaln.element_type_mapping = [1, 0]
    assert evaln.element_type_mapping == [1, 0]

    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    atoms.species_type = [1, 1, 1]
    ace.compute(atoms)

    assert abs(ace.energy - 28.457404084390994) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] + 13.052474776412932) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11


def test_elements_name():
    basis = ACEBBasisSet("tests/Al-r234.yaml")
    print(basis.elements_name)
    assert basis.elements_name == ["Al"]


def test_metadata():
    config = BBasisConfiguration("tests/Al.pbe.13.2-metadata.yaml")
    metadata = config.metadata
    print("config.metadata = ", metadata)
    assert len(metadata) == 2

    basis = ACEBBasisSet(config)
    metadata = basis.metadata
    metadata["origin"] = "test"
    print("basis.metadata = ", metadata)

    assert len(metadata) == 3
    basis.save("test_metadata.yaml")

    basis2 = ACEBBasisSet("test_metadata.yaml")
    metadata = basis2.metadata
    print("basis2.metadata = ", metadata)
    assert len(metadata) == 3


def test_BBasisConfiguration_copy():
    bBasisConfig = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")
    coeffs = bBasisConfig.get_all_coeffs()
    assert coeffs == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0, 1.0, 2.0, 3.0, 4.0]

    copyBasis = bBasisConfig.copy()
    coeffs[0] = 10
    print("coeffs=", coeffs)
    bBasisConfig.set_all_coeffs(coeffs)

    coeffs_copy = copyBasis.get_all_coeffs()
    print("coeffs_copy=", coeffs_copy)
    assert coeffs_copy == [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0, 1.0, 2.0, 3.0, 4.0]


def test_BBasisConfiguration_add():
    bBasisConfig1 = create_BBasisConfiguration(elm="Cu")
    bBasisConfig2 = create_BBasisConfiguration(elm="Al")
    assert len(bBasisConfig1.funcspecs_blocks) == 1
    assert len(bBasisConfig2.funcspecs_blocks) == 1

    assert bBasisConfig1.funcspecs_blocks[0].block_name == "Cu"
    assert bBasisConfig2.funcspecs_blocks[0].block_name == "Al"

    newBBasisConfig = bBasisConfig1 + bBasisConfig2

    assert len(newBBasisConfig.funcspecs_blocks) == 2
    assert newBBasisConfig.funcspecs_blocks[0].block_name == "Cu"
    assert newBBasisConfig.funcspecs_blocks[1].block_name == "Al"

    newBBasisConfig.validate(True)


def test_BBasisConfiguration_iadd():
    bBasisConfig1 = create_BBasisConfiguration(elm="Cu")
    bBasisConfig2 = create_BBasisConfiguration(elm="Al")
    assert len(bBasisConfig1.funcspecs_blocks) == 1
    assert len(bBasisConfig2.funcspecs_blocks) == 1

    assert bBasisConfig1.funcspecs_blocks[0].block_name == "Cu"
    assert bBasisConfig2.funcspecs_blocks[0].block_name == "Al"

    bBasisConfig1 += bBasisConfig2

    assert len(bBasisConfig1.funcspecs_blocks) == 2
    assert bBasisConfig1.funcspecs_blocks[0].block_name == "Cu"
    assert bBasisConfig1.funcspecs_blocks[1].block_name == "Al"


def test_BBasisConfiguration_save_load_another_element():
    bBasisConfig = create_BBasisConfiguration()
    block = bBasisConfig.funcspecs_blocks[0]
    funcspecs = block.funcspecs
    assert funcspecs[0].elements == ["Cu", "Cu", "Cu"]

    new_pot_name = "TEST_Cu_potential.yaml"
    bBasisConfig.save(new_pot_name)
    newBBasisConfig = BBasisConfiguration(new_pot_name)
    newblock = newBBasisConfig.funcspecs_blocks[0]
    assert newblock.mu0 == "Cu"
    newfuncspecs = newblock.funcspecs
    assert newfuncspecs[0].elements == ["Cu", "Cu", "Cu"]


def test_BBasisConfiguration_save_load_core_rep():
    bBasisConfig = create_BBasisConfiguration()
    block = bBasisConfig.funcspecs_blocks[0]

    block.core_rep_parameters = [200, 100]
    block.rho_cut = 20
    block.drho_cut = 2

    assert block.core_rep_parameters == [200, 100]
    assert block.rho_cut == 20
    assert block.drho_cut == 2

    new_pot_name = "TEST_Cu_potential.yaml"
    bBasisConfig.save(new_pot_name)
    newBBasisConfig = BBasisConfiguration(new_pot_name)
    newblock = newBBasisConfig.funcspecs_blocks[0]

    assert newblock.core_rep_parameters == [200, 100]
    assert newblock.rho_cut == 20
    assert newblock.drho_cut == 2


def test_BBasisConfiguration_total_number_of_functions():
    bBasisConfig = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")
    total = bBasisConfig.total_number_of_functions
    print("total=", total)
    assert total == 4


def test_BBasisFunctionSpecification_eq():
    func1 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[1, 1, 0], coeffs=[0.5, 0.5])
    func2 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[1, 1, 0], coeffs=[0.5, 0.5])
    func3 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[1, 1, 0], coeffs=[0.5, 0.6])
    func4 = BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[1, 1], coeffs=[0.5, 0.5])

    assert func1 == func2
    assert func1 != func3
    assert func3 != func4


def test_BBasisFunctionsSpecificationBlock_set_get_coeff():
    bBasisConfig = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")
    all_coeffs = np.array(bBasisConfig.get_all_coeffs())
    all_coeffs_ref = [1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12., 1., 2., 3., 4.]
    print("all_coeffs=", all_coeffs)
    assert np.allclose(all_coeffs, all_coeffs_ref)
    np.random.seed(42)
    new_coeffs = all_coeffs + np.random.rand(*np.shape(all_coeffs))

    bBasisConfig.set_all_coeffs(new_coeffs)

    new_res_coeffs = np.array(bBasisConfig.get_all_coeffs())

    assert np.allclose(new_coeffs, new_res_coeffs)


def test_pickle_BBasisFunctionsSpecificationBlock():
    bBasisConfig = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")

    block = bBasisConfig.funcspecs_blocks[0]

    pickled_block = pickle.dumps(block)

    unpickled_block = pickle.loads(pickled_block)

    assert block.block_name == unpickled_block.block_name
    assert block.rankmax == unpickled_block.rankmax
    assert block.number_of_species == unpickled_block.number_of_species
    assert block.elements_vec == unpickled_block.elements_vec
    assert block.mu0 == unpickled_block.mu0
    assert block.lmaxi == unpickled_block.lmaxi
    assert block.nradmaxi == unpickled_block.nradmaxi
    assert block.ndensityi == unpickled_block.ndensityi
    assert block.npoti == unpickled_block.npoti
    assert block.fs_parameters == unpickled_block.fs_parameters
    assert block.core_rep_parameters == unpickled_block.core_rep_parameters
    assert block.rho_cut == unpickled_block.rho_cut
    assert block.drho_cut == unpickled_block.drho_cut
    assert block.rcutij == unpickled_block.rcutij
    assert block.dcutij == unpickled_block.dcutij
    assert block.NameOfCutoffFunctionij == unpickled_block.NameOfCutoffFunctionij
    assert block.nradbaseij == unpickled_block.nradbaseij
    assert block.radbase == unpickled_block.radbase
    assert block.radparameters == unpickled_block.radparameters
    assert block.funcspecs == unpickled_block.funcspecs

    assert block.radcoefficients == unpickled_block.radcoefficients


def test_pickle_BBasisConfiguration():
    config = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")
    config.metadata["key"] = "value"
    config.auxdata.int_data = {"a": 1, "b": 2}
    config.auxdata.int_arr_data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    config.auxdata.double_data = {"a": 1.234, "b": 4.234}
    config.auxdata.double_arr_data = {"a": [1.1, 2.2, 3.3], "b": [4.4, 5.5, 6.6]}
    config.auxdata.string_data["a"] = "1.234"
    config.auxdata.string_data["b"] = "4.234"
    config.auxdata.string_arr_data = {"a": ["1.1", "2.2", "3.3"], "b": ["4.4", "5.5", "6.6"]}

    print(config.auxdata.int_data)
    print(config.auxdata.int_arr_data)
    print(config.auxdata.double_data)
    print(config.auxdata.double_arr_data)
    print(config.auxdata.string_data)
    print(config.auxdata.string_arr_data)
    block = config.funcspecs_blocks[0]

    pickled_config = pickle.dumps(config)
    unpickled_config = pickle.loads(pickled_config)

    unpickled_block = unpickled_config.funcspecs_blocks[0]

    assert config.deltaSplineBins == unpickled_config.deltaSplineBins
    assert config.metadata['key'] == unpickled_config.metadata['key']

    assert config.auxdata.int_data == unpickled_config.auxdata.int_data
    assert config.auxdata.double_data == unpickled_config.auxdata.double_data
    assert config.auxdata.string_data["a"] == unpickled_config.auxdata.string_data["a"]
    assert config.auxdata.string_data["b"] == unpickled_config.auxdata.string_data["b"]

    assert config.auxdata.int_arr_data == unpickled_config.auxdata.int_arr_data
    assert config.auxdata.double_arr_data == unpickled_config.auxdata.double_arr_data
    assert config.auxdata.string_arr_data == unpickled_config.auxdata.string_arr_data

    assert block.block_name == unpickled_block.block_name
    assert block.rankmax == unpickled_block.rankmax
    assert block.number_of_species == unpickled_block.number_of_species
    assert block.elements_vec == unpickled_block.elements_vec
    assert block.mu0 == unpickled_block.mu0
    assert block.lmaxi == unpickled_block.lmaxi
    assert block.nradmaxi == unpickled_block.nradmaxi
    assert block.ndensityi == unpickled_block.ndensityi
    assert block.npoti == unpickled_block.npoti
    assert block.fs_parameters == unpickled_block.fs_parameters
    assert block.core_rep_parameters == unpickled_block.core_rep_parameters
    assert block.rho_cut == unpickled_block.rho_cut
    assert block.drho_cut == unpickled_block.drho_cut
    assert block.rcutij == unpickled_block.rcutij
    assert block.dcutij == unpickled_block.dcutij
    assert block.NameOfCutoffFunctionij == unpickled_block.NameOfCutoffFunctionij
    assert block.nradbaseij == unpickled_block.nradbaseij
    assert block.radbase == unpickled_block.radbase
    assert block.radparameters == unpickled_block.radparameters
    assert block.funcspecs == unpickled_block.funcspecs

    assert block.radcoefficients == unpickled_block.radcoefficients


def test_BBasisConfiguration_auxdata():
    config = BBasisConfiguration("tests/Cu-aux_data.yaml")
    print(config.auxdata.int_data)
    print(config.auxdata.int_arr_data)
    print(config.auxdata.double_data)
    print(config.auxdata.double_arr_data)
    print(config.auxdata.string_data)
    print(config.auxdata.string_arr_data)

    auxdata = config.auxdata
    assert auxdata.int_data == {"i1": 1, "i2": 2}
    assert auxdata.double_data == {"d1": 1.5, "d2": 2.0}
    assert auxdata.string_data["s1"] == "Some string"
    assert auxdata.string_data["s2"] == "Some another string"

    assert auxdata.int_arr_data == {"ia1": [1, 2, 3], "ia2": [100, -100, 0]}
    assert auxdata.double_arr_data == {"da1": [1.1, 2.2, 3.3], "da2": [100.5, -100.1, 0.0]}
    assert auxdata.string_arr_data == {"sa1": ["1.1", "2.2", "3.3"], "sa2": ["100.5", "-100.1", "0.0"]}

    config.auxdata.int_data = {"i1": 1, "i2": 2, "i3": 3}
    assert config.auxdata.int_data == {"i1": 1, "i2": 2, "i3": 3}

    basis = ACEBBasisSet(config)

    auxdata = basis.auxdata
    assert auxdata.int_data == {"i1": 1, "i2": 2, "i3":3}
    assert auxdata.double_data == {"d1": 1.5, "d2": 2.0}
    assert auxdata.string_data["s1"] == "Some string"
    assert auxdata.string_data["s2"] == "Some another string"

    assert auxdata.int_arr_data == {"ia1": [1, 2, 3], "ia2": [100, -100, 0]}
    assert auxdata.double_arr_data == {"da1": [1.1, 2.2, 3.3], "da2": [100.5, -100.1, 0.0]}
    assert auxdata.string_arr_data == {"sa1": ["1.1", "2.2", "3.3"], "sa2": ["100.5", "-100.1", "0.0"]}

    new_conf = basis.to_BBasisConfiguration()
    auxdata = new_conf.auxdata
    assert auxdata.int_data == {"i1": 1, "i2": 2, "i3":3}
    assert auxdata.double_data == {"d1": 1.5, "d2": 2.0}
    assert auxdata.string_data["s1"] == "Some string"
    assert auxdata.string_data["s2"] == "Some another string"

    assert auxdata.int_arr_data == {"ia1": [1, 2, 3], "ia2": [100, -100, 0]}
    assert auxdata.double_arr_data == {"da1": [1.1, 2.2, 3.3], "da2": [100.5, -100.1, 0.0]}
    assert auxdata.string_arr_data == {"sa1": ["1.1", "2.2", "3.3"], "sa2": ["100.5", "-100.1", "0.0"]}

def test_pickle_ACEBBasisSet():
    config = BBasisConfiguration("tests/Al-r1234l12_crad_dif.yaml")
    bbasis_set = ACEBBasisSet(config)

    ace = ACECalculator()
    evaln = ACEBEvaluator(bbasis_set)

    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    E_before = ace.energy
    f_before = np.array(ace.forces)

    pickled_bbasis_set = pickle.dumps(bbasis_set)
    unpickled_bbasis_set = pickle.loads(pickled_bbasis_set)

    assert bbasis_set.all_coeffs == unpickled_bbasis_set.all_coeffs

    evaln = ACEBEvaluator(unpickled_bbasis_set)
    ace.set_evaluator(evaln)
    ace.compute(atoms)

    E_after = ace.energy
    f_after = np.array(ace.forces)

    assert abs(E_after - E_before) < 5e-10
    assert np.allclose(f_before, f_after)


def test_multispecies_load_ACEBBasisSet():
    bBasis = ACEBBasisSet("tests/multispecies_AlNiCu.yaml")
    print("bBasis=", bBasis)


def test_multispecies_ACEBBasisSet_to_ACECTildeBasisSet():
    bBasis = ACEBBasisSet("tests/multispecies_AlNiCu.yaml")
    print("bBasis=", bBasis)
    ctilde = bBasis.to_ACECTildeBasisSet()


def test_multispecies_to_ACECTildeBasisSet_save_load_yaml():
    bBasis = ACEBBasisSet("tests/multispecies_AlNiCu.yaml")
    print("bBasis=", bBasis)
    ctilde = bBasis.to_ACECTildeBasisSet()
    ctilde.save_yaml("multispecies_AlNiCu.yace")
    ctilde.load_yaml("multispecies_AlNiCu.yace")


def test_multispecies_ACEBBasisSet_all_coeffs_mask():
    bBasis = ACEBBasisSet("tests/multispecies_AlNiCu.yaml")
    all_coeffs_mask = bBasis.all_coeffs_mask
    all_coeffs = bBasis.all_coeffs

    crad_coeffs_mask = bBasis.crad_coeffs_mask
    crad_coeffs = bBasis.crad_coeffs

    basis_coeffs_mask = bBasis.basis_coeffs_mask
    basis_coeffs = bBasis.basis_coeffs

    print("bBasis.all_coeffs=", all_coeffs)
    print("bBasis.crad_coeffs_mask=", crad_coeffs_mask)
    print("bBasis.basis_coeffs_mask=", basis_coeffs_mask)
    print("bBasis.all_coeffs_mask=", all_coeffs_mask)
    assert len(crad_coeffs) == len(crad_coeffs_mask)
    assert len(basis_coeffs) == len(basis_coeffs_mask)
    assert len(all_coeffs) == len(all_coeffs_mask)
    assert crad_coeffs_mask + basis_coeffs_mask == all_coeffs_mask

    assert crad_coeffs_mask == [[0], [0], [0], [0], [0], [0], [0, 1], [0, 1], [0, 2], [0, 2], [0, 2], [0, 2], [1], [1],
                                [1], [1, 2], [1, 2], [2], [2], [2]]

    assert basis_coeffs_mask == [[0], [0], [0, 1], [0, 2], [0], [0, 1], [0, 2], [0, 1, 2], [0], [0, 1], [0, 1, 2],
                                 [0, 2], [0], [0, 1], [0, 1, 2], [0, 2], [1, 0], [1], [1, 2], [1, 0], [1, 0, 2], [1],
                                 [1, 2], [1, 0], [1, 0, 2], [1], [1, 2], [1, 0], [1], [1, 2], [2, 0], [2, 0], [2, 1],
                                 [2, 1], [2], [2], [2, 0, 1], [2, 0, 1], [2, 0], [2, 0], [2, 1], [2, 1], [2], [2],
                                 [2, 0, 1], [2, 0, 1], [2, 0], [2, 0], [2, 1], [2, 1], [2], [2], [2, 0, 1], [2, 0, 1],
                                 [2, 0], [2, 0], [2, 1], [2, 1], [2], [2]]

    assert all_coeffs_mask == [[0], [0], [0], [0], [0], [0], [0, 1], [0, 1], [0, 2], [0, 2], [0, 2], [0, 2], [1], [1],
                               [1], [1, 2], [1, 2], [2], [2], [2], [0], [0], [0, 1], [0, 2], [0], [0, 1], [0, 2],
                               [0, 1, 2], [0], [0, 1], [0, 1, 2], [0, 2], [0], [0, 1], [0, 1, 2], [0, 2], [1, 0], [1],
                               [1, 2], [1, 0], [1, 0, 2], [1], [1, 2], [1, 0], [1, 0, 2], [1], [1, 2], [1, 0], [1],
                               [1, 2], [2, 0], [2, 0], [2, 1], [2, 1], [2], [2], [2, 0, 1], [2, 0, 1], [2, 0], [2, 0],
                               [2, 1], [2, 1], [2], [2], [2, 0, 1], [2, 0, 1], [2, 0], [2, 0], [2, 1], [2, 1], [2], [2],
                               [2, 0, 1], [2, 0, 1], [2, 0], [2, 0], [2, 1], [2, 1], [2], [2]]

from sys import stdout

import pytest
import pickle

import numpy as np
from pyace.evaluator import ACECTildeEvaluator
import pyace.atomicenvironment as pe
from pyace.calculator import ACECalculator
from pyace.basis import (ACECTildeBasisSet,
                         ACEBBasisSet,
                         ACECTildeBasisFunction,
                         BBasisConfiguration,
                         BBasisFunctionSpecification,
                         BBasisFunctionsSpecificationBlock)

from ase import Atoms

def test_dimer_r1_energy_forces():
    basis = ACEBBasisSet("tests/Al-r1l0.yaml")
    ace = ACECalculator()

    evaln = ACECTildeEvaluator()
    cbasis = basis.to_ACECTildeBasisSet()
    evaln.set_basis(cbasis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    assert (ace.energy-1.9355078359256011) < 5e-10
    forces = ace.forces
    assert (forces[0][2] + 0.12757969575334369) < 1e-11
    assert (forces[1][2] - 0.12757969575334369) < 1e-11

def test_trimer_r234_energy_forces():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r234.yaml")
    ace = ACECalculator()

    evaln = ACECTildeEvaluator()
    cbasis = basis.to_ACECTildeBasisSet()
    evaln.set_basis(cbasis)
    ace.set_evaluator(evaln)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert (ace.energy-28.457404084390994) < 5e-10
    forces = ace.forces
    assert (forces[0][2] + 13.052474776412932) < 1e-11
    assert (forces[1][2] - 0.0000000000000000) < 1e-11

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
    basis = ACEBBasisSet()

    basis.initialize_basis(basisConfiguration)
    cbasis = basis.to_ACECTildeBasisSet()
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace = ACECalculator()
    ace.set_evaluator(evaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert np.abs((ace.energy-5.6213654940076045)) < 5e-10
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
    basisConfiguration.funcspecs_blocks = [block]
    basisConfiguration.deltaSplineBins = 0.001
    print(basisConfiguration)
    basis = ACEBBasisSet(basisConfiguration)
    cbasis = basis.to_ACECTildeBasisSet()
    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)
    ace = ACECalculator()
    ace.set_evaluator(evaluator)

    atoms = pe.create_linear_chain(3)
    ace.compute(atoms)

    assert np.abs((ace.energy-5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11


def test_BBasisFunctionSpecification_autocompletion():
    bfuncspec=BBasisFunctionSpecification(["Al","Al"], ns=[1], coeffs=[1])
    print(bfuncspec)

    bfuncspec=BBasisFunctionSpecification(["Al","Al","Al"], ns=[6,6], ls=[5], coeffs=[1]) # ls==[5,5]
    print(bfuncspec)

    ns=[5,5,5]
    ls=[4,3,1]
    bfuncspec=BBasisFunctionSpecification(["Al","Al","Al","Al"], ns=ns, ls=ls,  coeffs=[1]) #LS=[ls[-1]]
    print(bfuncspec)

    ns=[3,3,3,1]
    ls=[2,2,2,0]
    bfuncspec=BBasisFunctionSpecification(["Al","Al","Al","Al","Al"], ns=ns, ls=ls, LS=[2], coeffs=[1]) # LS[-2]==LS[-1]
    print(bfuncspec)

    ns=[3,3,3,1,3]
    ls=[2,2,2,2,0]
    bfuncspec=BBasisFunctionSpecification(["Al","Al","Al","Al","Al","Al"], ns=ns, ls=ls, LS=[2,2], coeffs=[1]) # //L(-1) = l(-1)
    # LS==[2,2, 0]
    print(bfuncspec)

def test_BBasisFunctionSpecification_exceptions():

    with pytest.raises(ValueError) as exc:
        b = BBasisFunctionSpecification(["Al","Al"], ns=[0], ls=[0], coeffs=[1.,2])
    print(str(exc.value))

    # with pytest.raises(ValueError) as exc:
    #     b = BBasisFunctionSpecification(["Al","Al"], ns=[1], ls=[-1], coeffs=[1.,2])
    # print(str(exc.value))

    with pytest.raises(ValueError) as exc:
        b = BBasisFunctionSpecification(["Al","Al","Al"], ns=[1,1], ls=[0,0], LS=[1], coeffs=[1.,2])
    print(str(exc.value))

    with pytest.raises(ValueError) as exc:
        BBasisFunctionSpecification(["Al","Al","Al"], ns=[1,1], ls=[0,0], LS=[1], coeffs=[1.,2])
    print(str(exc.value))

    with pytest.raises(ValueError):
        b = BBasisFunctionSpecification(["Al","Al"], ns=[1,1], ls=[0,0], LS=[1], coeffs=[1.,2])
    print(str(exc.value))

    with pytest.raises(ValueError):
        b = BBasisFunctionSpecification(["Al","Al","Al"], ns=[1,1], ls=[0], LS=[1], coeffs=[1.,2])
    print(str(exc.value))


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
    cbasis = basis.to_ACECTildeBasisSet()

    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)

    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    atoms = pe.create_linear_chain(2)
    ace.compute(atoms)

    print("Energy: ",ace.energy)
    forces=ace.forces
    print("forces: ", forces)

    assert np.abs((ace.energy-5.681698418855005)) < 5e-10
    assert abs(forces[0][2] - (-0.6214426974245455)) < 1e-11
    assert abs(forces[1][2] - 0.6214426974245455) < 1e-11
    def get_energy(x):
        ase_atoms = Atoms(positions=[[0.0, 0.0, 0], [0,0,x]], symbols=["W"]*2)
        ae = pe.aseatoms_to_atomicenvironment(ase_atoms)
        ace.compute(ae, False)
        return ace.energy
    xs = np.linspace(1,10,10)
    ens = [get_energy(xx) for xx in xs]
    assert len(ens)==10


def test_dimer_r1_initilize_bbasis_conv_to_ctilde_save_load():
    bfunc = BBasisFunctionSpecification(["Al", "Al"], [1], [0], [], [1.])
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
    basis = ACEBBasisSet()

    basis.initialize_basis(basisConfiguration)
    cbasis = basis.to_ACECTildeBasisSet()

    evaluator = ACECTildeEvaluator()
    evaluator.set_basis(cbasis)

    ace = ACECalculator()
    ace.set_evaluator(evaluator)
    trimer = pe.create_linear_chain(3)
    ace.compute(trimer)

    assert np.abs((ace.energy-5.6213654940076045)) < 5e-10
    forces = ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11

    cbasis.save("test_cbasis.ace")
    new_cbasis = ACECTildeBasisSet()
    new_cbasis.load("test_cbasis.ace")

    new_evaluator = ACECTildeEvaluator()
    new_evaluator.set_basis(new_cbasis)

    new_ace = ACECalculator()
    new_ace.set_evaluator(new_evaluator)

    new_ace.compute(trimer)

    assert np.abs((new_ace.energy-5.6213654940076045)) < 5e-10
    forces = new_ace.forces
    assert abs(forces[0][2] - (-0.36628330591785796)) < 1e-11
    assert abs(forces[1][2] - 0.0000000000000000) < 1e-11
    assert abs(forces[2][2] - 0.36628330591785796) < 1e-11

def test_ACECTildeBasisFunction_init():
    func = ACECTildeBasisFunction()

    assert func.ms_combs==[]
    assert func.ctildes==[]
    assert func.mus==[]
    assert func.ns==[]
    assert func.ls==[]
    assert func.num_ms_combs==0
    assert func.rank==0
    assert func.ndensity==0
    assert func.mu0==0
    assert func.is_half_ms_basis==False

def test_ACECTildeBasisFunction_getstate_emptyfunc():
    func = ACECTildeBasisFunction()
    print(func.__getstate__())
    assert func.__getstate__() == (0, 0, 0, [], [], [], [], [], False)

# def test_ACECTildeBasisFunction_setstate():
#     func = ACECTildeBasisFunction()
#     state = (2, 3, 0, [0,0], [1,1], [0,0], [[0,0],[1,-1]], [[1,2],[3,4]], True)
#     func.__setstate__(state)
#     print(func.__getstate__())
#     assert func.__getstate__() == state

# def test_ACECTildeBasisFunction_pickle_unpickle():
#     assert False

def test_ACECTildeBasisset_basis_rank_func():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r1l0.yaml")
    cbasis = basis.to_ACECTildeBasisSet()
    basis_rank1 = cbasis.basis_rank1
    print("basis_rank1=", basis_rank1)
    func = basis_rank1[0][0]
    print("func=",func)
    print("func state = ",func.__getstate__())

    assert len(cbasis.basis_rank1)==1
    assert len(cbasis.basis_rank1[0])==1

    assert func.ms_combs==[[0]]
    assert func.ctildes==[[1]]
    assert func.mus==[0]
    assert func.ns==[1]
    assert func.ls==[0]
    assert func.num_ms_combs==1
    assert func.rank==1
    assert func.ndensity==1
    assert func.mu0==0
    assert func.is_half_ms_basis==True
    assert func.is_proxy==False

    dumps = pickle.dumps(func)
    new_func = pickle.loads(dumps)

    assert new_func.ms_combs==func.ms_combs
    assert new_func.ctildes==func.ctildes
    assert new_func.mus==func.mus
    assert new_func.ns==func.ns
    assert new_func.ls==func.ls
    assert new_func.num_ms_combs==func.num_ms_combs
    assert new_func.rank==func.rank
    assert new_func.ndensity==func.ndensity
    assert new_func.mu0==func.mu0
    assert new_func.is_half_ms_basis==func.is_half_ms_basis
    assert new_func.is_proxy==func.is_proxy
    # assert func==new_func

    func.print()
    new_func.print()

def test_ACECTildeBasisset__getstate():
    basis = ACEBBasisSet()
    basis.load("tests/Al-r234.yaml")
    cbasis = basis.to_ACECTildeBasisSet()
    print(cbasis.__getstate__())



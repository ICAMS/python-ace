try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

import numpy as np

from pyace.basis import BBasisFunctionSpecification, ACERadialFunctions, ACEBBasisSet


def test_ACERadialFunctions_simple():
    basis = ACEBBasisSet("tests/Al-r1l0.yaml")
    acerad = basis.radial_functions
    print(acerad)

    print(acerad.nelements)
    print(acerad.lmax)
    print(acerad.nradial)
    print(acerad.nradbase)
    # print(acerad.cutoff)
    print(acerad.deltaSplineBins)

    print(acerad.crad)

    acerad.radfunc(0, 0)
    assert acerad.crad == [[[[[1.]]]]]
    assert acerad.nelements == 1
    assert acerad.lmax == 0
    assert acerad.nradial == 1
    assert acerad.nradbase == 1
    # assert acerad.cutoff == 8.7
    assert acerad.deltaSplineBins == 0.001


def test_ACERadialFunctions_Al_pbe_13_2():
    basis = ACEBBasisSet("tests/Al.pbe.13.2.yaml")
    acerad = basis.radial_functions
    print(acerad)

    print(acerad.nelements)
    print(acerad.lmax)
    print(acerad.nradial)
    print(acerad.nradbase)
    # print(acerad.cutoff)
    print(acerad.deltaSplineBins)

    print(acerad.crad)

    acerad.radfunc(0, 0)
    assert acerad.crad[0][0][0][0][0] == 0.9038675001065195
    assert acerad.nelements == 1
    assert acerad.lmax == 4
    assert acerad.nradial == 4
    assert acerad.nradbase == 10
    # assert acerad.cutoff == 8.7
    assert acerad.deltaSplineBins == 0.001


def test_ACERadialFunctions_compute_values():
    basis = ACEBBasisSet("tests/Al.pbe.13.2.yaml")
    acerad = basis.radial_functions
    print(acerad)
    cut = acerad.cut
    dcut = acerad.dcut
    lambd = acerad.lamb

    acerad.radbase(lambd[0][0], cut[0][0], dcut[0][0], "ChebExpCos", 3., 0, 0)
    print("gr=", acerad.gr)
    print("dgr=", acerad.dgr)

    assert abs(acerad.gr[0] - 0.7342042203498951) < 1e-7
    assert abs(acerad.dgr[0] - (-0.1595192498959612)) < 1e-7

    acerad.radfunc(0, 0)
    fr = np.array(acerad.fr)
    dfr = np.array(acerad.dfr)
    print("fr.shape=", fr.shape)
    print("dfr.shape=", dfr.shape)
    print("fr=", acerad.fr)
    print("dfr=", acerad.dfr)
    assert fr.shape == (4, 5)
    assert dfr.shape == (4, 5)

    assert abs(fr[0][0] - (-1.668699473670447)) < 1e-7
    assert abs(dfr[0][0] - (-0.8483823930963537)) < 1e-7

    acerad.evaluate(3, acerad.nradbase, acerad.nradial, 0, 0)

    fr = np.array(acerad.fr)
    dfr = np.array(acerad.dfr)
    print("fr.shape=", fr.shape)
    print("dfr.shape=", dfr.shape)
    print("fr=", acerad.fr)
    print("dfr=", acerad.dfr)
    assert fr.shape == (4, 5)
    assert dfr.shape == (4, 5)

    assert abs(fr[0][0] - (-1.668699473670447)) < 1e-7
    assert abs(dfr[0][0] - (-0.8483823930963537)) < 1e-7


def test_ACERadialFunctions_pickle():
    basis = ACEBBasisSet("tests/Al-r1l0.yaml")
    acerad0 = basis.radial_functions

    p = pickle.dumps(acerad0)
    acerad = pickle.loads(p)

    acerad.radfunc(0, 0)
    assert acerad.crad == [[[[[1.]]]]]
    assert acerad.nelements == 1
    assert acerad.lmax == 0
    assert acerad.nradial == 1
    assert acerad.nradbase == 1
    assert acerad.deltaSplineBins == 0.001
    assert acerad.radbasenameij == [["ChebExpCos"]]


def test_ACERadialFunctions_evaluate_range():
    basis = ACEBBasisSet("tests/Al.pbe.13.2.yaml")
    acerad = basis.radial_functions
    acerad.evaluate(1, acerad.nradbase, acerad.nradial, 0, 0)
    gr = acerad.gr
    acerad.evaluate_range([1, 2, 3], acerad.nradbase, acerad.nradial, 0, 0)
    gr_vec = acerad.gr_vec
    assert len(gr_vec) == 3
    assert np.all(gr_vec[0] == gr)

import numpy as np

from pyace import ACEBBasisSet
from pyace.radial import *


def test_RadialFunctionsValues():
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif.yaml")
    radVal = RadialFunctionsValues(bbasis)
    print(radVal)
    print(radVal.npoints)
    print(radVal.cutoff)
    print(radVal.dx)
    assert radVal.npoints == 8700
    assert radVal.cutoff == 8.7
    assert radVal.dx == 0.001


def test_RadialFunctionsValues_rl0():
    bbasis = ACEBBasisSet("tests/Al-r1l0.yaml")
    radVal = RadialFunctionsValues(bbasis)
    print(radVal)
    print(radVal.npoints)
    print(radVal.cutoff)
    print(radVal.dx)
    assert radVal.npoints == 8700
    assert radVal.cutoff == 8.7
    assert radVal.dx == 0.001


def test_RadialFunctionSmoothness():
    bbasis = ACEBBasisSet("tests/Al-r1234l12_crad_dif.yaml")
    radVal = RadialFunctionsValues(bbasis)
    smothness = RadialFunctionSmoothness(radVal)
    smooth_quad = smothness.smooth_quad
    loss_radial = 1. * smooth_quad[0] + 2. * smooth_quad[1] + 1. * smooth_quad[2]
    print(smothness)
    print(loss_radial)
    assert np.allclose(loss_radial, 45.78851338482764)


def test_RadialFunctionSmoothness_multispecies():
    bbasis = ACEBBasisSet("tests/multispecies_AlNiCu.yaml")
    radVal = RadialFunctionsValues(bbasis)
    smothness = RadialFunctionSmoothness(radVal)
    smooth_quad = smothness.smooth_quad
    loss_radial = 1. * smooth_quad[0] + 2. * smooth_quad[1] + 1. * smooth_quad[2]
    print("smothness=", smothness)
    print("loss_radial=", loss_radial)
    assert np.allclose(loss_radial, 4.839803345705426)

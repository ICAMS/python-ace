import pytest
import os
import pyace.sharmonics as pp
import numpy as np

def test_sanity():
    sh = pp.PyACESHarmonics(3)
    sh.compute_plm(0.33, 0.33)
    plm = sh.plm
    assert np.round(plm[0][0], decimals=2) == 0.4

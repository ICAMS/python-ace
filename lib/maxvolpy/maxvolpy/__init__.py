"""
Module `maxvolpy` is designed for constructing different low-rank
skeleton and cross approximations.

Right now, cross approximation techniques are not implemented yet, but
all kinds of algorithms of finding good submatrices to build skeleton
approximations are presented in `maxvol` submodule. What does good
submatrix mean is noted in documentation for `maxvol` submodule.
"""

from __future__ import absolute_import

__all__ = ['maxvol']

from . import maxvol
from .__version__ import __version__

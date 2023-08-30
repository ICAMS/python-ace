#!/usr/bin/env python
"""
Cross and skeleton approximations, based on maximum volume submatrices.
"""

DOCLINES = (__doc__ or '').split("\n")

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import platform

extensions = [
    Extension("maxvolpy._maxvol",
              ["maxvolpy/_maxvol.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-O3', '-march=native', '-ffast-math']
              )
]

ext_modules = None
if platform.system() != "Windows":
    ext_modules = cythonize(extensions)

setup(
    name='maxvolpy',
    version='0.3.8',
    maintainer="Alexander Mikhalev",
    maintainer_email="muxasizhevsk@gmail.com",
    description=DOCLINES[1],
    long_description=DOCLINES[1],
    url="https://bitbucket.org/muxas/maxvolpy",
    author="Alexander Mikhalev",
    author_email="muxasizhevsk@gmail.com",
    license='MIT',
    install_requires=['numpy>=1.10.1', 'scipy>=0.16.0', 'cython>=0.23.4'],
    packages=['maxvolpy'],
    ext_modules=ext_modules
)

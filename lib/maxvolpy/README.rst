About `maxvolpy`
================

Module `maxvolpy` is designed for constructing different low-rank
skeleton and cross approximations. There are different strategies of
doing this, but this module is focused on approximations, based on
submatrices of good volume.

Installation
============

Module `maxvolpy` can be installed in 2 ways:
 - Pip-installer
 - Installation from git repository

Requirements
------------

Package extensively uses `numpy` python package and `blas`/`lapack`
libraries.
Package `numpy` must be of version at least 1.10

Pip-installer
-------------

Simply run `pip` with necessary arguments:

.. code:: shell

    pip install maxvolpy

Installation from git repository
--------------------------------

To compile and install package from repository, you should have
`cython` and `numpy` python packages installed.

.. code:: shell

    git clone https://bitbucket.org/muxas/maxvolpy
    cd maxvolpy
    python setup.py install

Additional Information
----------------------

If module was installed from git repository, but was not properly
compiled during installation process due to any errors, it will still
work if module `scipy` is presented in system. Module has backup
functions, written in Python.

Documentation
=============

Is available at http://pythonhosted.org/maxvolpy/
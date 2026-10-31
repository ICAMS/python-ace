# Installation

Create a separate environment; for example:

```bash
mamba create -n ace python=3.12 pip
mamba activate ace
```

You can use `conda` instead of `mamba`.

## Install pacemaker

```bash
pip install --upgrade pip
pip install --upgrade python-ace
```

Wheels are available for Linux, Apple Silicon, and Intel Macs.

## Add the GRACE fitting backend

On Linux or Apple Silicon, install:

```bash
pip install --upgrade tensorpotential 'numpy<2.4'
```

GRACE is published as `tensorpotential`; pip selects its TensorFlow and Keras
dependencies. The NumPy upper bound works around a fitting bug in the current
GRACE release.

## Check the installation

```bash
pip check
pacemaker --help
```

On Linux, GPU fitting requires a compatible NVIDIA driver. To use the CPU, set
`CUDA_VISIBLE_DEVICES=-1` before starting pacemaker.

Continue with the [quick start](quickstart.md) to prepare data and run a fit.

# Installation

Use Python 3.12 in a separate environment:

```bash
mamba create -n ace python=3.12 pip
mamba activate ace
```

You can use `conda` instead of `mamba`.

## Install pacemaker

```bash
python -m pip install --upgrade pip
python -m pip install 'python-ace==0.4.0'
```

Wheels are available for Linux, Apple Silicon, and Intel Macs.

## Add the GRACE fitting backend

On Linux or Apple Silicon, install:

```bash
python -m pip install 'tensorpotential==0.6.2.1' 'numpy>=2,<2.4' 'tensorflow~=2.20.0' 'tf-keras~=2.20.0'
```

GRACE is published as `tensorpotential`. Use these versions together: this
backend release requires NumPy below 2.4 for pacemaker fitting.
TensorFlow 2.20 does not provide Intel Mac wheels.

## Check the installation

```bash
python -m pip check
pacemaker --help
```

On Linux, GPU fitting requires a compatible NVIDIA driver. To use the CPU, set
`CUDA_VISIBLE_DEVICES=-1` before starting pacemaker.

Continue with the [quick start](quickstart.md) to prepare data and run a fit.

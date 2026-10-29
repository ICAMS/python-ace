# Installation

`pacemaker` is installed by the `python-ace` package. TensorFlow fitting uses
`tensorpotential.compat.pace` from the current
[GRACE TensorPotential repository](https://github.com/ICAMS/grace-tensorpotential).
The old `ICAMS/TensorPotential` repository requires Python <3.11 and should not
be used for a new installation.

## Create an environment

Python 3.12 is a suitable starting point for a new environment:

```bash
mamba create -n ace python=3.12 pip
mamba activate ace
python -m pip install --upgrade pip
```

`conda create` and `conda activate` can be used in place of the corresponding
`mamba` commands. Use the same environment to build and run both packages.
A C++ compiler is needed when building `python-ace` from source; its Python
build dependencies, including CMake and Ninja, are installed automatically.

## Install the fitting backend

Install the current backend from source:

```bash
git clone https://github.com/ICAMS/grace-tensorpotential.git
cd grace-tensorpotential
python -m pip install .
cd ..
```

This backend must include the PACE metrics scalar-conversion fix for NumPy
2.4 and newer. The released TensorPotential 0.6.2.1 also caps TensorFlow at
2.20; TensorFlow 2.21 needs the updated backend dependency bound. Updating
`python-ace` alone does not update either of these backend requirements.

The updated backend permits TensorFlow releases below 2.22. To select the
2.21 series explicitly, install matching TensorFlow and legacy Keras versions:

```bash
python -m pip install 'tensorflow~=2.21.0' 'tf-keras~=2.21.0'
```

On Linux the backend dependencies include `tensorflow[and-cuda]`. GPU execution
also requires a compatible NVIDIA driver. CPU fitting is available with
`CUDA_VISIBLE_DEVICES=-1`. TensorFlow 2.21 publishes wheels for Python
3.10–3.13; Python 3.14 is not covered by this installation recipe.

## Install pacemaker

For the modern source branch:

```bash
git clone --branch feature/build-speedup https://github.com/ICAMS/python-ace.git
cd python-ace
python -m pip install .
```

When working in an existing checkout, run `python -m pip install .` there.
This build includes the maxvol extension used by the active-learning tools;
a separate `maxvolpy` installation is unnecessary.

Check the installed versions and command-line tools:

```bash
python -m pip check
python -c 'import pyace, tensorpotential, tensorflow as tf; print(pyace.__version__, tensorpotential.__version__, tf.__version__)'
pacemaker --help
pace_info --help
```

To run the test suite from the `python-ace` checkout:

```bash
python -m pip install pytest
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/ --runtensorpot
```

The test workflow is configured for Python 3.10–3.13 with TensorFlow 2.20 and
2.21 on Linux CPUs, plus Python 3.9 with TensorFlow 2.20. GPU execution needs
separate validation on the target hardware.

### Tested development environment

The source refresh was checked on 29 October 2026 with Python 3.12.15,
TensorFlow 2.21.0, tf-keras 2.21.0, NumPy 2.5.4, and pandas 2.3.3:

| Execution target | Validation |
| --- | --- |
| Linux CPU | 294 tests and both Cu–I and Ethanol CLI workflows passed |
| NVIDIA H200 NVL, driver 595.58.03 | All 9 fitting tests passed; 185 fitting calls verified on GPU |
| NVIDIA A100 80 GB PCIe, driver 575.57.08 | All 9 fitting tests passed; 185 fitting calls verified on GPU |

The GPU checks verify device placement of energies, forces, and coefficient
gradients as well as the fitting-test results. They cover short fits, rather
than long-running memory stability or performance. These results require the
source fixes described above and do not describe an unmodified TensorPotential
0.6.2.1 installation.

## Troubleshooting

### GPU validation

To check fitting on a selected GPU from the source checkout:

```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python tools/validate_gpu.py
```

This runs the TensorFlow fitting tests and checks that energies, forces, and
coefficient gradients reside on the GPU. It fails if no GPU is detected or if
those results fall back to the CPU. Select one GPU; pacemaker does not fit on
multiple GPUs in parallel. Memory growth keeps the test from reserving all
available device memory.

If TensorFlow reports `Cannot dlopen some GPU libraries`, an existing system
CUDA installation may take precedence over the libraries installed by pip.
For the current shell, put the environment's CUDA library directories first:

```bash
pace_cuda_libraries=$(python - <<'PY'
from pathlib import Path
import sysconfig
print(':'.join(str(path) for path in sorted(
    (Path(sysconfig.get_paths()['purelib']) / 'nvidia').glob('*/lib')
)))
PY
)
export LD_LIBRARY_PATH="$pace_cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Then rerun the GPU check. This changes the shell environment and leaves the
system CUDA installation untouched.

### Python and dependency errors

If fitting raises `TypeError: only 0-dimensional arrays can be converted to
Python scalars`, check that both `python-ace` and the backend include the
NumPy scalar-conversion fix, then rebuild the installed packages.

For an import error or segmentation fault, check `which python` and rebuild
the extensions using the interpreter from the active environment. Avoid mixing
extensions built for different Python versions.

For a protobuf descriptor error, check `python -m pip check` and reinstall a
consistent TensorFlow environment. The old TensorFlow 2.5/2.8 and protobuf
3.20 instructions do not describe the current dependency set.

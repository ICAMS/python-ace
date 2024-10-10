# Installation

## (Optional) Creating a Separate Environment

It is a common practice to create a separate Python environment to avoid dependency conflicts. You can create a new environment named `ace` with the minimal required packages using the following command:


### Using `mamba` (Recommended)

[Mamba](https://mamba.readthedocs.io/en/latest/index.html) is a fast, robust, and cross-platform package manager. Check the [Mamba installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for details.

Installation steps:
1. Download the Miniforge3 distribution from [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3).
2. Run the downloaded file:
   ```bash
   sh Miniforge3-Linux-x86_64.sh
   ```
   and follow the installation instructions.
3. Create the `mamba` environment:
   ```bash
   mamba create -n ace python=3.9
   ```

Then, activate the environment with either:
```bash
source activate ace
```
or
```bash
mamba activate ace
```
To deactivate the environment, use the command:
```bash
mamba deactivate
```

### Using `conda`
> **Note:** Due to recent changes in Anaconda's licensing, please ensure you are permitted to use it. The `mamba` package manager might be a preferable alternative.

To create the environment with `conda`:
```bash
conda create -n ace python=3.9
```

## Installation of `tensorpotential`

`tensorpotential` allows for the GPU accelerated optimization of the ACE potential using [TensorFlow](https://www.tensorflow.org/).
However, it is recommended to use it even if you don't have a GPU available.


Install it using the following commands:

1. Install Tensorflow (newer version should be also compatible)
```
pip install tensorflow==2.8.0 
```
or to have CUDA support in latest versions of TensorFlow
```
pip install tensorflow[and-cuda] 
```

2. Download the `tensorpotential` from [this repository](https://github.com/ICAMS/TensorPotential).
* Clone with
```
git clone https://github.com/ICAMS/TensorPotential.git
cd TensorPotential
```
* or download
```
wget https://github.com/ICAMS/TensorPotential/archive/refs/heads/main.zip
unzip main.zip
cd TensorPotential-main
```
3. Run installation script
```
pip install --upgrade .
```
or (for more installation details)
```
python setup.py install
```

## Installation of `pacemaker` and `pyace`

The `pyace` (aka `python-ace`) package is located in [this repository](https://github.com/ICAMS/python-ace).
It contains the `pacemaker` tools and other Python wrappers and utilities.

To install `pyace`:

1. Download `pyace` from [this repository](https://github.com/ICAMS/python-ace).
* Clone with
```
git clone https://github.com/ICAMS/python-ace.git
cd python-ace
```

* or download 
```
wget https://github.com/ICAMS/python-ace/archive/refs/heads/master.zip
cd python-ace-master
```
2. Run installation script
```
pip install --upgrade .
```
or (for more installation details)
```
python setup.py install
```

Now, `pacemaker` and other tools (`pace_yaml2yace`, `pace_info`, `pace_activeset`) should be available from the terminal, if corresponding conda environment is loaded.

## Known installation issues

### Segmentation fault
If you see `Segmentation fault` error message, then check that you are using correct version of Python from corresponding conda environment,
i.e. check that `which python` points to right location inside conda environment.

### TypeError: Descriptors cannot not be created directly
If you see this error message
```
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).```
```
then try to downgrade `protobuf` package, i.e.
```
pip install protobuf==3.20.*
```

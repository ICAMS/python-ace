# Installation

## (optional) Creating a conda environment
It is common practice creating a separate `conda environment` to avoid dependencies mixing.
You can create the new environment named `ace` with minimal amount of required packages with the following command: 

```
conda create -n ace python<3.9
```
Then, activate the environment with 
`source activate ace` or `conda activate ace`. To deactivate the environment, use `deactivate` command 

## Installation of `tensorpotential`

`tensorpotential` allows for the GPU accelerated optimization of the ACE potential using [TensorFlow](https://www.tensorflow.org/).
However, it is recommended to use it even if you don't have a GPU available.


Install it using the following commands:

1. Install Tensorflow
```
pip install tensorflow==2.8.0 # newer version should be also compatible
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
3. Run installation script
```
pip install --upgrade .
```
or (for more installation details)
```
python setup.py install
```

Now, `pacemaker` and other tools (`pace_yaml2yace`, `pace_info`, `pace_activeset`) should be available from the terminal, if corresponding conda environment is loaded.

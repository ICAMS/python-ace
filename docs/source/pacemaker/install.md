# Installation

## (optional) Creating a conda environment
It is common practice creating a separate `conda environment` to avoid dependencies mixing.
You can create the new environment named `ace` with minimal amount of required packages with the following command: 

```
conda env create -n ace python<3.9
```
Then, activate the environment with 
`source activate ace` or `conda activate ace`. To deactivate the environment, use `deactivate` command 

## Installation of `tensorpotential`

`tensorpotential` allows for the GPU accelerated optimization of the ACE potential using
[TensorFlow](https://www.tensorflow.org/). However, it is recommended to use it even if you don't have a GPU available.
Install it using the following commands:

```
pip install tensorflow==2.5.0 # newer version should be also compatible
cd tensorpotential
pip install --upgrade .
```

## Installation of `pyace`

Finally, `pyace` could be installed with 

```
cd pyace
pip install --upgrade .
```


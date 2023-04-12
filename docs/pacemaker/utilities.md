# Utilities

## Potential conversion

There are **two** basic formats for ACE potentials:

1. **B-basis set** - YAML format, i.e. 'Al.pbe.yaml'. This is an internal *complete* format for potential fitting.
2. **Ctilde-basis set** - YACE (special form of YAML) format, i.e. 'Al.pbe.yace'. This format is *irreversibly* converted from *B-basis set* for
   public potentials distribution and for using in LAMMPS simulations.

Please see [pacemaker paper] for more details about **B-basis** and **Ctilde-basis sets**

To convert potential you can use following utility, that is installed together with `pyace` package into you executable paths:
* `YAML` to `yace` : `pace_yaml2yace`. Usage:
```
  usage: pace_yaml2yace [-h] [-o OUTPUT] input [input ...]

Conversion utility from B-basis (.yaml file) to new-style Ctilde-basis (.yace
file)

positional arguments:
  input                 input B-basis file name (.yaml)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output Ctilde-basis file name (.yace)

```

## YAML potential timing

Utility to run the single-CPU timing test for PACE (.yaml) potential.
Usage:
```c
pace_timing [-h] potential_file
```

## YAML potential info

Utility to show the basic information (type of embedding, cutoff, radial functions, n-max, l-max etc.) for PACE (.yaml) potential.
Usage:
```c
pace_info [-h] potential_file
```

## Collect and store VASP data in pickle file

Utility to collect VASP calculations from a top-level directory and store them in a `*.pkl.gz` file that can be used for fitting with `pacemaker`. 
The reference energies could be provided for each element (default value is zero) 
or extracted automatically from the calculation with single atom and large enough (>500 Ang^3/atom) volume. Usage: 

```
usage: pace_collect [-h] [-wd WORKING_DIR] [--output-dataset-filename OUTPUT_DATASET_FILENAME]
 [--free-atom-energy [FREE_ATOM_ENERGY [FREE_ATOM_ENERGY ...]]] [--selection SELECTION]

optional arguments:
  -h, --help            show this help message and exit
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        top directory where keep calculations
  --output-dataset-filename OUTPUT_DATASET_FILENAME
                        pickle filename, default is collected.pkl.gz
  --free-atom-energy [FREE_ATOM_ENERGY [FREE_ATOM_ENERGY ...]]
                        dictionary of reference energies (auto for extraction from dataset), i.e. `Al:-0.123 Cu:-0.456 Zn:auto`, default is zero. If option is `auto`, then it will be extracted from dataset
  --selection SELECTION
                        Option to select from multiple configurations of single VASP calculation: first, last, all, first_and_last (default: last)

```
## Active set generation

Utility to generate active set (used for extrapolation grade calculation).

```
usage: pace_activeset [-h] [-d DATASET] [-f] [-b BATCH_SIZE] [-g GAMMA_TOLERANCE] [-i MAXVOL_ITERS] [-r MAXVOL_REFINEMENT] [-m MEMORY_LIMIT] potential_file

Utility to compute active set for PACE (.yaml) potential

positional arguments:
potential_file        B-basis file name (.yaml)

optional arguments:
   -h, --help            show this help message and exit
   -d DATASET, --dataset DATASET
   Dataset file name, ex.: filename.pkl.gz
   -f, --full            Compute active set on full (linearized) design matrix
   -b BATCH_SIZE, --batch_size BATCH_SIZE
   Batch size (number of structures) considered simultaneously.If not provided - all dataset at once is considered
   -g GAMMA_TOLERANCE, --gamma_tolerance GAMMA_TOLERANCE
   Gamma tolerance
   -i MAXVOL_ITERS, --maxvol_iters MAXVOL_ITERS
   Number of maximum iteration in MaxVol algorithm
   -r MAXVOL_REFINEMENT, --maxvol_refinement MAXVOL_REFINEMENT
   Number of refinements (epochs)
   -m MEMORY_LIMIT, --memory-limit MEMORY_LIMIT
   Memory limit (i.e. 1GB, 500MB or 'auto')
```

Example of usage:

```
pace_activeset -d fitting_data_info.pkl.gz output_potential.yaml
```
that will generate **linear** active set and store it into `output_potential.asi` file.

or

```
pace_activeset -d fitting_data_info.pkl.gz output_potential.yaml -f
```
that will generate **full** active set (including linearized part of non-linear embedding function)
and store it into `output_potential.asi.nonlinear` file.

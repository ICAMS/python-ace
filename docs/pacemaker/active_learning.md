# Extrapolation grade and active learning 

For any fitted ACE potential and corresponding training set 
(usually stored by `pacemaker` into `fitting_data_info.pckl.gzip` file in working directory)
one can generate corresponding active set for linear B-projections (default) of full non-linear embedding.
Practice shows that linear active set is enough for extrapolation grade estimation.
However, if you want more sensitive (and "over-secure") extrapolation grade, then full active set could be used.



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
   Dataset file name, ex.: filename.pckl.gzip
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
pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml
```
that will generate **linear** active set and store it into `output_potential.asi` file.

or

```
pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml -f
```
that will generate **full** active set (including linearized part of non-linear embedding function)
and store it into `output_potential.asi.nonlinear` file.

## Usage of active set with LAMMPS 

Example of usage of active set with LAMMPS
```
pair_style  pace/extrapolation
pair_coeff  * * output_potential.yaml output_potential.asi Al Cu

# compute per-atom extrapolation grade every 10 steps
fix pace_gamma all pair 10 pace/extrapolation gamma 1
# compute maximum extrapolation grade over complete structure
compute max_pace_gamma all reduce max f_pace_gamma

# dump extrapolative structures if c_max_pace_gamma > 5, skip otherwise, check every 20 steps 
variable dump_skip equal "c_max_pace_gamma < 5"
dump pace_dump all custom 20 extrapolative_structures.dump id type x y z f_pace_gamma
dump_modify pace_dump skip v_dump_skip

# stop simulation if maximum extrapolation grade exceeds 25
variable max_pace_gamma equal c_max_pace_gamma
fix extreme_extrapolation all halt 10 v_max_pace_gamma > 25
```

Check [LAMMPS documentation](https://docs.lammps.org/latest/pair_pace.html) for more details and example

With this setup you can run LAMMPS simulations and make use of per-atom extrapolation grade `f_pace_gamma` fix variable 
(i.e. in regular dump and visualization) or per-structure `c_max_pace_gamma` maximum extrapolation grade in thermo_style.

Two main scenarios:
1. Exploring new structures (and dump extrapolative structures with `dump pace_dump`).
In that case extrapolative structures will be stored into `extrapolative_structures.dump` file, that could be loaded 
(i.e. with ASE) and DFT calculations could be performed with the tools of your choice.
2. Performing normal simulations, observing extrapolation grade (printing `c_max_pace_gamma` variable)
and stopping at extreme_extrapolation (with `fix halt`)  

## Usage of active set with ASE calculator

```python
from pyace import *

calc = PyACECalculator("output_potential.yaml")
calc.set_active_set("output_potential.asi")

# set calculator to ASE atoms
atoms.set_calculator(calc)

# trigger calculation
atoms.get_potential_energy()

#per-atom extrapolation grades are stored in
calc.results["gamma"]
```

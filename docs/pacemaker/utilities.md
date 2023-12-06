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

Utility to collect VASP calculations from a top-level directory and store them in a `*.pckl.gzip` file that can be used for fitting with `pacemaker`. 
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
                        pickle filename, default is collected.pckl.gzip
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

## D-optimality structure selection

Utility to select most representative training structures from large dataset using D-optimality criterion.

```
usage: pace_select [-h] -p POTENTIAL_FILE [-a ACTIVE_SET_INV_FNAME] [-e ELEMENTS] [-m MAX_STRUCTURES] [-o SELECTED_STRUCTURES_FILENAME] [-b BATCH_SIZE] [-g GAMMA_TOLERANCE] [-i MAXVOL_ITERS] [-r MAXVOL_REFINEMENT] [-mem MEMORY_LIMIT] [-V] dataset [dataset ...]

Utility to select structures for training se based on D-optimality criterion

positional arguments:
  dataset               Dataset file name(s), ex.: filename.pckl.gzip [extrapolative_structures.dat]

optional arguments:
  -h, --help            show this help message and exit
  -p POTENTIAL_FILE, --potential_file POTENTIAL_FILE
                        B-basis file name (.yaml)
  -a ACTIVE_SET_INV_FNAME, --active-set-inv ACTIVE_SET_INV_FNAME
                        Active Set Inverted (ASI) filename
  -e ELEMENTS, --elements ELEMENTS
                        List of elements, used in LAMMPS, i.e. "Ni Nb O"
  -m MAX_STRUCTURES, --max-structures MAX_STRUCTURES
                        Maximum number of structures to select (default -1 = all)
  -o SELECTED_STRUCTURES_FILENAME, --output SELECTED_STRUCTURES_FILENAME
                        Selected structures filename, default: selected.pkl.gz
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size (number of structures) considered simultaneously.If not provided - all dataset at once is considered
  -g GAMMA_TOLERANCE, --gamma_tolerance GAMMA_TOLERANCE
                        Gamma tolerance
  -i MAXVOL_ITERS, --maxvol_iters MAXVOL_ITERS
                        Number of maximum iteration in MaxVol algorithm
  -r MAXVOL_REFINEMENT, --maxvol_refinement MAXVOL_REFINEMENT
                        Number of refinements (epochs)
  -mem MEMORY_LIMIT, --memory-limit MEMORY_LIMIT
                        Memory limit (i.e. 1GB, 500MB or 'auto')
  -V                    suppress verbosity of numerical procedures
```

Example of usage:
```bash
pace_select -p NiNb_potential.yaml -a NiNb_potential.asi -m 100 -e "Ni Nb"  extrapolative_structures_1.dump extrapolative_structures_2.dump
```

## Data augmentation

Utility to generate augmented dataset. Energies and forces will be predicted with ZBL potential.
```
usage: pace_augment [-h] -d DATASET [-a ACTIVE_SET_INV_FNAME] [-m MAX_STRUCTURES] [-o AUGMENTED_STRUCTURES_FILENAME] [-V] [-mnat MAX_NUM_AT] [-mss MAX_SEED_STRUCTURES] [-minepa MIN_AUG_EPA] [-maxepa MAX_AUG_EPA] [-eparmax EPA_RELIABLE_MAX] [-nnstep NN_DISTANCE_STEP]
                    [-nnmin NN_DISTANCE_MIN]
                    potential_file

Utility to generate augmented dataset with ZBL and/or EOS data

positional arguments:
  potential_file        B-basis file name (.yaml)

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset file name(s), ex.: -d filename.pckl.gzip [-d filename2.pckl.gzip]
  -a ACTIVE_SET_INV_FNAME, --active-set-inv ACTIVE_SET_INV_FNAME
                        Active Set Inverted (ASI) filename, considered as extra B-projections
  -m MAX_STRUCTURES, --max-structures MAX_STRUCTURES
                        Maximum number of structures to select (default -1 = all)
  -o AUGMENTED_STRUCTURES_FILENAME, --output AUGMENTED_STRUCTURES_FILENAME
                        Augmented structures filename, (default: aug_df.pkl.gz)
  -V                    suppress verbosity of numerical procedures
  -mnat MAX_NUM_AT, --max-num-atoms MAX_NUM_AT
                        Maximum number of atoms for seed structures, selected for augmentation (-1 = no limit, default = 32)
  -mss MAX_SEED_STRUCTURES, --max-seed-structures MAX_SEED_STRUCTURES
                        Maximum number of seed structures, selected for augmentation (-1 = all, default = 100)
  -minepa MIN_AUG_EPA, --min--aug-epa MIN_AUG_EPA
                        Minimal augmented energy-per-atom (default None = no limit)
  -maxepa MAX_AUG_EPA, --max--aug-epa MAX_AUG_EPA
                        Maximal augmented energy-per-atom (default None = 150)
  -eparmax EPA_RELIABLE_MAX, --epa-reliable-max EPA_RELIABLE_MAX
                        Maximum for reliable energy-per-atom (default None = no limit)
  -nnstep NN_DISTANCE_STEP, --nn-dist-step NN_DISTANCE_STEP
                        Nearest-neighbour distance step for data augmentation (default = 0.1)
  -nnmin NN_DISTANCE_MIN, --nn-dist-min NN_DISTANCE_MIN
                        Nearest-neighbour distance step for data augmentation (default = 1)

```

Example of usage:
```bash
pace_augment NiNb-FM-upfit-mu.yaml -a NiNb-FM-upfit-mu.asi -d df_all_NbNi_FM_new_str.pckl.gzip    
```


## Core-repulsion tuner

Utility for automatic/manual setup of ZBL core-repulsion.

```
usage: pace_corerep [-h] [-d DATASET] [-a ACTIVE_SET_INV_FNAME] [-o OUTPUT_FILE] [-V] [-nnstep NN_DISTANCE_STEP] [-nnmin NN_DISTANCE_MIN] [-nnmax NN_DISTANCE_MAX] [-n NUM_OF_STRUCTURES] [-g GAMMA_MAX] [--inner-cutoff [INNER_CUTOFF_DICT [INNER_CUTOFF_DICT ...]]] potential_file

Utility to (auto)tune potential and add ZBL core-repulsion ZBL

positional arguments:
  potential_file        B-basis file name (.yaml)

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset file name(s), ex.: -d filename.pckl.gzip [-d filename2.pckl.gzip]
  -a ACTIVE_SET_INV_FNAME, --active-set-inv ACTIVE_SET_INV_FNAME
                        Active Set Inverted (ASI) filename, considered as extra B-projections
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        Output filename for auto-tuned core-rep potential. default=none - same as `potential_file`. If `auto` - 'corerep' suffix will be added
  -V                    suppress verbosity of numerical procedures
  -nnstep NN_DISTANCE_STEP, --nn-dist-step NN_DISTANCE_STEP
                        Nearest-neighbour distance step for data augmentation (default = 0.05)
  -nnmin NN_DISTANCE_MIN, --nn-dist-min NN_DISTANCE_MIN
                        Min. nearest-neighbour distance for data augmentation (default = 1)
  -nnmax NN_DISTANCE_MAX, --nn-dist-max NN_DISTANCE_MAX
                        Max. nearest-neighbour distance for data augmentation (default = 2.5)
  -n NUM_OF_STRUCTURES, --num-of-structures NUM_OF_STRUCTURES
                        Number of structures selected to compress (default = 50)
  -g GAMMA_MAX, --gamma-max GAMMA_MAX
                        Max. extrapolation grade gamma for reliable atomic env. (default = 10)
  --inner-cutoff [INNER_CUTOFF_DICT [INNER_CUTOFF_DICT ...]]
                        dictionary of inner cutoff `Al:-0.123 Cu-Cu:-0.456 Al-Cu:auto`, default is zero. If option is `auto`, then it will be extracted from dataset

```

Example of usage, automatic:
```bash
pace_corerep NiNb-5-FM-ZBL.yaml -a NiNb-5-FM-ZBL.asi -d fitting_data_info.pckl.gzip
```
manual configuration of inner cutoff
```bash
pace_corerep AlLi-8a-auto.yaml --inner-cutoff Al:1.95 Li:1.90 Al-Li:1.85
```
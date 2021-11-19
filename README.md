# pyace

`pyace` is the python implementation of [`ace`](https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/ace).
It provides the basis functionality for analysis, potentials conversion and fitting.

## Installation

### Cloning `pyace` repo
Clone the git repo with the commands:

```
git clone https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/pyace.git --recurse-submodules
```
The last command here is needed to clone the original `ace` and `ace-evaluator` repositories.
 
### (optional) Creating a conda environment
It is common practice to create a separate `conda environment` to avoid dependencies mixing.
You can create the new environment named `ace` with minimal amount of required packages,
specified in `environment.yml` file with the following command: 
```
conda env create -f environment.yml
```
Then, activate the environment with 
`source activate ace` or `conda activate ace`. To deactivate the environment, use `deactivate` command 

### (optional) Installation of `tensorpotential`  
If you want to use `TensorFlow` implementation of atomic cluster expansion 
(made by *Dr. Anton Bochkarev*), then use the following commands:
```
git clone git@git.noc.ruhr-uni-bochum.de:atomicclusterexpansion/tensorpotential.git
cd tensorpotential
python setup.py install
```
### Installation of `pyace`
Finally, `pyace` could be installed with 
```
python setup.py install
```

### (!) Known issues
If you will encounter `segmentation fault` errors,  then try to upgrade the `numpy` package with the command:
```
pip install --upgrade numpy --force 
```

## Updating installation
```
git pull --recurse-submodules
python setup.py install
```
## Directory structure

- **lib/**: contains the extra libraries for `pyace`
- **src/pyace/**: bindings

# Utilities
## Potential conversion

There are **three** basic formats ACE potentials:

1. Fortran implementation format, i.e. 'Al.pbe.in'
2. **B-basis set** in YAML format, i.e. 'Al.pbe.yaml'. This is an internal developers *complete* format 
3. **Ctilde-basis set** in plain text format, i.e. 'Al.pbe.ace'. This format is *irreversibly* converted from *B-basis set* for
public potentials distribution and is used by LAMMPS.

To convert potential you can use following utilities, that are installed together with `pyace` package into you executable paths:
  * `Fortran` to  `YAML`: `pace_fortran2yaml`. Usage: 
``` 
pace_fortran2yaml <ace_fortran_potential> <output.yaml> [--verbose]
```
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
### Update YAML potential
If you see the following message 
```c
DEPRECATION WARNING!!! Old (flatten) radcoefficients parameter encounterd, whereas it should be three-dimensional with [nradmax][lmax+1][nradbase] shape.
Automatic reshaping will be done
```
then you could update given YAML potential file with a command: `pace_update_yaml_potential`

Usage:
```c
pace_update_yaml_potential [-h] [-o OUTPUT] input
```
### YAML potential timing 

Utility to run the single-CPU timing test for PACE (.yaml) potential.
Usage:
```c
pace_timing [-h] potential_file
```

## Pacemaker

`pacemaker` is a utility for fitting the atomic cluster expansion potential. Usage:

```
usage: pacemaker [-h] [-c] [-o OUTPUT] [-p POTENTIAL] [-ip INITIAL_POTENTIAL]
                 [-b BACKEND] [-d DATA] [--query-data] [--prepare-data]
                 [--rebuild] [-l LOG] [-dr] [-t]
                 [input]

Fitting utility for atomic cluster expansion potentials

positional arguments:
  input                 input YAML file, default: input.yaml

optional arguments:
  -h, --help            show this help message and exit
  -c, --clean           Remove all generated data
  -o OUTPUT, --output OUTPUT
                        output B-basis YAML file name, default:
                        output_potential.yaml
  -p POTENTIAL, --potential POTENTIAL
                        input potential YAML file name, will override input
                        file 'potential' section
  -ip INITIAL_POTENTIAL, --initial-potential INITIAL_POTENTIAL
                        initial potential YAML file name, will override input
                        file 'potential::initial_potential' section
  -b BACKEND, --backend BACKEND
                        backend evaluator, will override section
                        'backend::evaluator' from input file
  -d DATA, --data DATA  data file, will override section 'YAML:fit:filename'
                        from input file
  --query-data          query the training data from database, prepare and
                        save them
  --prepare-data        prepare and save training data only
  --rebuild             force to rebuild necessary neighbour lists
  -l LOG, --log LOG     log filename, default: log.txt
  -dr, --dry-run        Dry run: performs all preprocessing and analysis, but
                        do not do the fitting
  -t, --template        Create a template 'input.yaml' file
  -v, --version         Show version info
  --no-fit              Do not fit the potential
  --no-predict          Do not compute and save the predictions
  --verbose-tf          Make tensorflow more verbose (off by defeault)

``` 

The required settings are provided by input YAML file. The main sections
#### 1. Cutoff and  (optional) metadata

* Global cutoff for the fitting is setup as:

```YAML
cutoff: 10.0
```

* Metadata (optional)

This is arbitrary key (string)-value (string) pairs that would be added to the potential YAML file: 
```YAML
metadata:
  info: some info
  comment: some comment
  purpose: some purpose
```
Moreover, `starttime` and `user` fields would be added automatically

#### 2.Dataset specification section
Fitting dataset could be queried automatically from `structdb` (if corresponding `structdborm` package is installed and 
connection to database is configured, see `structdb.ini` file in home folder). Alternatively, dataset could be saved into
file as a pickled `pandas` dataframe with special names for columns: #TODO: add columns names
 
Example:
```YAML
data: # dataset specification section
  # data configuration section
  config:
    element: Al                    # element name
    calculator: FHI-aims/PBE/tight # calculator type from `structdb` 
    # ref_energy: -1.234           # single atom reference energy
                                   # if not specified, then it will be queried from database

  # seed: 42                       # random seed for shuffling the data  
  # query_limit: 1000              # limiting number of entries to query from `structdb`
                                   # ignored if reading from cache
    
  # cache_ref_df: True             # whether to store the queried or modified dataset into file, default - True
  # filename: some.pckl.gzip       # force to read reference pickled dataframe from given file
  # ignore_weights: False          # whether to ignore energy and force weighting columns in dataframe
  # datapath: ../data              # path to folder with cache files with pickled dataframes 
```
Alternatively, instead of `data::config` section, one can specify just the cache file 
with pickled dataframe as `data::filename`:
```YAML
data: 
  filename: small_df_tf_atoms.pckl
  datapath: ../tests/
```
`data:datapath` option, if not provided, could be replaced with *environment variable* **PACEMAKERDATAPATH**

Example of creating the **subselection of fitting dataframe** and saving it is given in `notebooks/data_preprocess.ipynb`

Example of generating **custom energy/forces weights** is given in `notebooks/data_custom_weights.ipynb`

##### Querying data
You can just query and preprocess data, without running potential fitting.
Here is the minimalistic input YAML:

```YAML
# input.yaml file

cutoff: 10.0  # use larger cutoff to have excess neighbour list
data: # dataset specification section
  config:
    element: Al                    # element name
    calculator: FHI-aims/PBE/tight # calculator type from `structdb`
  seed: 42
  datapath: ../data                # path to the directory with cache files
  # query_limit: 100               # number of entries to query  
```

Then execute `pacemaker --query-data input.yaml` to query and build the dataset with `pyace` neighbour lists.
For building *both* `pyace` and `tensorpot` neighbour lists use `pacemaker --query-data input.yaml -b tensorpot`

##### Preparing the data / constructing neighbourlists
You can use existing `.pckl.gzip` dataset and generate all necessary columns for that, including neighbourlists
Here is the minimalistic input YAML:

```YAML
# input.yaml file

cutoff: 10.

data:
  filename: my_dataset.pckl.gzip

backend:
  evaluator: tensorpot  # pyace, tensorpot

```

Then execute `pacemaker --prepare-data input.yaml`
Generation of the `my_dataset.pckl.gzip` from, for example, *pyiron* is shown in `notebooks/convert-pyiron-to-pacemaker.ipynb` 

##### Test set (experimental mode/tensorpot only)

You could provide test set either as a fraction or certain number of samples from the train set (option `test_size`) or
as a separate pckl.gzip file (option `test_filename`)

```yaml
data:
  test_filename: my_test_dataset.pckl.gzip
```

or

```yaml
data:
  test_size: 100 # would take 100 samples randomly from train/fit set
  # test_size: 0.1 #  if <1 - would take given fraction of samples randomly from train/fit set
```

#### 3. Interatomic potential (or B-basis) configuration
##### 3.1 Single specie (deprecated)
For single specie one could define the initial interatomic potential configuration as:
```YAML
potential:
  deltaSplineBins: 0.001
  element: Al
  fs_parameters: [1, 1, 1, 0.5]
  npot: FinnisSinclair
  NameOfCutoffFunction: cos

  rankmax: 3
  nradmax: [ 4, 3, 3 ]  # per-order values of nradmax
  lmax: [ 0, 1, 1 ]     # per-order values of lmax,  lmax=0 for first order always!

  ndensity: 2
  rcut: 8.7
  dcut: 0.01
  radparameters: [ 5.25 ]
  radbase: ChebExpCos

 # initial_potential: whatever.yaml                      # in "ladder" fitting scheme, potential from with to start fit
  
 ##hard-core repulsion (optional)
 # core-repulsion: [500, 10]
 # rho_core_cut: 50
 # drho_core_cut: 20

 # (optional): Initialization values for BBasisFunctions coefficients
 # func_coefs_init: zero #  zero (default) or random  
```


##### 3.2 Multiple species (recommended)
```YAML
potential:
  deltaSplineBins: 0.001
  elements: [Al, Ni]  # list of all element

  # Embeddings are specified for each individual elements,
  # all parameters could be distinct for different species
  embeddings: ## possible keywords: ALL, UNARY, elements: Al, Ni
    Al: {
      npot: 'FinnisSinclairShiftedScaled',
      fs_parameters: [1, 1, 1, 0.5], ## non-linear embedding function: 1*rho_1^1 + 1*rho_2^0.5
      ndensity: 2,
      
      # core repulsion parameters
      rho_core_cut: 200000,
      drho_core_cut: 250
    }

    Ni: {
      npot: 'FinnisSinclairShiftedScaled', ## linear embedding function: 1*rho_1^1
      fs_parameters: [1, 1],
      ndensity: 1,

      # core repulsion parameters
      rho_core_cut: 3000,
      drho_core_cut: 150
    }

  ## Bonds are specified for each possible pairs of elements
  ## One could use keywords: ALL (Al,Ni, AlNi, NiAl)
  bonds: ## possible keywords: ALL, UNARY, BINARY, elements pairs as AlAl, AlNi, NiAl, etc...  
    ALL: {
        radbase: ChebExpCos,
        radparameters: [5.25],

        ## outer cutoff, applied in a range [rcut - dcut, rcut]
        rcut: 5,
        dcut: 0.01,

        ## inner cutoff, applied in a range [r_in, r_in + delta_in]
        r_in: 1.0,
        delta_in: 0.5,
        
        ## core-repulsion parameters `prefactor` and `lambda` in
        ## prefactor*exp(-lambda*r^2)/r, >0 only r<r_in+delta_in
        core-repulsion: [0.0, 5.0],
    }

    ## BINARY overwrites ALL settings when they are repeated
    BINARY: {
        radbase: ChebPow,
        radparameters: [6.25],

        ## cutoff may vary for different bonds
        rcut: 5.5,
        dcut: 0.01,

        ## inner cutoff, applied in a range [r_in, r_in + delta_in]
        r_in: 1.0,
        delta_in: 0.5,

        ## core-repulsion parameters `prefactor` and `lambda` in
        ## prefactor*exp(-lambda*r^2)/r, >0 only r<r_in+delta_in
        core-repulsion: [0.0, 5.0],
    }
  

  functions: # possible keywords: ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY, element combinations as (Al,Al), (Al, Ni), (Al, Ni, Zn), etc...
    UNARY: {
      nradmax_by_orders: [15, 3, 2, 2, 1],
      lmax_by_orders: [ 0, 2, 2, 1, 1],
      # coefs_init: zero # initialization of functions coefficients: zero (default) or random
    }

    BINARY: {
      nradmax_by_orders: [15, 2, 2, 2],
      lmax_by_orders: [ 0, 2, 2, 1],
      # coefs_init: zero # initialization of functions coefficients: zero (default) or random
    }
```

In sections `embeddings`,  `bonds` and `functions` one could use keywords (ALL, UNARY, BINARY, TERNARY, QUATERNARY, QUINARY).
The settings provided by more specific keyword will override those from less specific keyword, i.e. ALL < UNARY < BINARY < ('Al','Ni') 

##### 3.3 Continuation of fitting
If you want to continue fitting of the existing potential in `potential.yaml` file, then specify
```YAML
potential: potential yaml
```
alternatively, one could use `pacemaker ... -p potential.yaml ` option.

For specifying both initial and target potential from the file one could provide:
```YAML
potential: 
  filename: potential.yaml
  # initial_potential: initial_potential.yaml   # in "ladder" fitting scheme, potential from with to start fit
  # reset: true # reset potential from potential.yaml, i.e. set radial coefficients to delta_nk and func coeffs=[0..]
```
or alternatively, one could use  `pacemaker ... -p potential.yaml -ip initial_potential.yaml ` options.

#### 4. Fitting settings
Example of `fit` section is:
```YAML
fit:
    ## LOSS FUNCTION OPTIONS ##
    loss: { kappa: 0, ## [0..1] or auto, relative force weight, 0 - energies-only fit, 1-forces-only fit, auto - determined from dataset 
          L1_coeffs: 0, ## 1e-5, L1-regularization coefficient
          L2_coeffs: 0, ## 1e-5, L2-regularization coefficient
          w0_rad: 0, ## w0 radial smoothness regularization coefficient
          w1_rad: 0, ## w1 radial smoothness regularization coefficient
          w2_rad: 0  ## w2 radial smoothness regularization coefficient
    }

    ## DATA WEIGHTING OPTIONS ##
    weighting: {
        ## weights for the structures energies/forces are associated according to the distance to E_min: convex hull ( energy: convex_hull)
        ## or minimal energy per atom (energy: cohesive)
        type: EnergyBasedWeightingPolicy,
        nfit: 10000, ## number of structures to randomly select from the initial dataset
        
        ## only the structures with energy up to E_min + DEup will be selected
        DEup: 10.0,  ## eV, upper energy range (E_min + DElow, E_min + DEup)
        
        ## only the structures with maximal force on atom  up to DFup will be selected
        DFup: 50.0, ## eV/A
        DElow: 1.0,  ## eV, lower energy range (E_min, E_min + DElow)
        
        DE: 1.0, ## delta_E  shift for weights, see paper
        DF: 1.0, ## delta_F  shift for weights, see paper
        
        wlow: 0.75, ## 0<wlow<1 or None: if provided, the renormalization weights of the structures on lower energy range (see DElow)
        energy: convex_hull, ##  "convex_hull" or "cohesive" : method to compute the E_min 
        reftype: all, ## all (default), bulk or cluster
        seed: 42 ## random number seed
    }
    
    ## Custom weights:  corresponding to main dataset index and `w_energy` and `w_forces` columns should be provided in pckl.gzip file
    #weighting: {type: ExternalWeightingPolicy, filename: custom_weights_only.pckl.gzip}
    
    ## OPTIMIZATION OPTIONS ##
    optimizer: BFGS # BFGS, L-BFGS-B, Nelder-Mead, etc. : scipy minimization algorithm
    ## additional options for scipy.minimize(..., options={...}, ...)
    #options: {maxcor: 100}    
    maxiter: 1000 # maximum number of iteration for EACH scipy minimization round
    
    ## EXTRA OPTIONS ##
    repulsion: auto            # set inner cutoff based on the minimal distance in the dataset
      
    #trainable_parameters: ALL  # ALL, UNARY, BINARY, ..., radial, func, {"AlNi": "func"}, {"AlNi": {"func","radial"}}, ...

    ##(optional) number of consequentive runs of fitting algorithm (for each ladder step), that helps convergence
    #fit_cycles: 1   
    
    ## starting from second fit_cycle:
    
    ## applies Gaussian noise with specified relative sigma/mean ratio to all potential trainable coefficients
    #noise_relative_sigma: 1e-3
    
    ## applies Gaussian noise with specified absolute sigma to all potential trainable coefficients
    #noise_absolute_sigma: 1e-3  
    
    # reset the function coefficients according to Gaussian distribution with given sigma; enable ensemble fitting mode
    #randomize_func_coeffs: 1e-3  

    ## LADDER SCHEME (i.e. hierarchical fitting) ##      
    ## enables hierarchical fitting (LADDER SCHEME), that sequentially add specified number of B-functions (LADDER STEP)    
    #ladder_step: [10, 0.02]  
    ##      - integer >= 1 - number of basis functions to add in ladder scheme,
    ##      - float between 0 and 1 - relative ladder step size wrt. current basis step
    ##      - list of both above values - select maximum between two possibilities on each iteration 
    ##     see. Ladder scheme fitting for more info
    
    #ladder_type: body_order    ## default
                                ## Possible values:
                                ## body_order  -  new basis functions are added according to the body-order, i.e., a function with higher body-order
                                ##                will not be added until the list of functions of the previous body-order is exhausted
                                ## power_order -  the order of adding new basis functions is defined by the "power rank" p of a function.
                                ##                p = len(ns) + sum(ns) + sum(ls). Functions with the smallest p are added first

    ## callbacks during the fitting. Module quick_validation.py should be available for import
    ## see example/pacemaker_with_callback for more details and examples
    #callbacks:
    #  - quick_validation.test_fcc_potential_callback
```

If not specified, then *uniform weight* and *energy-only* fit (kappa=0),
*fit_cycles*=1, *noise_relative_sigma* = 0 settings will be used.

If ladder fitting scheme is used, then intermediate version of the potential after each ladder step will be saved
into `interim_potential_ladder_step_{LADDER_STEP}.yaml`.

#### 5. Backend specification

```YAML
backend:
  evaluator: tensorpot  # pyace, tensorpot

 ## for `tensorpot` evaluator, following options are available:
 # batch_size: 10            # batch size for loss function evaluation, default is 10
 # batch_size_reduction: True # automatic batch_size reduction if not enough memory (default - True) 
 # batch_size_reduction_factor: 1.618  # batch size reduction factor
 # display_step: 20          # frequency of detailed metric calculation and printing
 
 ## for `pyace` evaluator, following options are available:
 # parallel_mode: process    # process, serial  - parallelization mode for `pyace` evaluator
 # n_workers: 4              # number of parallel workers for `process` parallelization mode
```
Alternatively, backend could be selected as `pacemaker ... -b tensorpot` 

##  Ladder scheme fitting 
In a ladder scheme potential fitting happens by adding new portion of basis functions step-by-step,
to form a "ladder" from *initial potential* to *final potential*. Following settings should be added to
the input YAML file:

* Specify *final potential* shape by providing `potential` section:
```yaml
potential:
  deltaSplineBins: 0.001
  element: Al
  fs_parameters: [1, 1, 1, 0.5]
  npot: FinnisSinclair
  NameOfCutoffFunction: cos
  rankmax: 3

  nradmax: [4, 1, 1]
  lmax: [0, 1, 1]

  ndensity: 2
  rcut: 8.7
  dcut: 0.01
  radparameters: [5.25]
  radbase: ChebExpCos 
```

* Specify *initial or intermediate potential* by providing `initial_potential` option in `potential` section: 
```yaml
potential:
    ...
    initial_potential: some_start_or_interim_potential.yaml    # potential to start fit from
```
If *initial or intermediate potential* is not specified, then fit will start from empty potential. 
Alternatively, you can specify *initial or intermediate potential* by command-line option

`pacemaker ... -ip some_start_or_interim_potential.yaml `

* Specify `ladder_step` in `fit` section:
```yaml
fit:

    ...

  ladder_step: [10, 0.02]       # Possible values:
                                #  - integer >= 1 - number of basis functions to add in ladder scheme,
                                #  - float between 0 and 1 - relative ladder step size wrt. current basis step
                                #  - list of both above values - select maximum between two possibilities on each iteration 
```

See `example/ladder_fit_pyace.yaml` and  `example/ladder_fit_tensorpot.yaml` example input files

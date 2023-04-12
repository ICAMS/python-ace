# Quick start

Running a fit with `pacemaker` requires at least two components: fitting dataset and configurational input file. Fitting dataset
contains structural information as well as corresponding energies and forces that are subject to fitting with ACE. 
Input file contains details about desired ACE potential configuration and various parameters influencing optimization 
process.  
In this section we will describe the format of the fitting dataset, we will run a fit with an example dataset and
overview the output produced by `pacemaker`. Input parameters are detailed in the [section](inputfile.md#Input_file) below.    

## Automatic DFT data collection

You can collect DFT calculations (currently only for VASP from `vasprun.xml` or `OUTCAR` files) by using `pace_collect`
utility. For example, if your data is in `my_dft_calculation/` folder and subfolders, and single atoms reference energies
are -0.123 eV for Al and  -0.456 eV for Cu, then run command:
```
pace_collect -wd path/to/my_dft_calculation --free-atom-energy Al:-0.123 Cu:-0.456
```
If you have free atom calculations (single atom in large volume) in subfolders, then it could be used as referenced with the following command:
```
pace_collect -wd path/to/my_dft_calculation --free-atom-energy auto 
```
Both commands will scan through all folders and subfolders and collect DFT free energies (that are force-consistent) and forces 
and make a single atom corrections. Resulting dataset will be stored into `collected.pkl.gz` file.

If you need more flexibility for DFT dataset manipulation,
please check [Manual fitting dataset preparation](#manual_fitting_dataset_preparation).

## Automatic input file generation

In order to fit an ACE potential, one need to create a configurational  file with relevant settings. 
`pacemaker` utilizes `.yaml` format for configurations. 

In order to interactively generate default `pacemaker` input file `input.yaml`, please run 
```
pacemaker -t
```
and enter requested information, such as dataset filename, test set size (optional), list of elements, cutoff,
number of functions.  Doing so will produce an `input.yaml` file with the most general
settings that can be adjusted for a particular task. Detailed overview of the input file parameters can be found in the [Input file](inputfile.md) section below.

## (optional) Manual fitting dataset preparation

In order to use your data for fitting with `pacemaker` one would need to provide it in the form of `pandas` DataFrame.
An example DataFrame can be red as:

```python
import pandas as pd
df = pd.read_pickle("../data/exmpl_df.pkl.gz")
```
And it contains the following entries:


|    |   energy | forces            | ase_atoms                                                                                                             |   energy_corrected |
|---:|---------:|:------------------|:----------------------------------------------------------------------------------------------------------------------|-------------------:|
|  0 | -3.69679 | [[0.0, 0.0, 0.0]] | Atoms(symbols='Al', pbc=True, cell=[[0.0, 1.949947, 1.949947], [1.949947, 0.0, 1.949947], [1.949947, 1.949947, 0.0]]) |           -3.69679 |
|  1 | -3.71569 | [[0.0, 0.0, 0.0]] | Atoms(symbols='Al', pbc=True, cell=[[0.0, 1.964285, 1.964285], [1.964285, 0.0, 1.964285], [1.964285, 1.964285, 0.0]]) |           -3.71569 |
|  2 | -3.72955 | [[0.0, 0.0, 0.0]] | Atoms(symbols='Al', pbc=True, cell=[[0.0, 1.978417, 1.978417], [1.978417, 0.0, 1.978417], [1.978417, 1.978417, 0.0]]) |           -3.72955 |
|  3 | -3.7389  | [[0.0, 0.0, 0.0]] | Atoms(symbols='Al', pbc=True, cell=[[0.0, 1.99235, 1.99235], [1.99235, 0.0, 1.99235], [1.99235, 1.99235, 0.0]])       |           -3.7389  |
|  4 | -3.74421 | [[0.0, 0.0, 0.0]] | Atoms(symbols='Al', pbc=True, cell=[[0.0, 2.006091, 2.006091], [2.006091, 0.0, 2.006091], [2.006091, 2.006091, 0.0]]) |           -3.74421 |


 - Columns have the following meaning:
    - `ase_atoms`: is the instance of the [ASE](https://wiki.fysik.dtu.dk/ase/) Atoms class. This is the main form of storing structural information 
    that `pacemaker` relies on. It must contain information about atomic positions, corresponding atom types, pbc and lattice vectors.
    - `energy`: total energy of the corresponding `ase_atoms` structure (in eV).
    - `forces`: corresponding atomic forces in the form of 2D array with dimensions [NumberOfAtoms, 3] (in eV/A).
    - `energy_corrected`: total energy of a structure minus a reference energy.
    
    Reference energy might be different depending on the
    dataset at hand. In general, one would prefer to reference `energy` against the free atom energies. In this case `energy_corrected` 
    corresponds to the cohesive energy. If the free atom energies are not available, reference energy might be any constant shift or 0.
    In this example `energy` is already the cohesive energy.  
    NOTE: regardless how `energy_corrected` is produced, *this is the energy that will be used for fitting*.    

One could create such DataFrame from raw data following this example:

```python
import pandas as pd
from ase import Atoms

# Collect raw data for the first structure
# Positions
pos1 = [[2.04748516, 2.04748516, 0.        ],
       [0.        , 0.        , 0.        ],
       [2.04748516, 0.        , 1.44281847],
       [0.        , 2.04748516, 1.44475745]]
# Matrix of lattice vectors
lattice1 = [[4.09497 , 0.      , 0.      ],
       [0.      , 4.09497 , 0.      ],
       [0.      , 0.      , 2.887576]]
# Atomic symbols
symbls1 = ['Al', 'Al', 'Ni', 'Ni']
# energy
e1 = -21.07723361
# Forces
f1 = [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.00725587],
     [0.0, 0.0, -0.00725587]]
# create ASE atoms
at1 = Atoms(symbols=symbls1, positions=pos1, cell=lattice1, pbc=True)

#Collect raw data for the second structure
pos2  = [[0., 0., 0.]]
lattice2 = [[0.      , 1.781758, 1.781758],
           [1.781758, 0.      , 1.781758],
           [1.781758, 1.781758, 0.      ]]
symbls2 = ['Ni']
e2 = -5.45708644
f2 = [[0.0, 0.0, 0.0]]
at2 = Atoms(symbols=symbls2, positions=pos2, cell=lattice2, pbc=True)

# set reference energy to 0
reference_energy = 0
# collect all the data into a dictionary 
data = {'energy': [e1, e2], 
        'forces': [f1, f2], 
        'ase_atoms': [at1, at2], 
        'energy_corrected': [e1 - reference_energy, e2 - reference_energy]}
# create a DataFrame
df = pd.DataFrame(data)
# and save it 
df.to_pickle('my_data.pkl.gz', protocol=4)
```

or use the utility `pace_collect` from a top-level directory to collect VASP calculations and store them in a 
`collected.pkl.gz` file.
The resulting dataframe can be used for fitting with `pacemaker`.

### Creating an input file
 
In this example we will use template as it is, however one would need to provide a path to the
example dataset `exmpl_df.pkl.gz`. This can be done by changing `filename` parameter in the `data` section of the 
`input.yaml`:

```yaml
data:
   filename: /path/to/the/pyace/data/exmpl_df.pkl.gz
```

Please check [examples folder](https://github.com/ICAMS/python-ace/tree/master/examples) for more examples of input file.

## Run fitting

Running a fit is as easy as executing the command:

```
pacemaker input.yaml
``` 
or to run the fitting process in the background:
```
nohup pacemaker input.yaml &
```
For more `pacemaker` command options see the corresponding [CLI](cli.md).  

Default behavior of pacemaker is to utilize a GPU accelerated fitting of ACE using `tensorpotential`. However, 
parallelization over multiple GPU is not supported at the moment. Therefore, if your machine has a multi GPU setup one would need to select
a single one before running `pacemaker`. This can be done by executing  `export CUDA_VISIBLE_DEVICES=ind` in the shell
replacing `ind` with the GPU index (i.g. 0, 1, ...) or -1 to disable GPU usage.  
Note, that `tensorpotential` can be used without a GPU as well.

## Analysis

During and after the fitting `pacemaker` produces several outputs, including:

- `interim_potential_X.yaml`: current state of the potential at each iteration of [fit cycle](inputfile.md#fitting-settings) (i.g. X=0, 1, ...)
- `interim_potential_best_cycle.yaml`: best out of X interim potentials
- `log.txt`: log file containing all current information including summary of the optimization steps.
- `report`: folder containing figures displaying various error statistics and distributions. 
- `output_potential.yaml`: final fitted potential.

There are two main types of the information in the log file:

- optimization step log: 

```text
Iteration   #999  (1052 evals):     Loss: 0.000192 | RMSE Energy(low): 17.95 (16.79) meV/at | Forces(low): 7.89 (7.04) meV/A | Time/eval: 517.83 mcs/at
```
where `Iteration` is the index of the optimization step performed by the [optimizer](inputfile.md#fitting-settings)
(number in parentheses shows the number of function evaluation calls done by optimizaer), `Loss` 
is the current value of the loss function, `RMSE Energy/Forces` is the current root mean-squared error 
for energy/forces wrt. training dataset (numbers in paretheses show corresponding values for the structures which
energy is not greater than `e_min + 1 eV`, where `e_min` is the lowest energy in the training set). `Time/eval` 
shows the computational time spent on evaluating loss function and it's gradient for the training dataset
averaged across evaluations and divided by the number of atoms. This timing doesn't include optimization step itself.

- fit statistics:  

```text
--------------------------------------------FIT STATS--------------------------------------------
Iteration:  #1000Loss:    Total:  1.9159e-04 (100%) 
                        Energy:  1.6074e-04 ( 84%) 
                        Force:  3.0859e-05 ( 16%) 
                            L1:  0.0000e+00 (  0%) 
                            L2:  0.0000e+00 (  0%) 
Number of params./funcs:    232/86                                   Avg. time:     526.93 mcs/at
-------------------------------------------------------------------------------------------------
            Energy/at, meV/at   Energy_low/at, meV/at      Force, meV/A        Force_low, meV/A   
    RMSE:          17.93                16.73                 7.86                    7.06
    MAE:           12.22                11.11                 5.31                    3.30
    MAX_AE:        53.19                38.30                35.19                   20.32
-------------------------------------------------------------------------------------------------

```


Every [display_step](inputfile.md#backend-specification) the summary of fit statistics is printed out. It displays the total 
loss function value and contributions to it from energy, forces and other [regularizations parameters](inputfile.md#fitting-settings).
In addition to RMSE, mean-absolute error (MAE) and maximum absolute error (MAX_AE) are also printed.


## Using fitted potential

Fitted potential can be used for calculations both in [LAMMPS](https://docs.lammps.org/latest/pair_pace.html) as well as  within [python/ASE](https://wiki.fysik.dtu.dk/ase/).



### LAMMPS

Using potential with [LAMMPS](https://www.lammps.org/) requires its [conversion](utilities.md#potential-conversion) into **YACE** format with command
```asm
pace_yaml2yace output_potential.yaml
```
that will generate `output_potential.yace` file, which you could use in LAMMPS input file
```
## in.lammps

pair_style  pace 
pair_coeff  * * output_potential.yace Al Ni
```
See more details on [pair_style pace](https://docs.lammps.org/latest/pair_pace.html)

#### LAMMPS compilation:

You could get the supported version of LAMMPS from [GitHub repository](https://github.com/lammps/lammps)

##### Build with `make`

Follow LAMMPS installation instructions

1. Go to `lammps/src` folder
2. Compile the ML-PACE library by running `make lib-pace args="-b"`
3. Include `ML-PACE` in the compilation by running `make yes-ml-pace`
4. Compile lammps as usual, i.e. `make serial` or `make mpi`.

##### Build with `cmake`


1. Create build directory and go there with 

```
cd lammps
mkdir build
cd build
```

2. Configure the lammps build with

```
cmake -DCMAKE_BUILD_TYPE=Release -DPKG_ML-PACE=ON ../cmake 
```

or 

```
cmake -DCMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -DPKG_ML-PACE=ON ../cmake
```

For more information see [here](https://lammps.sandia.gov/doc/Build_cmake.html).

   
3. Build LAMMPS using `cmake --build .` or `make`


Please note, that there is a KOKKOS implementation of PACE for LAMMPS as `pair_style pace/kk`, but you need to compile
LAMMPS with this support, see official documentation [here](https://docs.lammps.org/Build_extras.html#kokkos).
This implementation allows to run calculations on GPU which give the speedup of **up to x100** on modern GPU architectures 
in comparison to single-core CPU. In that case you should modify LAMMPS input script as 
```
## in.lammps

pair_style  pace product 
pair_coeff  * * output_potential.yace Al Ni
```

### ASE

Python interface of the ACE potential is realized via [ASE](https://wiki.fysik.dtu.dk/ase/) calculator:
```python
from ase.build import bulk
from pyace import PyACECalculator

# Create simple ASE atoms structure
atoms = bulk('Al', cubic=True)

# Create calculator
calc = PyACECalculator('output_potential.yaml')

# Attach it to the Atmos
atoms.set_calculator(calc)

# Evaluate properties
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Check more properties
calc.results
```

## More examples

Please, find more examples in `python-ace/examples` folder and subfolders.

# Frequently asked questions (FAQ)

## What is a good value for batch_size?
In order to achieve better fitting performance large *batch_size* (i.e. 100 or 1000) is recommended. 
If `batch_size_reduction=True` (default option), then automatic batch size reduction will happen and you could start from initial large *batch_size* value.

## My fit on GPU crushes with OOM error, what to do ?
This means that you are trying to fit at once too much data into the GPU memory. The amount of data processed by GPU
at once is controlled by [batch_size](#backend-specification) parameter, try to reduce it or set `batch_size_reduction=True`. An optimal value for
this parameter is totally empirical as it depends on data, potential configuration and GPU itself.


## Can I toggle between CPU and GPU when starting pacemaker?
If you have GPU configured on your machine it will be used by default.
You can have additional control over GPU configuration via  `input.yaml::backend::gpu_config`:
```yaml
backend:
  gpu_config: {gpu_ind: <int>, mem_limit: <int>}
```
 - `gpu_ind`: index of the GPU you want to use for fitting. This need to be specified in case your
machine has multiple GPUs (multi GPU fitting is not supported at the moment). Set this parameter to -1 to disable GPU utilization.
Default is 0.
 - `mem_limit`: maximum amount of GPU memory in MB that is allowed to be used by fitting process. Default is 0 which 
allows to consume the whole available memory.  
NOTE: memory reserved by the fitting process is not available to anything else. Therefore, it's recommended to set this 
restriction if you also use the same machine for processes requiring GUI.

## I dont have a GPU. Should I use backend, evaluator= `tensorpotential` or `pyace`?
It is recommended to use `tensorpotential` evaluator for fitting anyways. Even without GPU acceleration non-linear 
optimization greatly benefits from autogradients provided by TensorFlow.

## How to continue fitting ?
Fitting an ACE potential can be continued or restarted from any `.yaml` potential file produced previously.  
If you want to continue fit without changing the basis size, you can do the following:

-  Provide the path to the starting potential in corresponding field in the `input.yaml` file
  
  ```yaml
  potential: /path/to/your/potential.yaml
  ```
 
- or provide this path through the command line interface
  
  ```
  pacemaker input.yaml -p /path/to/your/potential.yaml
  ```  
    
  doing this will override specifications in the `input.yaml`.

If you want to extend the basis (aka do the [ladder scheme fitting](#inputfile.md#ladder_hiererchical_basis_extension)):
 
- Specify your potential as initial potential
  
  ```yaml
  potential:
         ...
         initial_potential: /path/to/your/potential.yaml
         ...
  ```

- or use the CLI:

  ```
  pacemaker input.yaml -ip /path/to/your/potential.yaml
  ```

## I want to preserve the "shape" of potential, but refit it from scratch

```yaml
#input.yaml

potential:
  filename: /path/to/your/potential.yaml
  reset: true
```

It will  reset potential from potential.yaml, i.e. set radial coefficients to delta_nk and B-basis function coefficients to zero.

## What is the best value for the relative force weight in the loss function, `kappa`?

The default value of `kappa: 0.3` in the loss function `fit:: loss: {kappa: 0.3, L1_coeffs: 1e-8, L2_coeffs: 1e-8,}` provides a good initial guess.

Additionally, there is an option to use `kappa: auto`, which analyzes the distribution (standard deviation) of training energies and forces to balance the contributions of these two factors in loss function.

However, you can try using the following two-stage scheme to improve both energy and force metrics:

1) Initially, perform the fit with a very high weight on forces, such as `kappa: 0.95` or `kappa: 0.99`.

2) Then, switch to a high weight on energies, ranging from `kappa: 0.05` to `kappa: 0.1`. To do this, you need to continue the fitting process starting from the last version of the potential from the previous step.

```
pacemaker input.yaml # step 1 with kappa: 0.95
cp interim_potential_0.yaml cont.yaml
# or cp output_potential.yaml cont.yaml
pacemaker input_k-0.1.yaml -p cont.yaml # step 2 with kappa: 0.05 .. 0.1
```

In the second step, the force metrics may initially increase for the first few dozen iterations, but they should eventually revert and decrease. 
On the other hand, the energy metrics should decrease immediately. If this is not the case, you can try slightly increasing the value of kappa.

## My potential behaves unphysical at short distances, how to fix it?

If training data lacks data at shorter distances, expected repulsive behaviour is not always reproduced.
In order to avoid it, you should use core-repulsion potential when you define the potential in `input.yaml` 
which replaces ACE potential with an exponential repulsion:

```yaml
## input.yaml

potential:
  embeddings:
    ALL: {
      ...
      # core repulsion parameters
      rho_core_cut: 5,
      drho_core_cut: 5
      ...
    }

  bonds: 
    ALL: {
      ## inner cutoff, applied in a range [r_in - delta_in, r_in]
      r_in: 2.3, # distance, below which the core repulsion start
      delta_in: 0.1,

      ## core-repulsion parameters `prefactor` and `lambda` in
      ## prefactor*exp(-lambda*r^2)/r, >0 only r<r_in+delta_in
      core-repulsion: [1e3, 1.0],
    }
```
If you did not specify it before the fit, you still could add it after with Python API:
```python
from pyace import *

bbasisconf = BBasisConfiguration("original_potential.yaml")

for block in bbasisconf.funcspecs_blocks:
    block.r_in = 2.3 # minimal interatomic distance in dataset
    block.delta_in = 0.1
    block.core_rep_parameters=[1e3, 1.0]
    block.rho_cut = block.drho_cut = 5
bbasisconf.save("tuned_potential.yaml")
```
or by manually changing corresponding parameters in `original_potential.yaml` file.

**NOTE** However, it is strongly recommended to add more data, that describe the behaviour ar short distances rather than relying on the core repulsion completely.

## How to split train/test data for the fitting?
Just use `test_size` keyword in `input.yaml::data`:

```yaml
data:
  test_size: 0.1 # for 10% of data used for testing
```
Alternatively, you can provide train and test datasets separately:

```yaml
data:
  filename: /path/to/train_data.pckl.gzip    
  test_filename: /path/to/test_data.pckl.gzip 
``` 

## I want to change the cutoff, what should I do ?

If you change cutoff, i.e. from `rcut: 7` to `rcut: 6.5`, then potential should be refitted from the scratch.
`pacemaker` will recompute neighbourlists on every run, so, no need to extra options except for specifying cutoff. 



## How better to organize my dataset files ?

It is recommended to store all dataset files (i.e. `df*.pckl.gzip`) in one folder and
specify the environment variable `$PACEMAKERDATAPATH` (exectue it in terminal or add to for example `.bashrc`) 

```
export PACEMAKERDATAPATH=/path/to/my/dataset/files
```  

## What are good values for regularization parameters ?

Ideally, one would prefer avoid using regularizations and would use additional data instead. When this is not possible, 
it is recommended that relative contribution of the regularization terms into the total loss do not exceed a few percents. 
So, regularization parameters of order **1e-5 ~ 1e-8** are good initial values, but check their relative contribution in 
detailed statistics, printed every `input.yaml::backend::display_step` step.

## How to fit only certain part of the potential, i.e. binary interaction only ?
If you have already fitted potential `Al.yaml` and `Ni.yaml` and would like to create a binary potential by fitting
to binary data, i.e. AlNi structures, then in `input.yaml::potential` you could provide only binary interaction parts:

```yaml
potential:
  deltaSplineBins: 0.001,
  elements: [Al, Ni],
  bonds: {
        BINARY: {
            rcut': 6.2,
            dcut': 0.01,
            core-repulsion': [0.0, 5.0],
            radbase': ChebExpCos,
            radparameters': [5.25]
    }
  }
  functions: {
      BINARY: {
        nradmax_by_orders:  [5, 2, 2, 1],
        lmax_by_orders: [0, 2, 2, 1],
        }
  }
  
  ## provide list of initial potentials
  initial_potential: [Al.yaml, Ni.yaml]
```

and in `input.yaml::fit` you add 

```yaml
fit:
  ...
  trainable_parameters: BINARY
  ...
```

## I see different metrics text files during the fit, what is it ?

All metrics files contain values of loss function (*loss*), its energy/forces contributions (*e_loss_contrib*, *f_loss_contrib*),
regularization contributions (*reg_loss*) and also root mean squared error (RMSE)/ mean absolute error (MAE) (*rmse_*, *mae_*)
of energies (*rmse_epa*,*mae_epa*) and forces (norm of error vector *rmse_f* and per-component *mae_f_comp*)  for whole dataset
as well as for structures within 1eV/atom above minumum (*low_**).  
`metrics.txt` and `test_metrics.txt` are update every train/test step, whereas `ladder_metrics.txt`/`test_ladder_metrics.txt`
are updated after each ladder step and `cycle_metrics.txt`/`test_cycle_metrics.txt` are updated after each cycle on ladder step.


## Optimization stops too early due to too small updates, but I want to run it longer...

You need to decrease certain tolerance parameters for corresponding minimization algorithm. 
For example, for [BFGS](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html), there is `gtol: 1e-5` 
default parameter, that you could decrease in `input.yaml`
```yaml
fit:
  options: {gtol: 5e-7}
```

## How to create a custom weights dataframe for ExternalWeightingPolicy? How to add more weights to certain structures ?

Please check [this](https://github.com/ICAMS/python-ace/blob/master/examples/custom-weights/data_custom_weights.ipynb) example notebook.

## How to compute B-basis projections for various structures?

If you have  ACE potential (fitted or just constructed from scratch), then you can compute the B-basis projections for all atoms in your structure(s).
Please check [this](https://github.com/ICAMS/python-ace/blob/master/examples/pyace/bbasis_projections.ipynb) example notebook.

## What is early stopping and how to use it?

Early stopping is a technique used to prevent overfitting in the ML model.
The main idea is to halt the optimization procedure once the loss/metrics on the test set stops improving. 
Additional criteria, such as slow improvement in the training loss, can be employed to prevent the unnecessary expenditure 
of computational resources. In pacemaker, you can activate these early stopping criteria by including the following lines
into `input.yaml`:
```yaml
fit:
  # Early stopping
  min_relative_train_loss_per_iter: 5e-5
  min_relative_test_loss_per_iter: 1e-5
  early_stopping_patience: 200
```
The parameters `min_relative_train_loss_per_iter` and `min_relative_test_loss_per_iter` characterize the minimal relative 
change in the train and test loss per iteration, respectively (in absolute value). 
The relative loss at iteration t is computed as:
```
relative_loss_per_iter(t) = (loss(t) - loss(t - N)) / loss(t - N) / N
```
Here, `N` represents the number of iterations between the computation of loss functions (`N=1` for train loss and `N=display_step`
for test loss). `early_stopping_patience` characterizes the minimal number of iterations before stopping can occur.
In a normal situation, `relative_loss_per_iter` should be negative and large in absolute value.

Stopping occurs when **any** of the following criteria are met *each* step over the last `early_stopping_patience` steps:
* `relative_TRAIN_loss_per_iter > -|min_relative_train_loss_per_iter|`
* `relative_TEST_loss_per_iter > -|min_relative_test_loss_per_iter|`
 
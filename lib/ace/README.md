# About

This is the (p)erformant C/C++ implementation of the (A)tomic (C)luster (E)xpansion 
 or shortly "pace" with both **B- and C-tilde basises**.

# WARNING

This repository is a part of `pyace` package now, please use this repository!


# Git work cycle 

 ### Cloning

  Clone new project
    
    git clone --recurse-submodules https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/ace.git

 ### Update existing local repo
   
    git pull  --recurse-submodules
 
 ### Pushing updates
 
 As `ace-evaluator` is a submodule (i.e. git repository inside another git repository), commiting and pushing
 should be done in two stages:
  * If you are doing modifications in the `ace-evaluator` submodule
  
         cd ace-evaluator
         git checkout <branch>
    
   ... do some modifications ...
   
        git commit -a -m "commit in submodule"
        git push
    
  * Update root `ace` repository
  
        cd ..
        git add ace-evaluator
        git commit -m "Updated submodule"
      
# Compilation

1. Clone and compile YAML-CPP
   
    * `cd utils`
    
    * `./compile_yaml.sh`
2. Compile <a name="yaml2ace">`yaml2ace`</a>
    * `mkdir build`
    * `cd build`
    * `cmake -DCMAKE_BUILD_TYPE=Release ..`
    * `make yaml2ace`
        
3. (optional) Compile and run tests
    * `cd ace-evaluator/google-tests`
    * `./compile_googletest.sh`
    * `cd ..`
    * `mkdir build; cd build`
    * `cmake -DCMAKE_BUILD_TYPE=Release ..`
    * `make test_extended`
    * `cd ../test`
    * `../build/test_extended  --gtest_filter=* --gtest_color=yes`


# Directories structure

* **ace-evaluator/**

    ACE evaluator core (as a _git submodule_)

* **benchmark/**

    (_currently outdated_) Google Benchmark and benchmark tests .

* **doc/**

    (_not complete_) Automatically generated documentation.

* **potentials/**

    * **pace/**
    
         Examples of C-tilde basis set potentials
         
    * **yaml/**
    
         Examples of B-basis set potentials in YAML format
         
    * **yaml-files-prototypes/**
    
        Documented examples of B-basis set potentials in YAML file format
        
* **src/**

    Common header and source files for advanced functionality of ACE implementation.
    * **fitting/**
    
    Header and source files for Clebsch-Gordan coefficients, coupling schemes, ACE_B_to_CTildeBasisSet, YAML and B-basis set
    
    * **utils/**
    
* **test/**

    Test for C(B)-tilde ACE implementation from YAML format and corresponding potentials

* **utils/**

    Third party libraries and frameworks, like  YAML-parser in C++.
     
    * **compile_yaml.sh** and (optional) **yaml-cpp/**
    
        Script to clone latest version of YAML-CPP framework from GitHub to **yaml-cpp/** and compile it.

    * **coupling-constants-generation/**
    
        Example input file for coupling constants generation utility. See `README` in there for more details.

    * **multiarray/generate_multiarray_headers.py**
    
        Automatic C++ header generation tool for multi-dimensional contiguous array implementation. See `README` in there for more details.
    
    *  <a name="fortran2yaml">**potentials-conversion/fortran2yaml.py**</a>
    
        Python utility for Fortran to YAML (both B-basis set) potential conversion. See `README` in there for more details.

# Fortran to ACE/C++ potential coversion

1. Use [`fortran2yaml.py`](#fortran2yaml) to convert Fortran ACE potential to B-basis YAML
2. Use [`yaml2ace`](#yaml2ace) to convert B-basis to C-tilde basis potential.

# About

This is the (p)erformant C/C++ implementation of the (A)tomic (C)luster (E)xpansion 
 or shortly "pace" in C-tilde basis.

# Directories structure

* extra/
    
    Header and source files for extra functionality for tests

* google-tests/
    
    directory for Google Test. 
    
    Please run ./compile_googletest.sh to automatically clone and compile Google Test

* src/

    Common header and source files for C-tilde ACE implementation

* test
 
    Tests for C-tilde ACE implementation

   * test/potentials:     Potentials for tests

* .gitlab-ci.yml

    settings for automatic testing on GitLab CI/CD platform
    
# Compiling and running tests
        
To run the tests manually:

1. Download and compile _Google Test_.  

        cd google-tests
        ./compile_googletest.sh  
2. Create **build** folder in _repo root_

        mkdir build
3. Compile tests

        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make test_evaluator -j
4. Run tests. From _repo root_

        cd test
        ../build/test_evaluator  --gtest_filter=* --gtest_color=yes

# Building documentation

The automatic documentation could be built with Doxygen (http://www.doxygen.nl/index.html)

    cd doc
    doxygen Doxyfile
    cd latex
    make       
    
will result in `doc/latex/refman.pdf` file
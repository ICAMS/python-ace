# wigner-cpp

C++ templated library for Wigner 3*j*, 6*j*, 9*j* symbols and Gaunt coefficients

## Description

This library is for the numerical computation of special functions appearing in
angular monentum coupling theory:

 - Wigner 3-*j* symbol ([Wikipedia](https://en.wikipedia.org/wiki/3-j_symbol),
   [Wolfarm](http://mathworld.wolfram.com/Wigner3j-Symbol.html))
 - Wigner 6-*j* symbol ([Wikipedia](https://en.wikipedia.org/wiki/6-j_symbol),
   [Wolfarm](http://mathworld.wolfram.com/Wigner6j-Symbol.html))
 - Wigner 9-*j* symbol ([Wikipedia](https://en.wikipedia.org/wiki/9-j_symbol),
   [Wolfarm](http://mathworld.wolfram.com/Wigner9j-Symbol.html))
 - Gaunt coefficient

Wigner 3-*j* and 6-*j* symbols and Gaunt coefficiens are calculated using the
three-term recurrence relations originally derived by Schulten and Gordon
[^Schulten1975] and improved to avoid overflows by Luscombe and Luban
[^Luscombe1998].


## Requirement

You need a newer C++ compiler that supports the C++11 standard, such as
GCC (>= 4.8.0) and Clang (>= 3.2).

This library depends only C++ standard libraries: no external library is
required.


For building unit tests, [Catch](https://github.com/squared/Catch) testing
framework is required. This step is optional.


## Install

`wigner-cpp` is a header only library. You can use it by including header files
under `wigner` directory.

## Usage
See, example program in `example` directory.

## Licence

Copyright (c) 2016 Hidekazu Ikeno

Released under the [MIT license](http://opensource.org/licenses/mit-license.php)


## References

[Schulten1975]: Klaus Schulten and Roy G. Gordon, "Exact recursive evaluation
 of 3j and 6j-coefficients for quantum-mechanical coupling of angular J
 momenta," J. Math. Phys. 16, pp 1961-1970 (1975).

[Luscombe1998]: James H. Luscombe and Marshall Luban, "Simplified recursive
 algorithm for Wigner 3j and 6j symbols," Phys. Rev. E 57, pp. 7274-7277 (1998).

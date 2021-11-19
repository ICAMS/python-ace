//
// Created by Lysogorskiy Yury on 11.05.2020.
//

#ifndef PYACE_ACE_RADIAL_HELPER_H
#define PYACE_ACE_RADIAL_HELPER_H

#include <pybind11/pytypes.h>

#include "ace_radial.h"

pybind11::tuple ACERadialFunctions_getstate(const AbstractRadialBasis *radial_functions);

ACERadialFunctions *ACERadialFunctions_setstate(const pybind11::tuple &t);

#endif //PYACE_ACE_RADIAL_HELPER_H

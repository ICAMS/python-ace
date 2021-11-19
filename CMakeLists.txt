cmake_minimum_required(VERSION 3.7)

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)
project(pyace C CXX)

set(CMAKE_CXX_STANDARD 11)

# Set source directory
#------------------------------------------------------
set(SOURCE_DIR "src/pyace")
set(ACE_DIR "lib/ace")

#Headers
#-------------------------------------------------------
#include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR}/ace)
include_directories(${SOURCE_DIR}/ace-evaluator)
include_directories(${SOURCE_DIR}/utils)

include_directories(${ACE_DIR}/src/fitting)
include_directories(${ACE_DIR}/ace-evaluator/src)
include_directories(${ACE_DIR}/ace-evaluator/extra)


#   Add external libraries
#-------------------------------------------------------

#add_subdirectory(lib/yaml-cpp)
set(YAML_INCLUDE_PATH lib/ace/ace-evaluator/lib/yaml-cpp/include)
include_directories(${YAML_INCLUDE_PATH})

set(WIGNER_PATH lib/ace/utils/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include/wigner)
include_directories(${WIGNER_INCLUDE_PATH})

add_subdirectory(lib/pybind11)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

# for advanced functionality
# TODO: check speed influence of this flag
add_definitions(-DEXTRA_C_PROJECTIONS)
#add_definitions(-DDEBUG_FORCES_CALCULATIONS)
#add_definitions(-DMULTIARRAY_INDICES_CHECK)
add_subdirectory(lib/ace)

#Now add sources
#--------------------------------------------------------
set(SOURCES_SHARMONICS "${ACE_DIR}/src/fitting/ace_couplings.cpp"
            "${ACE_DIR}/src/fitting/ace_spherical_polar.cpp"
            "${SOURCE_DIR}/ace/ace_spherical_polar_binding.cpp"
            )

set(SOURCES_COUPLING "${ACE_DIR}/src/fitting/ace_couplings.cpp"
        "${ACE_DIR}/src/fitting/ace_clebsch_gordan.cpp"
        "${SOURCE_DIR}/ace/ace_coupling_binding.cpp")

set(SOURCES_BASIS "${ACE_DIR}/ace-evaluator/src/ace_c_basis.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_radial.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_spherical_cart.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_abstract_basis.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_flatten_basis.cpp"
        "${ACE_DIR}/ace-evaluator/src/ships_radial.cpp"
        "${ACE_DIR}/src/fitting/ace_b_basis.cpp"
        "${ACE_DIR}/src/fitting/ace_b_basisfunction.cpp"
        "${ACE_DIR}/src/fitting/ace_clebsch_gordan.cpp"
        "${ACE_DIR}/src/fitting/ace_couplings.cpp"
        "${ACE_DIR}/src/fitting/ace_yaml_input.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_c_basis_binding.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_bbasis_spec_helper.cpp"
        "${SOURCE_DIR}/ace/ace_radial_helper.cpp"
        "${SOURCE_DIR}/ace/ace_c_basisfunction_helper.cpp"
        "${SOURCE_DIR}/ace/ace_c_basis_helper.cpp"
  )

set(SOURCES_EVALUATOR "${ACE_DIR}/ace-evaluator/src/ace_c_basis.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_abstract_basis.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_flatten_basis.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_evaluator.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_recursive.cpp"
             "${ACE_DIR}/ace-evaluator/extra/ace_atoms.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_radial.cpp"
             "${ACE_DIR}/ace-evaluator/src/ace_spherical_cart.cpp"
             "${ACE_DIR}/src/fitting/ace_b_evaluator.cpp"
             "${ACE_DIR}/src/fitting/ace_b_basis.cpp"
             "${ACE_DIR}/src/fitting/ace_clebsch_gordan.cpp"
             "${ACE_DIR}/src/fitting/ace_yaml_input.cpp"
             "${ACE_DIR}/src/fitting/ace_couplings.cpp"
             "${SOURCE_DIR}/ace-evaluator/ace_evaluator_binding.cpp"
        )

set(SOURCES_CATOMICENVIRONMENT "${ACE_DIR}/ace-evaluator/extra/ace_atoms.cpp"
            "${SOURCE_DIR}/utils/ace_atoms_binding.cpp")


set(SOURCES_CALCULATOR
        "${ACE_DIR}/ace-evaluator/src/ace_radial.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_abstract_basis.cpp"
        "${ACE_DIR}/ace-evaluator/extra/ace_atoms.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_spherical_cart.cpp"
        "${ACE_DIR}/ace-evaluator/extra/ace_calculator.cpp"
        "${ACE_DIR}/ace-evaluator/src/ace_evaluator.cpp"
        "${SOURCE_DIR}/utils/ace_calculator_binding.cpp")


# C++ FLAGS
#---------------------------------------------------------
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-Ofast -DNDEBUG")

#Finally create the package
#--------------------------------------------------------
pybind11_add_module(sharmonics ${SOURCES_SHARMONICS} )

pybind11_add_module(coupling ${SOURCES_COUPLING})
target_link_libraries(coupling PRIVATE yaml-cpp-pace)


pybind11_add_module(catomicenvironment ${SOURCES_CATOMICENVIRONMENT} )

pybind11_add_module(basis ${SOURCES_BASIS} )
target_link_libraries(basis PRIVATE yaml-cpp-pace)

pybind11_add_module(evaluator ${SOURCES_EVALUATOR}  ${SOURCES_BASIS} )
target_link_libraries(evaluator PRIVATE yaml-cpp-pace)

pybind11_add_module(calculator ${SOURCES_CALCULATOR} )



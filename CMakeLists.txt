cmake_minimum_required(VERSION 3.7)

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)
project(pyace C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE True)
# Set source directory
#------------------------------------------------------
set(SOURCE_DIR "src/pyace")
set(ACE_ROOT "lib/ace")
set(ACE_SRC ${ACE_ROOT}/src/ace)
set(ACE_INCLUDE ${ACE_ROOT}/src)

set(ACE_EVALUATOR_ROOT ${ACE_ROOT}/ace-evaluator)
set(ACE_EVALUATOR_SRC ${ACE_EVALUATOR_ROOT}/src/ace-evaluator)
set(ACE_EVALUATOR_EXTRA ${ACE_EVALUATOR_ROOT}/src/extra)
set(ACE_EVALUATOR_INCLUDE ${ACE_EVALUATOR_ROOT}/src)
#Headers
#-------------------------------------------------------
#include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR}/ace)
include_directories(${SOURCE_DIR}/ace-evaluator)
include_directories(${SOURCE_DIR}/utils)

include_directories(${ACE_INCLUDE})
include_directories(${ACE_EVALUATOR_INCLUDE})
#include_directories(${ACE_DIR}/ace-evaluator/extra)


#   Add external libraries
#-------------------------------------------------------

set(YAML_INCLUDE_PATH lib/ace/ace-evaluator/lib/yaml-cpp/include)
include_directories(${YAML_INCLUDE_PATH})

set(WIGNER_PATH lib/ace/utils/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include)
include_directories(${WIGNER_INCLUDE_PATH})

set(CNPY_PATH lib/ace/utils/cnpy)
set(CNPY_INCLUDE_PATH lib/ace/utils/)
include_directories(${CNPY_INCLUDE_PATH})

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
set(SOURCES_SHARMONICS "${ACE_SRC}/ace_couplings.cpp"
            "${ACE_SRC}/ace_spherical_polar.cpp"
            "${ACE_SRC}/ace_clebsch_gordan.cpp"
            "${SOURCE_DIR}/ace/ace_spherical_polar_binding.cpp"
            )

set(SOURCES_COUPLING "${ACE_SRC}/ace_couplings.cpp"
        "${ACE_SRC}/ace_clebsch_gordan.cpp"
        "${SOURCE_DIR}/ace/ace_coupling_binding.cpp")

set(SOURCES_BASIS "${ACE_EVALUATOR_SRC}/ace_c_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ace_radial.cpp"
        "${ACE_EVALUATOR_SRC}/ace_spherical_cart.cpp"
        "${ACE_EVALUATOR_SRC}/ace_abstract_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ace_flatten_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ships_radial.cpp"
        "${ACE_SRC}/ace_b_basis.cpp"
        "${ACE_SRC}/ace_b_basisfunction.cpp"
        "${ACE_SRC}/ace_clebsch_gordan.cpp"
        "${ACE_SRC}/ace_couplings.cpp"
        "${ACE_SRC}/ace_yaml_input.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_c_basis_binding.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_bbasis_spec_helper.cpp"
        "${SOURCE_DIR}/ace/ace_radial_helper.cpp"
        "${SOURCE_DIR}/ace/ace_c_basisfunction_helper.cpp"
        "${SOURCE_DIR}/ace/ace_c_basis_helper.cpp"
  )

set(SOURCES_EVALUATOR "${ACE_EVALUATOR_SRC}/ace_c_basis.cpp"
             "${ACE_EVALUATOR_SRC}/ace_abstract_basis.cpp"
             "${ACE_EVALUATOR_SRC}/ace_flatten_basis.cpp"
             "${ACE_EVALUATOR_SRC}/ace_evaluator.cpp"
             "${ACE_EVALUATOR_SRC}/ace_recursive.cpp"
             "${ACE_EVALUATOR_EXTRA}/ace_atoms.cpp"
             "${ACE_EVALUATOR_SRC}/ace_radial.cpp"
             "${ACE_EVALUATOR_SRC}/ace_spherical_cart.cpp"
             "${ACE_SRC}/ace_b_evaluator.cpp"
             "${ACE_SRC}/ace_b_basis.cpp"
             "${ACE_SRC}/ace_clebsch_gordan.cpp"
             "${ACE_SRC}/ace_yaml_input.cpp"
             "${ACE_SRC}/ace_couplings.cpp"
             "${SOURCE_DIR}/ace-evaluator/ace_evaluator_binding.cpp"
        )

set(SOURCES_CATOMICENVIRONMENT "${ACE_EVALUATOR_EXTRA}/ace_atoms.cpp"
            "${SOURCE_DIR}/utils/ace_atoms_binding.cpp")


set(SOURCES_CALCULATOR
        "${ACE_EVALUATOR_SRC}/ace_radial.cpp"
        "${ACE_EVALUATOR_SRC}/ace_abstract_basis.cpp"
        "${ACE_EVALUATOR_EXTRA}/ace_atoms.cpp"
        "${ACE_EVALUATOR_SRC}/ace_spherical_cart.cpp"
        "${ACE_EVALUATOR_EXTRA}/ace_calculator.cpp"
        "${ACE_EVALUATOR_SRC}/ace_evaluator.cpp"
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
target_link_libraries(basis PRIVATE cnpy)

pybind11_add_module(evaluator ${SOURCES_EVALUATOR}  ${SOURCES_BASIS} )
target_link_libraries(evaluator PRIVATE yaml-cpp-pace)
target_link_libraries(evaluator PRIVATE cnpy)

pybind11_add_module(calculator ${SOURCES_CALCULATOR} )



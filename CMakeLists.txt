cmake_minimum_required(VERSION 3.7)

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)
project(pyace C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE True)

cmake_policy(SET CMP0068 NEW)

# set rpath for macOS
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

# Set source directory
#------------------------------------------------------
set(SOURCE_DIR "src/pyace")
set(ACE_ROOT "lib/ace")
set(ACE_SRC ${ACE_ROOT}/src/ace)
set(ACE_EXTRA_SRC ${ACE_ROOT}/src/extra)
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


# for advanced functionality
# TODO: check speed influence of this flag
add_definitions(-DEXTRA_C_PROJECTIONS)
add_definitions(-DCOMPUTE_B_GRAD)
#add_definitions(-DDEBUG_FORCES_CALCULATIONS)
#add_definitions(-DMULTIARRAY_INDICES_CHECK)
add_subdirectory(lib/ace)

#Now add sources
#--------------------------------------------------------

# Translation units shared by more than one module live in OBJECT libraries so
# they are compiled once instead of once per consuming module. The grouping is
# by the *exact* set of modules that already used each source, so every module
# still links precisely the objects it linked before.
#
# Compile settings match what pybind11_add_module applies to the modules: PIC,
# hidden visibility (required -- some of these are pybind11 binding units), and
# the same LTO selection, so each object is byte-identical to before.
function(ace_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    target_link_libraries(${name} PRIVATE pybind11::module)
    if (TARGET pybind11::lto)
        target_link_libraries(${name} PRIVATE pybind11::lto)
    endif ()
    set_target_properties(${name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden)
    if (CMAKE_COMPILER_IS_GNUCC)
        target_compile_options(${name} PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
    endif ()
endfunction()

# used by: sharmonics, coupling, basis, evaluator
ace_add_object_library(ace_coupling_obj
        "${ACE_SRC}/ace_clebsch_gordan.cpp"
        "${ACE_SRC}/ace_couplings.cpp")

# used by: basis, evaluator, calculator, grace_fs
ace_add_object_library(ace_radial_obj
        "${ACE_EVALUATOR_SRC}/ace_abstract_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ace_radial.cpp"
        "${ACE_EVALUATOR_SRC}/ace_spherical_cart.cpp")

# used by: catomicenvironment, evaluator, calculator
ace_add_object_library(ace_atoms_obj
        "${ACE_EVALUATOR_EXTRA}/ace_atoms.cpp")

# used by: evaluator, calculator
ace_add_object_library(ace_evaluator_obj
        "${ACE_EVALUATOR_SRC}/ace_evaluator.cpp")

# used by: basis, evaluator
ace_add_object_library(ace_bbasis_obj
        "${ACE_EVALUATOR_SRC}/ace_c_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ace_flatten_basis.cpp"
        "${ACE_EVALUATOR_SRC}/ships_radial.cpp"
        "${ACE_SRC}/ace_b_basis.cpp"
        "${ACE_SRC}/ace_b_basisfunction.cpp"
        "${ACE_SRC}/ace_yaml_input.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_bbasis_spec_helper.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_c_basis_binding.cpp"
        "${SOURCE_DIR}/ace/ace_c_basis_helper.cpp"
        "${SOURCE_DIR}/ace/ace_c_basisfunction_helper.cpp"
        "${SOURCE_DIR}/ace/ace_radial_helper.cpp")

# Per-module sources: only the units unique to that module.
set(SOURCES_SHARMONICS
        "${ACE_SRC}/ace_spherical_polar.cpp"
        "${SOURCE_DIR}/ace/ace_spherical_polar_binding.cpp"
        $<TARGET_OBJECTS:ace_coupling_obj>)

set(SOURCES_COUPLING
        "${SOURCE_DIR}/ace/ace_coupling_binding.cpp"
        $<TARGET_OBJECTS:ace_coupling_obj>)

set(SOURCES_CATOMICENVIRONMENT
        "${SOURCE_DIR}/utils/ace_atoms_binding.cpp"
        $<TARGET_OBJECTS:ace_atoms_obj>)

set(SOURCES_BASIS
        $<TARGET_OBJECTS:ace_coupling_obj>
        $<TARGET_OBJECTS:ace_radial_obj>
        $<TARGET_OBJECTS:ace_bbasis_obj>)

set(SOURCES_EVALUATOR
        "${ACE_EVALUATOR_SRC}/ace_recursive.cpp"
        "${ACE_SRC}/ace_b_evaluator.cpp"
        "${SOURCE_DIR}/ace-evaluator/ace_evaluator_binding.cpp"
        $<TARGET_OBJECTS:ace_coupling_obj>
        $<TARGET_OBJECTS:ace_radial_obj>
        $<TARGET_OBJECTS:ace_atoms_obj>
        $<TARGET_OBJECTS:ace_evaluator_obj>
        $<TARGET_OBJECTS:ace_bbasis_obj>)

set(SOURCES_CALCULATOR
        "${ACE_EVALUATOR_EXTRA}/ace_calculator.cpp"
        "${SOURCE_DIR}/utils/ace_calculator_binding.cpp"
        $<TARGET_OBJECTS:ace_radial_obj>
        $<TARGET_OBJECTS:ace_atoms_obj>
        $<TARGET_OBJECTS:ace_evaluator_obj>)

set(GRACE_FS_SOURCES
        "${ACE_SRC}/grace_fs_evaluator.cpp"
        "${ACE_EXTRA_SRC}/grace_fs_calculator.cpp"
        "${SOURCE_DIR}/ace/grace_fs_binding.cpp"
        $<TARGET_OBJECTS:ace_radial_obj>)


# C++ FLAGS
#---------------------------------------------------------
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
# keep Debug builds debuggable (only used with -DCMAKE_BUILD_TYPE=Debug / DEBUG=1)
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")

#Finally create the package
#--------------------------------------------------------
pybind11_add_module(sharmonics ${SOURCES_SHARMONICS} )
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(sharmonics PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(coupling ${SOURCES_COUPLING})
target_link_libraries(coupling PRIVATE yaml-cpp-pace)
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(coupling PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(catomicenvironment ${SOURCES_CATOMICENVIRONMENT} )
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(catomicenvironment PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(basis ${SOURCES_BASIS} )
target_link_libraries(basis PRIVATE yaml-cpp-pace)
target_link_libraries(basis PRIVATE cnpy)
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(basis PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(evaluator ${SOURCES_EVALUATOR} )
target_link_libraries(evaluator PRIVATE yaml-cpp-pace)
target_link_libraries(evaluator PRIVATE cnpy)
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(evaluator PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(grace_fs  ${GRACE_FS_SOURCES})
target_link_libraries(grace_fs PRIVATE yaml-cpp-pace)
target_link_libraries(grace_fs PRIVATE cnpy)
if (CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(grace_fs PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

pybind11_add_module(calculator ${SOURCES_CALCULATOR} )
if (CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(calculator PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
endif ()

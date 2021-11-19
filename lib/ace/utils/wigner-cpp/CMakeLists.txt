cmake_minimum_required (VERSION 3.2.0)

project(wigner-cpp CXX)

include(GNUInstallDirs)

option(BUILD_TEST "Build tests" OFF)
option(BUILD_DOCUMENTATION "Build API documents (requires Doxygen)" OFF)

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "\
In-source builds not allowed. Please make a new directory \
(called a build directory) and run CMake from there. \
(you may need to remove CMakeCache.txt")
endif()

# Compiler setup
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Additional CMake module path
if (EXISTS ${PROJECT_SOURCE_DIR}/cmake)
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
endif()

# directory structure
add_subdirectory(include)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library definitions
add_library(wigner-cpp INTERFACE)

target_include_directories(wigner-cpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

target_compile_features(wigner-cpp
  INTERFACE cxx_alias_templates cxx_auto_type cxx_constexpr cxx_decltype
  cxx_defaulted_functions cxx_defaulted_move_initializers cxx_lambdas
  )

install(TARGETS wigner-cpp EXPORT wigner-cpp-config INCLUDES DESTINATION "include")
install(EXPORT wigner-cpp-config DESTINATION "lib/cmake/wigner-cpp")
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Building API documentation with Doxygen
if(BUILD_DOCUMENTATION)
  find_package(Doxygen)

  if(NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen is needed to build the documentation.")
  endif()

  set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${doxyfile_in} ${doxyfile} @ONLY)

  add_custom_target(
    doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()

# Build test
if (BUILD_TEST)
  enable_testing()
  find_package(catch REQUIRED CONFIG)
  add_subdirectory(test)
endif()

cmake_minimum_required(VERSION 3.7)  # CMake version check
project(ace)
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard

set(EVALUATOR_PATH ace-evaluator)
include(${EVALUATOR_PATH}/CMakeLists.txt)
message("EVALUATOR_PATH=${EVALUATOR_PATH}")

set(FITTING_PATH src/fitting)
FILE(GLOB FITTING_SOURCE_FILES ${FITTING_PATH}/*.cpp)
set(FITTING_INCLUDE_DIR ${FITTING_PATH})

FILE(GLOB UTILS_SOURCE_FILES src/utils/*.cpp)
set(UTILS_INCLUDE_DIR src/utils)
FILE(GLOB TEST_FITTING test/fitting/test_*.cpp)
FILE(GLOB TEST_UTILS test/utils/test_*.cpp)

set(TEST_DIR test)

set(YAML_PATH ace-evaluator/lib/yaml-cpp)
set(YAML_INCLUDE_PATH ${YAML_PATH}/include)
set(YAML_LIB_PATH ${YAML_PATH}/build)

set(WIGNER_PATH utils/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include/wigner)

#add_subdirectory(${YAML_PATH})
#link_directories(${YAML_LIB_PATH})
#link_directories(${GOOGLE_TEST_LIB_DIR})

#########################    main_extended   ##########################################################################
# Add executable target with source files listed in EVALUATOR_SOURCE_FILES variable
add_executable(main_extended src/main_extended.cpp ${EVALUATOR_SOURCE_FILES} ${EVALUATOR_EXTRA_FILES} ${FITTING_SOURCE_FILES})
target_include_directories(main_extended PUBLIC ${EVALUATOR_INCLUDE_DIR} ${EVALUATOR_EXTRA_INCLUDE_DIR})
target_include_directories(main_extended PUBLIC ${FITTING_INCLUDE_DIR})
target_include_directories(main_extended PUBLIC ${YAML_INCLUDE_PATH})
target_link_libraries(main_extended yaml-cpp-pace)
#target_compile_definitions(main_extended PUBLIC MULTIARRAY_INDICES_CHECK) # enforce to check the indices range of multiarrays
target_compile_definitions(main_extended PUBLIC FINE_TIMING) #
#target_compile_definitions(main_extended PUBLIC PRINT_BASIS_FUNCTION) # print B and C-tilde basis functions
#target_compile_definitions(main_extended PUBLIC PRINT_MAIN_STEPS) # print main steps (loop over atoms, energy calc, force calc, etc.)
#target_compile_definitions(main_extended PUBLIC PRINT_INTERMEDIATE_VALUES) # print intermediate energies and forces values
#target_compile_definitions(main_extended PUBLIC DEBUG_ENERGY_CALCULATIONS) # print values of terms for energy calculations
#target_compile_definitions(main_extended PUBLIC DEBUG_FORCES_CALCULATIONS) # print values of terms for force calculations
#target_compile_definitions(main_extended PUBLIC PRINT_LOOPS_INDICES) # print loops indices (rank, density, basis function
#target_compile_definitions(main_extended PUBLIC DEBUG_C_TILDE)
#target_compile_definitions(main_extended PUBLIC DEBUG_RADIAL)
#target_compile_definitions(main_extended PUBLIC DEBUG_COUPLING)
#target_compile_definitions(main_extended PUBLIC DEBUG_READ_YAML)
#target_compile_definitions(main_extended PUBLIC BASIS_FUNCTION_LIFE_CYCLE)
#target_compile_definitions(main_extended PUBLIC MULTIARRAY_LIFE_CYCLE)
target_compile_options(main_extended PRIVATE -Ofast)
target_compile_options(main_extended PRIVATE -march=native)
target_compile_options(main_extended PRIVATE -mtune=native)

#target_compile_options(main_extended PUBLIC -Wconversion)


#########################    main_evaluator   ##########################################################################
# Add executable target with source files listed in EVALUATOR_SOURCE_FILES variable
add_executable(main_evaluator src/main_evaluator.cpp ${EVALUATOR_SOURCE_FILES} ${EVALUATOR_EXTRA_FILES})
target_include_directories(main_evaluator PUBLIC ${EVALUATOR_INCLUDE_DIR} ${EVALUATOR_EXTRA_INCLUDE_DIR})
#################### compiler flags ####################
#target_compile_definitions(main_evaluator PUBLIC MULTIARRAY_INDICES_CHECK) # enforce to check the indices range of multiarrays
target_compile_definitions(main_evaluator PUBLIC FINE_TIMING) # enforce to check the indices range of multiarrays
#target_compile_definitions(main_evaluator PUBLIC PRINT_BASIS_FUNCTION) # print B and C-tilde basis functions
#target_compile_definitions(main_evaluator PUBLIC PRINT_MAIN_STEPS) # print main steps (loop over atoms, energy calc, force calc, etc.)
#target_compile_definitions(main_evaluator PUBLIC PRINT_INTERMEDIATE_VALUES) # print intermediate energies and forces values
#target_compile_definitions(main_evaluator PUBLIC DEBUG_ENERGY_CALCULATIONS) # print values of terms for energy calculations
#target_compile_definitions(main_evaluator PUBLIC DEBUG_FORCES_CALCULATIONS) # print values of terms for force calculations
#target_compile_definitions(main_evaluator PUBLIC PRINT_LOOPS_INDICES) # print loops indices (rank, density, basis function
#target_compile_definitions(main_evaluator PUBLIC DEBUG_C_TILDE)
#target_compile_definitions(main_evaluator PUBLIC DEBUG_RADIAL)
target_compile_options(main_evaluator PRIVATE -Ofast)
target_compile_options(main_evaluator PRIVATE -march=native)
target_compile_options(main_evaluator PRIVATE -mtune=native)


########################################################################################################################
##### test_extend #####
add_executable(test_extended ${EVALUATOR_SOURCE_FILES} ${EVALUATOR_EXTRA_FILES} ${EVALUATOR_EXTRA_UTILS_FILES}
        ${TEST_FITTING} ${TEST_UTILS} ${FITTING_SOURCE_FILES} ${UTILS_SOURCE_FILES})
target_include_directories(test_extended PUBLIC ${TEST_DIR})
target_include_directories(test_extended PUBLIC ${EVALUATOR_INCLUDE_DIR} ${EVALUATOR_EXTRA_INCLUDE_DIR} ${EVALUATOR_EXTRA_UTILS_INCLUDE_DIR})
target_include_directories(test_extended PUBLIC ${FITTING_INCLUDE_DIR})
target_include_directories(test_extended PUBLIC ${UTILS_INCLUDE_DIR})
#Wigner
target_include_directories(test_extended PUBLIC ${WIGNER_INCLUDE_PATH})
#YAML
target_include_directories(test_extended PUBLIC ${YAML_INCLUDE_PATH})
target_link_libraries(test_extended yaml-cpp-pace)
#Google Test
target_include_directories(test_extended PUBLIC ${GOOGLE_TEST_INCLUDE_DIR})
target_link_libraries(test_extended gtest_main gtest)
#pthread
target_link_libraries(test_extended pthread)
#compiler options
target_compile_definitions(test_extended PUBLIC MULTIARRAY_INDICES_CHECK) #
#target_compile_definitions(test_extended PUBLIC FINE_TIMING) # enforce to check the indices range of multiarrays
#target_compile_definitions(test_extended PUBLIC PRINT_BASIS_FUNCTION) # print B and C-tilde basis functions
#target_compile_definitions(test_extended PUBLIC PRINT_MAIN_STEPS) # print main steps (loop over atoms, energy calc, force calc, etc.)
#target_compile_definitions(test_extended PUBLIC PRINT_INTERMEDIATE_VALUES) # print intermediate energies and forces values
#target_compile_definitions(test_extended PUBLIC DEBUG_ENERGY_CALCULATIONS) # print values of terms for energy calculations
#target_compile_definitions(test_extended PUBLIC DEBUG_FORCES_CALCULATIONS) # print values of terms for force calculations
#target_compile_definitions(test_extended PUBLIC DEBUG_RADIAL) #debug radial function calculations
#target_compile_definitions(test_extended PUBLIC PRINT_LOOPS_INDICES)
#target_compile_definitions(test_extended PUBLIC DEBUG_READ_YAML)
#target_compile_options(test_extended PRIVATE -march=native)
#target_compile_options(test_extended PRIVATE -mtune=native)
target_compile_options(test_extended PRIVATE -O1)

########################################################################################################################



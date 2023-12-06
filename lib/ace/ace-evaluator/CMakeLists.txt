cmake_minimum_required(VERSION 3.7)  # CMake version check
project(ace-evaluator)
set(CMAKE_CXX_STANDARD 11) # Enable c++11 standard
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EVALUATOR_PATH ${CMAKE_CURRENT_LIST_DIR})
set(EVALUATOR_SRC_PATH ${EVALUATOR_PATH}/src/ace-evaluator)


FILE(GLOB EVALUATOR_SOURCE_FILES ${EVALUATOR_SRC_PATH}/*.cpp)
set(EVALUATOR_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

FILE(GLOB EVALUATOR_EXTRA_FILES ${EVALUATOR_PATH}/src/extra/*.cpp)
set(EVALUATOR_EXTRA_INCLUDE_DIR ${EVALUATOR_PATH}/src/)
FILE(GLOB TEST_EVALUATOR ${EVALUATOR_PATH}/test/test_*.cpp)

if(DEFINED BUILD_TESTS)
    #utils-test
    set(EVALUATOR_EXTRA_UTILS_INCLUDE_DIR ${EVALUATOR_PATH}/src/extra/utils)
    FILE(GLOB EVALUATOR_EXTRA_UTILS_FILES ${EVALUATOR_PATH}/src/extra/utils/*.cpp)

    #Google Test
    set(gtest_force_shared_crt on)
    set(GOOGLE_TEST_INCLUDE_DIR ${EVALUATOR_PATH}/google-tests/googletest/googletest/include/)
    set(GOOGLE_TEST_LIB_DIR ${EVALUATOR_PATH}/google-tests/googletest/build/lib)
    link_directories(${GOOGLE_TEST_LIB_DIR})

    if (IS_DIRECTORY "${EVALUATOR_PATH}/google-tests/googletest")
        add_subdirectory(${EVALUATOR_PATH}/google-tests/googletest)
    endif ()
endif()

#YAML

set(YAML_PATH ${EVALUATOR_PATH}/lib/yaml-cpp)
set(YAML_INCLUDE_PATH ${YAML_PATH}/include)
set(YAML_LIB_PATH ${YAML_PATH}/build)
add_subdirectory(${YAML_PATH})

########################################################################################################################
##### test_evaluator #####
if(BUILD_TESTS)
    add_executable(test_evaluator ${TEST_EVALUATOR} ${EVALUATOR_SOURCE_FILES} ${EVALUATOR_EXTRA_FILES} ${EVALUATOR_EXTRA_UTILS_FILES})
    if (CMAKE_COMPILER_IS_GNUCC)
        target_compile_options(test_evaluator PRIVATE -Wunused-variable -Wvla  -Werror=vla) # -Wall -Wextra -Wsign-compare
    endif ()
    target_include_directories(test_evaluator PUBLIC ${TEST_DIR})
    target_include_directories(test_evaluator PUBLIC ${EVALUATOR_INCLUDE_DIR} ${EVALUATOR_EXTRA_INCLUDE_DIR})
    #Google Test
    target_include_directories(test_evaluator PUBLIC ${GOOGLE_TEST_INCLUDE_DIR})
    target_link_libraries(test_evaluator gtest_main gtest)

    #YAML
    target_include_directories(test_evaluator PUBLIC ${YAML_INCLUDE_PATH})
    target_link_libraries(test_evaluator yaml-cpp-pace)
    #pthread
    # target_link_libraries(test_evaluator pthread)
    target_compile_definitions(test_evaluator PUBLIC MULTIARRAY_INDICES_CHECK)
    #target_compile_definitions(test_evaluator PUBLIC FINE_TIMING) # enforce to check the indices range of multiarrays
    #target_compile_definitions(test_evaluator PUBLIC PRINT_BASIS_FUNCTION) # print B and C-tilde basis functions
    #target_compile_definitions(test_evaluator PUBLIC PRINT_MAIN_STEPS) # print main steps (loop over atoms, energy calc, force calc, etc.)
    #target_compile_definitions(test_evaluator PUBLIC PRINT_INTERMEDIATE_VALUES) # print intermediate energies and forces values
    #target_compile_definitions(test_evaluator PUBLIC DEBUG_ENERGY_CALCULATIONS) #
    #target_compile_definitions(test_evaluator PUBLIC DEBUG_FORCES_CALCULATIONS) # print values of terms for force calculations
    #target_compile_definitions(test_evaluator PUBLIC DEBUG_RADIAL) #
    #target_compile_definitions(test_evaluator PUBLIC PRINT_LOOPS_INDICES)
    #target_compile_options(test_evaluator PRIVATE -march=native)
    #target_compile_options(test_evaluator PRIVATE -mtune=native)
    # target_compile_options(test_evaluator PRIVATE -O2)
endif()
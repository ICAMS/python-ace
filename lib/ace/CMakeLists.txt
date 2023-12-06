cmake_minimum_required(VERSION 3.7)  # CMake version check
project(ace)
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EVALUATOR_PATH ace-evaluator)
include(${EVALUATOR_PATH}/CMakeLists.txt)

set(FITTING_PATH src/ace)
FILE(GLOB FITTING_SOURCE_FILES ${FITTING_PATH}/*.cpp)
set(FITTING_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

FILE(GLOB UTILS_SOURCE_FILES src/utils/*.cpp)
set(UTILS_INCLUDE_DIR src/utils)
FILE(GLOB TEST_FITTING test/fitting/test_*.cpp)
FILE(GLOB TEST_UTILS test/utils/test_*.cpp)

set(TEST_DIR test)

set(YAML_PATH ace-evaluator/lib/yaml-cpp)
set(YAML_INCLUDE_PATH ${YAML_PATH}/include)
set(YAML_LIB_PATH ${YAML_PATH}/build)

set(WIGNER_PATH utils/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include)

set(CNPY_PATH utils/cnpy)
set(CNPY_INCLUDE_PATH utils)
add_subdirectory(${CNPY_PATH})

########################################################################################################################
##### test_extend #####
if(DEFINED BUILD_TESTS)
    add_executable(test_extended ${EVALUATOR_SOURCE_FILES} ${EVALUATOR_EXTRA_FILES} ${EVALUATOR_EXTRA_UTILS_FILES}
            ${TEST_FITTING} ${TEST_UTILS} ${FITTING_SOURCE_FILES} ${UTILS_SOURCE_FILES})
    if (CMAKE_COMPILER_IS_GNUCC)
        target_compile_options(test_extended PRIVATE -Wunused-variable -Wvla -Wno-effc++ -Werror=vla)
    endif ()
    target_include_directories(test_extended PUBLIC ${TEST_DIR})
    target_include_directories(test_extended PUBLIC ${EVALUATOR_INCLUDE_DIR} ${EVALUATOR_EXTRA_INCLUDE_DIR} ${EVALUATOR_EXTRA_UTILS_INCLUDE_DIR})
    target_include_directories(test_extended PUBLIC ${FITTING_INCLUDE_DIR})
    target_include_directories(test_extended PUBLIC ${UTILS_INCLUDE_DIR})
    #Wigner
    target_include_directories(test_extended PUBLIC ${WIGNER_INCLUDE_PATH})
    # cnpy
    target_include_directories(test_extended PUBLIC ${CNPY_INCLUDE_PATH})
    target_link_libraries(test_extended cnpy)
    #YAML
    target_include_directories(test_extended PUBLIC ${YAML_INCLUDE_PATH})
    target_link_libraries(test_extended yaml-cpp-pace)
    #Google Test
    target_include_directories(test_extended PUBLIC ${GOOGLE_TEST_INCLUDE_DIR})
    target_link_libraries(test_extended gtest_main gtest)
    #pthread
    # target_link_libraries(test_extended pthread)
    #compiler options
    target_compile_definitions(test_extended PUBLIC EXTRA_C_PROJECTIONS)
    target_compile_definitions(test_extended PUBLIC COMPUTE_B_GRAD)
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
    #target_compile_options(test_extended PRIVATE -O1)
endif()
########################################################################################################################



if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

set(CMAKE_CXX_STANDARD 11) # Enable c++11 standard
#option(ENABLE_STATIC "Build static (.a) library" ON)

# Set CMAKE_BUILD_TYPE to Release if not already set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()



add_library(cnpy "cnpy.cpp")
target_include_directories(cnpy PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add compiler flags for Release builds
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#    target_compile_options(cnpy PRIVATE /O2 /Ob2 /Oi /Ot /Oy)
endif()

#if(ENABLE_STATIC)
#    add_library(cnpy-static STATIC "cnpy.cpp")
#    set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
#endif(ENABLE_STATIC)


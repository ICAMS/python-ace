if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

set(CMAKE_CXX_STANDARD 11) # Enable c++11 standard

# Set CMAKE_BUILD_TYPE to Release if not already set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()



add_library(cnpy "cnpy.cpp")
target_include_directories(cnpy PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

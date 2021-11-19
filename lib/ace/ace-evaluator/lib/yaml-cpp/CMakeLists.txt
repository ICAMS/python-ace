# 3.5 is actually available almost everywhere, but this a good minimum
cmake_minimum_required(VERSION 3.4)
project(YAML_CPP_PACE VERSION 0.6.3 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)

find_program(YAML_CPP_CLANG_FORMAT_EXE NAMES clang-format)

option(YAML_CPP_BUILD_CONTRIB "Enable yaml-cpp contrib in library" ON)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" ON)
option(YAML_BUILD_SHARED_LIBS "Build yaml-cpp shared library" ${BUILD_SHARED_LIBS})

cmake_dependent_option(YAML_CPP_BUILD_TESTS
        "Enable yaml-cpp tests" ON
        "BUILD_TESTING;CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR" OFF)
cmake_dependent_option(YAML_CPP_INSTALL
        "Enable generation of yaml-cpp install targets" ON
        "CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR" OFF)
cmake_dependent_option(YAML_MSVC_SHARED_RT
        "MSVC: Build yaml-cpp with shared runtime libs (/MD)" ON
        "MSVC" OFF)

set(yaml-cpp-type STATIC)
set(yaml-cpp-label-postfix "static")
if (YAML_BUILD_SHARED_LIBS)
    set(yaml-cpp-type SHARED)
    set(yaml-cpp-label-postfix "shared")
endif ()

set(build-shared $<BOOL:${YAML_BUILD_SHARED_LIBS}>)
set(build-windows-dll $<AND:$<BOOL:${CMAKE_HOST_WIN32}>,${build-shared}>)
set(not-msvc $<NOT:$<CXX_COMPILER_ID:MSVC>>)

if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
            MultiThreaded$<$<CONFIG:Debug>:Debug>$<${build-shared}:DLL>)
endif ()

set(contrib-pattern "src/contrib/*.cpp")
set(src-pattern "src/*.cpp")
if (CMAKE_VERSION VERSION_GREATER 3.12)
    list(INSERT contrib-pattern 0 CONFIGURE_DEPENDS)
    list(INSERT src-pattern 0 CONFIGURE_DEPENDS)
endif ()

file(GLOB yaml-cpp-contrib-sources ${contrib-pattern})
file(GLOB yaml-cpp-sources ${src-pattern})

set(msvc-rt $<TARGET_PROPERTY:MSVC_RUNTIME_LIBRARY>)

set(msvc-rt-mtd-static $<STREQUAL:${msvc-rt},MultiThreadedDebug>)
set(msvc-rt-mt-static $<STREQUAL:${msvc-rt},MultiThreaded>)

set(msvc-rt-mtd-dll $<STREQUAL:${msvc-rt},MultiThreadedDebugDLL>)
set(msvc-rt-mt-dll $<STREQUAL:${msvc-rt},MultiThreadedDLL>)

set(backport-msvc-runtime $<VERSION_LESS:${CMAKE_VERSION},3.15>)

add_library(yaml-cpp-pace ${yaml-cpp-type} "")
add_library(yaml-cpp-pace::yaml-cpp-pace ALIAS yaml-cpp-pace)

set_property(TARGET yaml-cpp-pace
        PROPERTY
        MSVC_RUNTIME_LIBRARY ${CMAKE_MSVC_RUNTIME_LIBRARY})
set_property(TARGET yaml-cpp-pace
        PROPERTY
        CXX_STANDARD_REQUIRED ON)

target_include_directories(yaml-cpp-pace
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set_target_properties(yaml-cpp-pace
            PROPERTIES
            CXX_STANDARD 11)
endif ()

target_compile_options(yaml-cpp-pace
        PRIVATE
        $<${not-msvc}:-Wall -Wextra -Wshadow -Weffc++ -Wno-long-long>
        $<${not-msvc}:-pedantic -pedantic-errors>

        $<$<AND:${backport-msvc-runtime},${msvc-rt-mtd-static}>:-MTd>
        $<$<AND:${backport-msvc-runtime},${msvc-rt-mt-static}>:-MT>
        $<$<AND:${backport-msvc-runtime},${msvc-rt-mtd-dll}>:-MDd>
        $<$<AND:${backport-msvc-runtime},${msvc-rt-mt-dll}>:-MD>

        # /wd4127 = disable warning C4127 "conditional expression is constant"
        # http://msdn.microsoft.com/en-us/library/6t66728h.aspx
        # /wd4355 = disable warning C4355 "'this' : used in base member initializer list
        # http://msdn.microsoft.com/en-us/library/3c594ae3.aspx
        $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4127 /wd4355>)

target_compile_definitions(yaml-cpp-pace
        PRIVATE
        $<${build-windows-dll}:${PROJECT_NAME}_DLL>
        $<$<NOT:$<BOOL:${YAML_CPP_BUILD_CONTRIB}>>:YAML_CPP_NO_CONTRIB>)

target_sources(yaml-cpp-pace
        PRIVATE
        $<$<BOOL:${YAML_CPP_BUILD_CONTRIB}>:${yaml-cpp-contrib-sources}>
        ${yaml-cpp-sources})

if (NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif ()

set_target_properties(yaml-cpp-pace PROPERTIES
        VERSION "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
        PROJECT_LABEL "yaml-cpp ${yaml-cpp-label-postfix}"
        DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/yaml-cpp-config-version.cmake"
        COMPATIBILITY AnyNewerVersion)


if (YAML_CPP_CLANG_FORMAT_EXE)
    add_custom_target(format
            COMMAND clang-format --style=file -i $<TARGET_PROPERTY:yaml-cpp-pace,SOURCES>
            COMMAND_EXPAND_LISTS
            COMMENT "Running clang-format"
            VERBATIM)
endif ()

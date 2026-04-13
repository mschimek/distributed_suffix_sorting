cmake_minimum_required(VERSION 3.13)
project(distributed_string_sorting)

include(FetchContent)

FetchContent_Declare(
  kaval
  GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
  GIT_TAG 1600bda
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG 685c5ab#5c8e84f
  SYSTEM)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM)

FetchContent_Declare(magic-enum
  GIT_REPOSITORY  https://github.com/Neargye/magic_enum.git
  GIT_TAG         v0.9.7
  SYSTEM
)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         cf83363
  SYSTEM
)
FetchContent_MakeAvailable(tlx)

FetchContent_Declare(ips4o
  GIT_REPOSITORY  https://github.com/mschimek/ips4o.git
  GIT_TAG         eb30352
  SYSTEM
)
FetchContent_MakeAvailable(ips4o)

FetchContent_Declare(kadis
  GIT_REPOSITORY  https://github.com/mschimek/KaDiS.git
  GIT_TAG         6713a37
  SYSTEM
)

FetchContent_Declare(kassert
  GIT_REPOSITORY https://github.com/kamping-site/kassert.git
  GIT_TAG v1.0.0
  SYSTEM)

FetchContent_MakeAvailable(kaval kamping kassert cli11 magic-enum kadis)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_NINJA_PARALLEL_PROCESSING ON)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# -Werror stops compiliation because of some files in KaDiS
list(APPEND
  DSSS_WARNING_FLAGS
  # "-Werror"
  "-Wall"
  "-Wextra"
  "-Wundef"
  "-Wunreachable-code"
  "-Wno-unused-parameter"
  # "-Wpedantic"
  "-Wnull-dereference"
  "-Wimplicit-fallthrough"
  "-Wno-vla"
  "-Wno-pessimizing-move"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND DSSS_WARNING_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DDEBUG -march=native -ggdb")

# compile options
option(DCX_OPTIMIZE_DATA_TYPES "use smallest data type possible for ranks in pdcx" OFF)
option(DCX_INCLUDE_ALL_SORTERS "include all axtmann sorters" ON)
option(DCX_BUILD_EXAMPLES "Build example executables" ON)
set(KASSERT_ASSERTION_LEVEL 0 CACHE STRING "KAssert assertion level (0=off, 30=normal, 60=light, 90=heavy)")
message(STATUS "---> DCX_OPTIMIZE_DATA_TYPES: ${DCX_OPTIMIZE_DATA_TYPES}")
message(STATUS "---> DCX_INCLUDE_ALL_SORTERS: ${DCX_INCLUDE_ALL_SORTERS}")
message(STATUS "---> DCX_BUILD_EXAMPLES: ${DCX_BUILD_EXAMPLES}")
message(STATUS "---> KASSERT_ASSERTION_LEVEL: ${KASSERT_ASSERTION_LEVEL}")



# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# Create a library containing all implementation, to be linked against by executables
add_library(dss_base INTERFACE)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(dss_base INTERFACE ${DSSS_WARNING_FLAGS})
target_compile_definitions(dss_base INTERFACE KASSERT_ASSERTION_LEVEL=${KASSERT_ASSERTION_LEVEL})

#########target_include_directories(dss_base INTERFACE
#  "${CMAKE_CURRENT_BINARY_DIR}/include")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(dss_base INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dss_base INTERFACE kamping::kamping kassert::kassert)
target_link_libraries(dss_base INTERFACE MPI::MPI_CXX)
target_link_libraries(dss_base INTERFACE tlx)
target_link_libraries(dss_base INTERFACE ips4o)
target_link_libraries(dss_base INTERFACE kadis)
target_link_libraries(dss_base INTERFACE magic_enum::magic_enum)
target_link_libraries(dss_base INTERFACE CLI11::CLI11)

set(StringSorting "src/string_sorting")
add_subdirectory(src)

# dcx static library
add_library(dcx_lib STATIC
  src/dcx/dcx.cpp
  src/dcx/pdcx_u8_3bit.cpp
  src/dcx/pdcx_u8_5bit.cpp
  src/dcx/pdcx_u8_8bit.cpp
  src/dcx/pdcx_u8_unpacked.cpp
  src/dcx/pdcx_u40_unpacked.cpp
)
target_include_directories(dcx_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dcx_lib PUBLIC dss_base)
set_target_properties(dcx_lib PROPERTIES OUTPUT_NAME dcx)
add_library(dcx::dcx ALIAS dcx_lib)

# main
# add_executable(main src/main.cpp)
add_executable(main src/executables/main.cpp)
target_include_directories(main PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(main PRIVATE ${DSSS_WARNING_FLAGS})
target_link_libraries(main kamping)
target_link_libraries(main dss_base)
target_link_libraries(main kadis)

# cli
add_executable(cli src/executables/cli.cpp)
target_include_directories(cli PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(cli PRIVATE ${DSSS_WARNING_FLAGS})
target_link_libraries(cli PRIVATE dss_base)
target_link_libraries(cli PRIVATE dcx::dcx)


# add compile flags
if(DCX_OPTIMIZE_DATA_TYPES)
  message(STATUS "setting OPTIMIZE_DATA_TYPES in main")
  message(STATUS "setting OPTIMIZE_DATA_TYPES in cli")
  message(STATUS "setting OPTIMIZE_DATA_TYPES in dcx_lib")
  target_compile_definitions(main PRIVATE OPTIMIZE_DATA_TYPES)
  target_compile_definitions(cli PRIVATE OPTIMIZE_DATA_TYPES)
  target_compile_definitions(dcx_lib PRIVATE OPTIMIZE_DATA_TYPES)
endif()

if(DCX_INCLUDE_ALL_SORTERS)
  message(STATUS "setting INCLUDE_ALL_SORTERS in main")
  message(STATUS "setting INCLUDE_ALL_SORTERS in cli")
  message(STATUS "setting INCLUDE_ALL_SORTERS in dcx_lib")
  target_compile_definitions(main PRIVATE INCLUDE_ALL_SORTERS)
  target_compile_definitions(cli PRIVATE INCLUDE_ALL_SORTERS)
  target_compile_definitions(dcx_lib PRIVATE INCLUDE_ALL_SORTERS)
endif()

# examples
if(DCX_BUILD_EXAMPLES)
  add_executable(dcx_lib_example_usage examples/dcx_lib_example_usage.cpp)
  target_link_libraries(dcx_lib_example_usage PRIVATE dcx::dcx)
endif()

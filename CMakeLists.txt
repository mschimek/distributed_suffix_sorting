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
  GIT_TAG 3bb24ce
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(magic-enum
  GIT_REPOSITORY  https://github.com/Neargye/magic_enum.git
  GIT_TAG         v0.9.7
  SYSTEM EXCLUDE_FROM_ALL
)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         cf83363
  SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(tlx)

FetchContent_Declare(ips4o
  GIT_REPOSITORY  https://github.com/mschimek/ips4o.git
  GIT_TAG         eb30352
  SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ips4o)

FetchContent_Declare(kadis
  GIT_REPOSITORY  https://github.com/mschimek/KaDiS.git
  GIT_TAG         6713a37
  SYSTEM EXCLUDE_FROM_ALL
)

FetchContent_Declare(kassert
  GIT_REPOSITORY https://github.com/kamping-site/kassert.git
  GIT_TAG v1.0.0
  SYSTEM EXCLUDE_FROM_ALL)

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
target_compile_options(dcx_lib PRIVATE ${DSSS_WARNING_FLAGS})
target_compile_definitions(dcx_lib PUBLIC KASSERT_ASSERTION_LEVEL=${KASSERT_ASSERTION_LEVEL})
target_link_libraries(dcx_lib PUBLIC
  kamping::kamping
  kassert::kassert
  MPI::MPI_CXX
  tlx
  ips4o
  kadis
  magic_enum::magic_enum
)
target_link_libraries(dcx_lib PRIVATE CLI11::CLI11)
set_target_properties(dcx_lib PROPERTIES OUTPUT_NAME dcx)
add_library(dcx::dcx ALIAS dcx_lib)

if(DCX_OPTIMIZE_DATA_TYPES)
  message(STATUS "setting OPTIMIZE_DATA_TYPES in dcx_lib")
  target_compile_definitions(dcx_lib PUBLIC OPTIMIZE_DATA_TYPES)
endif()

if(DCX_INCLUDE_ALL_SORTERS)
  message(STATUS "setting INCLUDE_ALL_SORTERS in dcx_lib")
  target_compile_definitions(dcx_lib PUBLIC INCLUDE_ALL_SORTERS)
endif()

# add header sources for IDE integration
add_subdirectory(src)

# main
add_executable(main src/executables/main.cpp)
target_compile_options(main PRIVATE ${DSSS_WARNING_FLAGS})
target_link_libraries(main PRIVATE dcx::dcx)

# cli
add_executable(cli src/executables/cli.cpp)
target_compile_options(cli PRIVATE ${DSSS_WARNING_FLAGS})
target_link_libraries(cli PRIVATE dcx::dcx CLI11::CLI11)

# examples
if(DCX_BUILD_EXAMPLES)
  add_executable(dcx_lib_example_usage examples/dcx_lib_example_usage.cpp)
  target_link_libraries(dcx_lib_example_usage PRIVATE dcx::dcx)
endif()

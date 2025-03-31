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
  GIT_TAG 1effe84
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

#FetchContent_Declare(tlx
#  GIT_REPOSITORY  https://github.com/tlx/tlx.git
#  GIT_TAG         2dd63ab
#  SYSTEM
#)

#FetchContent_Declare(
#  kadis
#  GIT_REPOSITORY https://github.com/mschimek/KaDiS.git
#  GIT_TAG 2f2cdd2
#  SYSTEM)

FetchContent_MakeAvailable(kaval kamping cli11 magic-enum)

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
option(OPTIMIZE_DATA_TYPES "use smallest data type possible for ranks in pdcx" OFF)
option(INCLUDE_ALL_SORTERS "include all axtmann sorters" OFF)
message(STATUS "---> OPTIMIZE_DATA_TYPES: ${OPTIMIZE_DATA_TYPES}")
message(STATUS "---> INCLUDE_ALL_SORTERS: ${INCLUDE_ALL_SORTERS}")



# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

set(TLX_USE_LTO ON)
add_subdirectory(external/tlx)

# IPS40 doesn't compile wihtout linking to TBB
set(IPS4O_DISABLE_PARALLEL OFF)
add_subdirectory(external/ips4o)

#add_subdirectory(external/kamping)
add_subdirectory(external/KaDiS)


# Create a library containing all implementation, to be linked against by executables
add_library(dss_base INTERFACE)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(dss_base INTERFACE ${DSSS_WARNING_FLAGS})

#########target_include_directories(dss_base INTERFACE
#  "${CMAKE_CURRENT_BINARY_DIR}/include")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(dss_base INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dss_base INTERFACE kamping::kamping)
target_link_libraries(dss_base INTERFACE MPI::MPI_CXX)
target_link_libraries(dss_base INTERFACE tlx)
target_link_libraries(dss_base INTERFACE ips4o)
target_link_libraries(dss_base INTERFACE kadis)
target_link_libraries(dss_base INTERFACE magic_enum::magic_enum)
target_link_libraries(dss_base INTERFACE CLI11::CLI11)

set(StringSorting "src/string_sorting")
add_subdirectory(src)

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


# add compile flags
if(OPTIMIZE_DATA_TYPES)
  message(STATUS "setting OPTIMIZE_DATA_TYPES in main")
  message(STATUS "setting OPTIMIZE_DATA_TYPES in cli")
  target_compile_definitions(main PRIVATE OPTIMIZE_DATA_TYPES)
  target_compile_definitions(cli PRIVATE OPTIMIZE_DATA_TYPES)
endif()
  
if(INCLUDE_ALL_SORTERS)
  message(STATUS "setting INCLUDE_ALL_SORTERS in main")
  message(STATUS "setting INCLUDE_ALL_SORTERS in cli")
  target_compile_definitions(main PRIVATE INCLUDE_ALL_SORTERS)
  target_compile_definitions(cli PRIVATE INCLUDE_ALL_SORTERS)
endif()

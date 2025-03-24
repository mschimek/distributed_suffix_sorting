#pragma once
#include <mpi.h>
#include <CLI/CLI.hpp>

#ifndef CLI11_PARSE_MPI
#define CLI11_PARSE_MPI(app, argc, argv)  \
  int retval = -1;                        \
  try {                                   \
    (app).parse((argc), (argv));          \
  } catch (const CLI::ParseError& e) {    \
    int rank;                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank == 0) {                      \
      retval = (app).exit(e);             \
    } else {                              \
      retval = 1;                         \
    }                                     \
  }                                       \
  if (retval > -1) {                      \
    MPI_Finalize();                       \
    exit(retval);                         \
  }
#endif

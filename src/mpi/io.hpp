// adapted from: https://github.com/kurpicz/dsss/blob/master/dsss/mpi/distribute_input.hpp
/*******************************************************************************
 * mpi/distribute_input.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "kamping/communicator.hpp"

namespace dsss::mpi {

using namespace kamping;
using Byte = unsigned char;

bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static std::vector<Byte> read_and_distribute_string(const std::string& input_path,
                                                    Communicator<>& comm,
                                                    size_t max_size = 0) {
    MPI_File mpi_file;

    MPI_File_open(comm.mpi_communicator(),
                  (char*)input_path.c_str(), // ugly cast to use old C interface
                  MPI_MODE_RDONLY,
                  MPI_INFO_NULL,
                  &mpi_file);

    MPI_Offset global_file_size = 0;
    MPI_File_get_size(mpi_file, &global_file_size);
    if (max_size > 0) {
        global_file_size = std::min(max_size, static_cast<size_t>(global_file_size));
    }

    size_t local_slice_size = global_file_size / comm.size();
    uint64_t larger_slices = global_file_size % comm.size();

    size_t offset;
    if (comm.rank() < larger_slices) {
        ++local_slice_size;
        offset = local_slice_size * comm.rank();
    } else {
        offset = larger_slices * (local_slice_size + 1);
        offset += (comm.rank() - larger_slices) * local_slice_size;
    }

    MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

    std::vector<Byte> result(local_slice_size);

    MPI_File_read(mpi_file, result.data(), local_slice_size, MPI_BYTE, MPI_STATUS_IGNORE);

    return result;
}

// read string and cast characters into correct datatype
template <typename char_type>
static std::vector<char_type> read_and_distribute_string(const std::string& input_path,
                                                    Communicator<>& comm,
                                                    size_t max_size = 0) {
    std::vector<Byte> data = read_and_distribute_string(input_path, comm, max_size);
    std::vector<char_type> casted_data(data.begin(), data.end());
    return casted_data;
}

template <typename DataType>
static void
write_data(std::vector<DataType>& local_data, const std::string file_name, Communicator<>& comm) {
    MPI_File mpi_file;

    MPI_File_open(comm.mpi_communicator(),
                  const_cast<char*>(file_name.c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &mpi_file);

    MPI_File_write_ordered(mpi_file,
                           local_data.data(),
                           local_data.size() * sizeof(DataType),
                           MPI_BYTE,
                           MPI_STATUS_IGNORE);

    MPI_File_close(&mpi_file);
}

} // namespace dsss::mpi

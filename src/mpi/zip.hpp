#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"
#include "util/printing.hpp"

namespace dsss::mpi_util {

using namespace kamping;

template <typename InputType, typename OutputType>
std::vector<OutputType>
zip_with_index(std::vector<InputType>& input, auto index_function, Communicator<>& comm) {
    uint64_t local_size = input.size();
    const uint64_t offset = mpi_util::ex_prefix_sum(local_size, comm);

    std::vector<OutputType> zipped;
    zipped.reserve(local_size);
    for (uint64_t i = 0; i < local_size; ++i) {
        uint64_t index = offset + i;
        zipped.push_back(index_function(index, input[i]));
    }
    return zipped;
}
} // namespace dsss::mpi_util
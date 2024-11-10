#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/distribute.hpp"
#include "mpi/stats.hpp"

namespace dsss {

template <typename T>
bool redistribute_if_imbalanced(std::vector<T>& data,
                                double min_imbalance,
                                kamping::Communicator<>& comm) {
    double imbalance = mpi_util::compute_min_imbalance(data.size(), comm);
    if (imbalance <= min_imbalance) {
        data = mpi_util::distribute_data(data, comm);
        return true;
    }
    return false;
}
} // namespace dsss
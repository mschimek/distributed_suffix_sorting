#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"

namespace dsss::mpi_util {

double compute_max_imbalance(uint64_t local_size, kamping::Communicator<>& comm) {
    using namespace kamping;
    uint64_t total_size = all_reduce_sum(local_size, comm);
    uint64_t largest_size = all_reduce_max(local_size, comm);
    double avg_size = (double)total_size / comm.size();
    KASSERT(avg_size > 0.0);
    double imbalance = ((double)largest_size / avg_size) - 1.0;
    return imbalance;
}

double compute_min_imbalance(uint64_t local_size, kamping::Communicator<>& comm) {
    using namespace kamping;
    uint64_t total_size = all_reduce_sum(local_size, comm);
    uint64_t smallest_size = all_reduce_min(local_size, comm);
    double avg_size = (double)total_size / comm.size();
    KASSERT(avg_size > 0.0);
    double imbalance = ((double)smallest_size / avg_size);
    return imbalance;
}

template <typename T>
double avg_value(std::vector<T>& v, kamping::Communicator<>& comm) {
    uint64_t local_sum = std::accumulate(v.begin(), v.end(), uint64_t(0));
    uint64_t global_sum = all_reduce_sum(local_sum, comm);
    uint64_t total_len = all_reduce_sum(v.size(), comm);
    double avg = (double)global_sum / total_len;
    return avg;
}

template <typename T>
T max_value(std::vector<T>& v, kamping::Communicator<>& comm) {
    T local_max = *std::max_element(v.begin(), v.end());
    T global_max = all_reduce_max(local_max, comm);
    return global_max;
}

template <typename T>
T min_value(std::vector<T>& v, kamping::Communicator<>& comm) {
    T local_max = *std::min_element(v.begin(), v.end());
    T global_max = all_reduce_max(local_max, comm);
    return global_max;
}
} // namespace dsss::mpi_util
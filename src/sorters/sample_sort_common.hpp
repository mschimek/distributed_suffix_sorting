#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "sorters/sample_sort_config.hpp"
#include "util/printing.hpp"

namespace dsss::mpi {

using namespace kamping;


template <typename DataType>
bool input_is_small(std::vector<DataType>& local_data, Communicator<>& comm) {
    const uint64_t local_size = local_data.size();
    const uint64_t total_size = mpi_util::all_reduce_sum(local_size, comm);
    const uint64_t small_size = std::max(4ull * comm.size(), 1000ull);
    return total_size <= small_size;
}
// if input is small enough, send all data to the root and locally sort
// returns true, if input was sorted on root
template <typename DataType>
void sort_on_root(std::vector<DataType>& local_data, Communicator<>& comm, auto sorter) {
    const uint64_t local_n = local_data.size();
    std::vector<DataType> global_data = comm.gatherv(kamping::send_buf(local_data));
    sorter(global_data);
    local_data = mpi_util::distribute_data_custom(global_data, local_n, comm);
}

// sample 16 * log2 p splitters uniform at random
template <typename DataType>
std::vector<DataType> sample_random_splitters(std::vector<DataType>& local_data,
                                              Communicator<>& comm) {
    const size_t log_p = std::ceil(std::log2(comm.size()));
    const size_t nr_splitters = std::min(16 * log_p, local_data.size());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<uint64_t> dist(0, local_data.size() - 1);

    std::vector<DataType> local_splitters;
    local_splitters.reserve(nr_splitters);
    for (size_t i = 0; i < nr_splitters; ++i) {
        uint64_t r = dist(rng);
        local_splitters.emplace_back(local_data[r]);
    }
    return local_splitters;
}

// samples p splitter in regular interval of sorted data
template <typename DataType>
std::vector<DataType> sample_uniform_splitters(std::vector<DataType>& local_data,
                                               Communicator<>& comm) {
    const size_t local_n = local_data.size();
    size_t nr_splitters = std::min<size_t>(comm.size() - 1, local_n);
    size_t splitter_dist = local_n / (nr_splitters + 1);

    std::vector<DataType> local_splitters;
    local_splitters.reserve(nr_splitters);
    for (size_t i = 1; i <= nr_splitters; ++i) {
        local_splitters.emplace_back(local_data[i * splitter_dist]);
    }
    return local_splitters;
}

// allgather data and locally sort
template <typename DataType>
std::vector<DataType> sample_global_splitters_centralized(std::vector<DataType>& local_splitters,
                                                          auto local_sorter,
                                                          kamping::Communicator<>& comm) {
    // Collect and sort all splitters
    std::vector<DataType> all_splitters = comm.allgatherv(kamping::send_buf(local_splitters));
    local_sorter(all_splitters);

    // select subset of splitters as global splitters
    return sample_uniform_splitters(all_splitters, comm);
}

// use distributed sorter to sort splitters and then select splitters using prefixsums
template <typename DataType>
std::vector<DataType> sample_global_splitters_distributed(std::vector<DataType>& local_splitters,
                                                          auto distributed_sorter,
                                                          kamping::Communicator<>& comm) {
    distributed_sorter(local_splitters);

    const int64_t local_n = local_splitters.size();
    const int64_t global_n = mpi_util::all_reduce_sum(local_n, comm);
    const int64_t nr_splitters = std::min((int64_t)comm.size() - 1, global_n);
    const int64_t splitter_dist = global_n / (nr_splitters + 1);
    const int64_t elements_before = mpi_util::ex_prefix_sum(local_n, comm);
    const int64_t last_element = elements_before + local_n - 1;

    // find splitter positions
    std::vector<DataType> partial_splitters;
    const int64_t first_splitter = 1 + ((elements_before - 1) / splitter_dist);
    const int64_t last_splitter = std::min(nr_splitters, last_element / splitter_dist);
    for (int64_t i = first_splitter; i <= last_splitter; ++i) {
        int64_t global_index = i * splitter_dist;
        int64_t local_index = global_index - elements_before;
        KASSERT(0 <= local_index && local_index < local_n);
        partial_splitters.emplace_back(local_splitters[local_index]);
    }

    // collect global splitters
    std::vector<DataType> global_splitters = comm.allgatherv(kamping::send_buf(partial_splitters));
    KASSERT((int64_t)global_splitters.size() == nr_splitters);
    return global_splitters;
}

template <typename DataType>
std::vector<DataType> get_global_splitters(std::vector<DataType>& local_data,
                                           auto local_sorter,
                                           auto distributed_sorter,
                                           kamping::Communicator<>& comm,
                                           SampleSortConfig& config) {
    // Compute the local splitters given the sorted data
    std::vector<DataType> local_splitters;
    if (config.splitter_sampling == SplitterSampling::Uniform) {
        local_splitters = sample_uniform_splitters(local_data, comm);
    } else {
        local_splitters = sample_random_splitters(local_data, comm);
    }

    // select subset of splitters as global splitters
    std::vector<DataType> global_splitters;
    if (config.splitter_sorting == SplitterSorting::Distributed) {
        global_splitters =
            sample_global_splitters_distributed(local_splitters, distributed_sorter, comm);
    } else {
        global_splitters = sample_global_splitters_centralized(local_splitters, local_sorter, comm);
    }
    return global_splitters;
}


// compute size of intervals into which element are divided by splitters
template <typename DataType, class Compare>
std::vector<int64_t> compute_interval_sizes(std::vector<DataType>& local_data,
                                            std::vector<DataType>& splitters,
                                            Communicator<>& comm,
                                            Compare comp) {
    const size_t local_n = local_data.size();
    if (local_n == 0) {
        return std::vector<int64_t>(splitters.size(), 0);
    }
    size_t nr_splitters = std::min<size_t>(comm.size() - 1, local_n);
    size_t splitter_dist = local_n / (nr_splitters + 1);

    std::vector<int64_t> interval_sizes;
    size_t element_pos = 0;
    for (size_t i = 0; i < splitters.size(); ++i) {
        // inital guess of border borders
        element_pos = ((i + 1) * splitter_dist);

        // search for splitter border
        while (element_pos > 0 && !comp(local_data[element_pos], splitters[i])) {
            --element_pos;
        }
        while (element_pos < local_n && comp(local_data[element_pos], splitters[i])) {
            ++element_pos;
        }
        interval_sizes.emplace_back(element_pos);
    }

    // convert position to interval sizes
    interval_sizes.emplace_back(local_n);
    for (size_t i = interval_sizes.size() - 1; i > 0; --i) {
        interval_sizes[i] -= interval_sizes[i - 1];
    }
    return interval_sizes;
}

} // namespace dsss::mpi

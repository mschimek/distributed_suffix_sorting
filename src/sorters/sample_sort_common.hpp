#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kassert/kassert.hpp"
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
template <typename DataType>
void sort_on_root(std::vector<DataType>& local_data, Communicator<>& comm, auto sorter) {
    std::vector<DataType> global_data = comm.gatherv(kamping::send_buf(local_data));
    sorter(global_data);
    local_data = global_data;
}

template <typename DataType>
void redistribute_imbalanced_data(std::vector<DataType>& local_data, Communicator<>& comm) {
    uint64_t min_size = mpi_util::all_reduce_min(local_data.size(), comm);
    if (min_size <= comm.size()) {
        local_data = mpi_util::distribute_data(local_data, comm);
    }
}


// sample splitters uniform at random
template <typename DataType>
std::vector<DataType> sample_random_splitters1(uint64_t total_elements,
                                               size_t nr_splitters,
                                               auto get_element_at,
                                               Communicator<>& comm) {
    std::mt19937 rng(comm.rank());
    std::uniform_int_distribution<uint64_t> dist(0, total_elements - 1);

    std::vector<DataType> local_splitters;
    local_splitters.reserve(nr_splitters);
    for (size_t i = 0; i < nr_splitters; ++i) {
        uint64_t r = dist(rng);
        local_splitters.emplace_back(get_element_at(r));
    }
    return local_splitters;
}

template <typename DataType>
std::vector<DataType> sample_random_splitters(std::vector<DataType>& local_data,
                                              size_t nr_splitters,
                                              Communicator<>& comm) {
    auto get_element_at = [&](uint64_t i) { return local_data[i]; };
    return sample_random_splitters1<DataType>(local_data.size(),
                                              nr_splitters,
                                              get_element_at,
                                              comm);
}

template <typename DataType>
std::vector<DataType> sample_random_splitters(std::vector<DataType>& local_data,
                                              Communicator<>& comm) {
    const size_t log_p = std::ceil(std::log2(comm.size()));
    const size_t nr_splitters = std::min(16 * log_p, local_data.size());
    return sample_random_splitters(local_data, nr_splitters, comm);
}

// samples splitters in regular interval of sorted data

template <typename DataType>
std::vector<DataType> sample_uniform_splitters(std::vector<DataType>& local_data,
                                               size_t nr_splitters,
                                               Communicator<>& comm) {
    const size_t local_n = local_data.size();
    size_t splitter_dist = local_n / (nr_splitters + 1);

    std::vector<DataType> local_splitters;
    local_splitters.reserve(nr_splitters);
    for (size_t i = 1; i <= nr_splitters; ++i) {
        local_splitters.emplace_back(local_data[i * splitter_dist]);
    }
    return local_splitters;
}

template <typename DataType>
std::vector<DataType> sample_uniform_splitters(std::vector<DataType>& local_data,
                                               Communicator<>& comm) {
    const size_t local_n = local_data.size();
    size_t nr_splitters = std::min<size_t>(comm.size() - 1, local_n);
    return sample_uniform_splitters(local_data, nr_splitters, comm);
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


template <typename DataType, class Compare>
size_t linear_scan_splitter_position(std::vector<DataType>& local_data,
                                     Compare comp,
                                     DataType& splitter,
                                     size_t initial_guess = 0) {
    size_t element_pos = initial_guess;
    // search for splitter border
    while (element_pos > 0 && !comp(local_data[element_pos], splitter)) {
        --element_pos;
    }
    while (element_pos < local_data.size() && comp(local_data[element_pos], splitter)) {
        ++element_pos;
    }
    return element_pos;
}

// compute size of intervals into which element are divided by splitters
template <typename DataType, class Compare>
std::vector<int64_t> compute_interval_sizes(std::vector<DataType>& local_data,
                                            std::vector<DataType>& splitters,
                                            Communicator<>& comm,
                                            Compare comp,
                                            SampleSortConfig& config) {
    const size_t local_n = local_data.size();
    const size_t nr_splitters = splitters.size();
    const size_t nr_send_counts = splitters.size() + 1;

    if (local_n == 0) {
        return std::vector<int64_t>(nr_send_counts, 0);
    }

    KASSERT(std::is_sorted(splitters.begin(), splitters.end(), comp));

    std::vector<int64_t> interval_sizes;
    interval_sizes.reserve(nr_send_counts);
    size_t element_pos = 0;
    for (size_t i = 0; i < splitters.size(); ++i) {
        if (config.use_binary_search_for_splitters) {
            // start left interval from last found 
            const size_t start_pos = std::min(element_pos, local_data.size() - 1);
            KASSERT(start_pos < local_data.size());
            auto it = std::lower_bound(local_data.begin() + start_pos,
                                       local_data.end(),
                                       splitters[i],
                                       comp);
            element_pos = it - local_data.begin();
        } else {
            // assume splitters to be distributed equally in remaining interval
            KASSERT(local_n >= element_pos);
            const size_t remaining_n = local_n - element_pos;
            KASSERT(nr_splitters + 1 > i);
            const size_t splitter_dist = remaining_n / (nr_splitters + 1 - i);
            const size_t initial_guess = element_pos + splitter_dist;
            element_pos =
                linear_scan_splitter_position(local_data, comp, splitters[i], initial_guess);
        }
        interval_sizes.emplace_back(element_pos);
    }

    for (size_t i = 0; i < interval_sizes.size() - 1; ++i) {
        KASSERT(interval_sizes[i] <= interval_sizes[i + 1]);
    }

    // convert position to interval sizes
    interval_sizes.emplace_back(local_n);
    for (size_t i = interval_sizes.size() - 1; i > 0; --i) {
        interval_sizes[i] -= interval_sizes[i - 1];
    }
    KASSERT(interval_sizes.size() == nr_send_counts);
    KASSERT(std::accumulate(interval_sizes.begin(), interval_sizes.end(), int64_t(0)) == int64_t(local_n));
    return interval_sizes;
}

} // namespace dsss::mpi

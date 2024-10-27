#pragma once

#include <algorithm>
#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"

namespace dsss::mpi {

using namespace kamping;

// if input is small enough, send all data to the root and locally sort
// returns true, if input was sorted on root
template <typename DataType>
bool sort_on_root(std::vector<DataType>& local_data, Communicator<>& comm, auto& sorter) {
    uint64_t local_size = local_data.size();
    uint64_t total_size = mpi_util::all_reduce_sum(local_size, comm);
    uint64_t small_size = std::max(4ull * comm.size(), 1000ull);
    bool do_local_sort = total_size <= small_size;
    if (do_local_sort) {
        auto global_data = comm.gatherv(kamping::send_buf(local_data));
        sorter(global_data);
        local_data = mpi_util::distribute_data_custom(global_data, local_size, comm);
    }
    return do_local_sort;
}

// gets splitter in regular interval of sorted data
template <typename DataType>
std::vector<DataType> get_uniform_splitters(std::vector<DataType>& local_data,
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

// compute size of intervals into which element are divided by splitters
template <typename DataType, class Compare>
std::vector<int64_t> compute_interval_sizes(std::vector<DataType>& local_data,
                                            std::vector<DataType>& splitters,
                                            Communicator<>& comm,
                                            Compare comp) {
    const size_t local_n = local_data.size();
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

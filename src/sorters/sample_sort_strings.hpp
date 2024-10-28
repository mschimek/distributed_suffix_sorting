#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/stats.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"

namespace dsss::mpi {

template <typename DataType>
inline void sample_sort_strings(std::vector<DataType>& local_data,
                                kamping::Communicator<>& comm,
                                SeqStringSorterWrapper sorting_wrapper,
                                double& avg_lcps_length,
                                bool use_lcps = true) {
    // set memory in string sorter?
    std::vector<SeqStringSorterWrapper::LcpType> lcps;
    if (use_lcps) {
        lcps.resize(local_data.size(), 0);
    }

    auto local_sorter = [&](std::vector<DataType>& local_data) {
        sorting_wrapper.sort(local_data);
    };
    auto local_sorter_with_lcp = [&](std::vector<DataType>& local_data) {
        if (use_lcps) {
            sorting_wrapper.sort_with_lcps(local_data, lcps);
        } else {
            sorting_wrapper.sort(local_data);
        }
    };
    auto comp = [&](DataType& s1, DataType& s2) { return string_cmp(s1, s2) < 0; };

    // code breaks for very small inputs --> switch to sequential sorting
    if (sort_on_root(local_data, comm, local_sorter)) {
        return;
    }

    // Sort data locally
    local_sorter_with_lcp(local_data);

    // Compute the local splitters given the sorted data
    std::vector<DataType> local_splitters = get_uniform_splitters(local_data, comm);

    // Collect and sort all splitters
    std::vector<DataType> all_splitters = comm.allgatherv(kamping::send_buf(local_splitters));
    local_sorter(all_splitters);

    // select subset of splitters as global splitters
    std::vector<DataType> global_splitters = get_uniform_splitters(all_splitters, comm);

    // Use the final set of splitters to find the intervals
    std::vector<int64_t> interval_sizes =
        compute_interval_sizes(local_data, global_splitters, comm, comp);

    // exchange data in intervals
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);
    if (use_lcps) {
        // invalidate first lcp of each interval
        uint64_t pos = 0;
        for (uint64_t i = 0; i < interval_sizes.size(); i++) {
            lcps[pos] = 0;
            pos += interval_sizes[i];
        }
        lcps = mpi_util::alltoallv_combined(lcps, interval_sizes, comm);
    }

    // TODO use loser tree, for now locally sort
    // merge buckets
    local_sorter_with_lcp(local_data);

    avg_lcps_length = mpi_util::avg_value(lcps, comm);
    // uint64_t local_max_lcps = *std::max_element(lcps.begin(), lcps.end());
    // print_concatenated(local_max_lcps, comm, "max_lcps");
}

} // namespace dsss::mpi

/******************************************************************************/
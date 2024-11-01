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
#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"
#include "util/uint_types.hpp"

#ifdef INCLUDE_ALL_SORTERS
#include "RQuick/RQuick.hpp"
#endif

namespace dsss::mpi {

using LcpType = SeqStringSorterWrapper::LcpType;

template <typename DataType>
inline std::vector<LcpType> sample_sort_strings(std::vector<DataType>& local_data,
                                kamping::Communicator<>& comm,
                                SeqStringSorterWrapper sorting_wrapper,
                                SampleSortConfig config = SampleSortConfig(),
                                bool output_lcps = false) {
    
    std::vector<LcpType> lcps;

    // set memory in string sorter?

    auto local_sorter = [&](std::vector<DataType>& local_data) {
        sorting_wrapper.sort(local_data);
    };
    auto local_sorter_with_lcp = [&](std::vector<DataType>& local_data) {
        if (output_lcps) {
            lcps.resize(local_data.size());
            std::fill(lcps.begin(), lcps.end(), LcpType(0));
            sorting_wrapper.sort_with_lcps(local_data, lcps);
        } else {
            sorting_wrapper.sort(local_data);
        }
    };
    auto comp = [&](const DataType& s1, const DataType& s2) { return string_cmp(s1, s2) < 0; };

    auto distributed_sorter = [&](std::vector<DataType>& local_splitters) {
        SampleSortConfig config2 = config;
        config2.splitter_sorting = SplitterSorting::Central;
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
        std::mt19937_64 gen;
        int tag = 42;
        MPI_Comm mpi_comm(comm.mpi_communicator());
        if (config.use_rquick_for_splitters) {
            RQuick::sort(my_mpi_type, local_splitters, tag, gen, mpi_comm, comp);
        } else {
            sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
        }
#else
        sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
#endif
    };

    // code breaks for very small inputs --> switch to sequential sorting
    if (input_is_small(local_data, comm)) {
        sort_on_root(local_data, comm, local_sorter_with_lcp);
        if(output_lcps) {
            lcps = mpi_util::distribute_data_custom(lcps, local_data.size(), comm);
        }
        return lcps;
    }

    // Sort data locally
    local_sorter(local_data);

    // compute global splitters
    std::vector<DataType> global_splitters =
        get_global_splitters(local_data, local_sorter, distributed_sorter, comm, config);

    // Use the final set of splitters to find the intervals
    std::vector<int64_t> interval_sizes =
        compute_interval_sizes(local_data, global_splitters, comm, comp);

    // exchange data in intervals
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);
    // if (write_lcps) {
    //     // invalidate first lcp of each interval
    //     uint64_t pos = 0;
    //     for (uint64_t i = 0; i < interval_sizes.size(); i++) {
    //         lcps[pos] = 0;
    //         pos += interval_sizes[i];
    //     }
    //     lcps = mpi_util::alltoallv_combined(lcps, interval_sizes, comm);
    // }

    // TODO use loser tree, for now locally sort
    // merge buckets
    local_sorter_with_lcp(local_data);
    return lcps;
}

} // namespace dsss::mpi

/******************************************************************************/
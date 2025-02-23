// source: https://github.com/kurpicz/dsss/blob/master/dsss/mpi/sort.hpp

/*******************************************************************************
 * mpi/sort.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

#include <tlx/container/loser_tree.hpp>

#include "ips4o.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/sample_sort_config.hpp"
#include "util/printing.hpp"

#ifdef INCLUDE_ALL_SORTERS
#include "RQuick/RQuick.hpp"
#endif

namespace dsss::mpi {

template <typename DataType, class Compare>
inline void sample_sort(std::vector<DataType>& local_data,
                        Compare comp,
                        kamping::Communicator<>& comm,
                        SampleSortConfig config = SampleSortConfig()) {
    auto& timer = kamping::measurements::timer();

    auto local_sorter = [&](std::vector<DataType>& data) {
        ips4o::sort(data.begin(), data.end(), comp);
    };
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
            sample_sort(local_splitters, comp, comm, config2);
        }
#else
        sample_sort(local_splitters, comp, comm, config2);
#endif
    };

    // code breaks for very small inputs --> switch to sequential sorting
    if (input_is_small(local_data, comm)) {
        uint64_t local_n = local_data.size();
        sort_on_root(local_data, comm, local_sorter);
        local_data = mpi_util::distribute_data_custom(local_data, local_n, comm);
        return;
    }


    // handle cases with empty PEs
    timer.synchronize_and_start("sample_sort_distribute_data");
    redistribute_imbalanced_data(local_data, comm);
    timer.stop();


    // Sort data locally
    timer.synchronize_and_start("sample_sort_local_sorting_01");
    local_sorter(local_data);
    timer.stop();

    // compute global splitters
    timer.synchronize_and_start("sample_sort_global_splitters");
    std::vector<DataType> global_splitters =
        get_global_splitters(local_data, local_sorter, distributed_sorter, comm, config);
    timer.stop();


    DBG("interval sizes");
    // Use the final set of splitters to find the intervals
    std::vector<int64_t> interval_sizes =
        compute_interval_sizes(local_data, global_splitters, comm, comp, config);


    DBG("receiving sizes");
    std::vector<int64_t> receiving_sizes = comm.alltoall(kamping::send_buf(interval_sizes));
    for (size_t i = interval_sizes.size(); i < comm.size(); ++i) {
        interval_sizes.emplace_back(0);
    }

    // exchange data in intervals
    timer.synchronize_and_start("sample_sort_alltoall");
    DBG("alltoall");
    if constexpr (DEBUG_SIZE) {
        print_concatenated_size(local_data, comm, "local data size");
        print_concatenated_size(interval_sizes, comm, "interval_sizes size");
        print_concatenated(interval_sizes, comm, "interval_sizes");
    }
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);
    timer.stop();

    if (config.use_loser_tree) {
        DBG("loser tree");

        std::vector<decltype(local_data.cbegin())> string_it(comm.size(), local_data.cbegin());
        std::vector<decltype(local_data.cbegin())> end_it(comm.size(),
                                                          local_data.cbegin() + receiving_sizes[0]);

        [[maybe_unused]] size_t received_elements = receiving_sizes[0];
        for (size_t i = 1; i < comm.size(); ++i) {
            string_it[i] = string_it[i - 1] + receiving_sizes[i - 1];
            received_elements += receiving_sizes[i];
            end_it[i] = end_it[i - 1] + receiving_sizes[i];
        }

        struct item_compare {
            item_compare(Compare compare) : comp_(compare) {}

            bool operator()(const DataType& a, const DataType& b) { return comp_(a, b); }

        private:
            Compare comp_;
        }; // struct item_compare

        tlx::LoserTreeCopy<false, DataType, item_compare> lt(comm.size(), item_compare(comp));

        size_t filled_sources = 0;
        for (size_t i = 0; i < comm.size(); ++i) {
            if (string_it[i] >= end_it[i]) {
                lt.insert_start(nullptr, i, true);
            } else {
                lt.insert_start(&*string_it[i], i, false);
                ++filled_sources;
            }
        }

        lt.init();

        std::vector<DataType> result;
        result.reserve(local_data.size());
        while (filled_sources) {
            int32_t source = lt.min_source();
            result.push_back(*string_it[source]);
            ++string_it[source];
            if (string_it[source] < end_it[source]) {
                lt.delete_min_insert(&*string_it[source], false);
            } else {
                lt.delete_min_insert(nullptr, true);
                --filled_sources;
            }
        }
        local_data = std::move(result);
    } else if (local_data.size() > 0) {
        timer.synchronize_and_start("sample_sort_local_sorting_02");
        DBG("local_sorting 2");
        local_sorter(local_data);
        timer.stop();
    }
}

} // namespace dsss::mpi

/******************************************************************************/
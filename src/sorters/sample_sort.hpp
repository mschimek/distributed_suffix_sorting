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
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "sorters/sample_sort_common.hpp"

namespace dsss::mpi {

template <typename DataType, class Compare>
inline void
sample_sort(std::vector<DataType>& local_data, Compare comp, kamping::Communicator<>& comm) {
    auto local_sorter = [&](std::vector<DataType>& local_data) {
        ips4o::sort(local_data.begin(), local_data.end(), comp);
    };

    // code breaks for very small inputs --> switch to sequential sorting
    if (sort_on_root(local_data, comm, local_sorter)) {
        return;
    }

    // Sort data locally
    local_sorter(local_data);

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

    std::vector<int64_t> receiving_sizes = comm.alltoall(kamping::send_buf(interval_sizes));
    for (size_t i = interval_sizes.size(); i < comm.size(); ++i) {
        interval_sizes.emplace_back(0);
    }

    // exchange data in intervals
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);

    //   if (false && local_data.size() > 1024 * 1024) {
    // constexpr bool use_loser_tree = true;
    constexpr bool use_loser_tree = false;
    if (use_loser_tree) {
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
        local_sorter(local_data);
    }
}

} // namespace dsss::mpi

/******************************************************************************/
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
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"

namespace dsss::mpi {

template <typename DataType, class Compare>
inline void sample_sort(std::vector<DataType>& local_data, Compare comp, kamping::Communicator<>& comm) {
    // code breaks for very small inputs --> switch to sequential sorting
    uint64_t local_size = local_data.size();
    uint64_t total_size = mpi_util::all_reduce_sum(local_size, comm);
    if (total_size < 100) {
        auto global_data = comm.gatherv(kamping::send_buf(local_data));
        ips4o::sort(global_data.begin(), global_data.end(), comp);
        local_data = mpi_util::distribute_data_custom(global_data, local_size, comm);
        return;
    }

    // Sort locally
    ips4o::sort(local_data.begin(), local_data.end(), comp);

    // Compute the local splitters given the sorted data
    const size_t local_n = local_data.size();
    auto nr_splitters = std::min<size_t>(comm.size() - 1, local_n);
    auto splitter_dist = local_n / (nr_splitters + 1);

    std::vector<DataType> local_splitters;
    local_splitters.reserve(nr_splitters);
    for (size_t i = 1; i <= nr_splitters; ++i) {
        local_splitters.emplace_back(local_data[i * splitter_dist]);
    }

    // Distribute the local splitters, which results in the set of global
    // splitters. Those are then sorted ...

    auto global_splitters = comm.allgatherv(kamping::send_buf(local_splitters));
    ips4o::sort(global_splitters.begin(), global_splitters.end(), comp);

    // ... to get the final set of splitters.
    nr_splitters = std::min<size_t>(comm.size() - 1, global_splitters.size());
    splitter_dist = global_splitters.size() / (nr_splitters + 1);
    local_splitters.clear();
    for (size_t i = 1; i <= nr_splitters; ++i) {
        local_splitters.emplace_back(global_splitters[i * splitter_dist]);
    }

    // Use the final set of splitters to find the intervals
    std::vector<int> interval_sizes;

    size_t element_pos = 0;
    splitter_dist = local_n / (nr_splitters + 1);

    for (size_t i = 0; i < local_splitters.size(); ++i) {
        element_pos = ((i + 1) * splitter_dist);
        while (element_pos > 0 && !comp(local_data[element_pos], local_splitters[i])) {
            --element_pos;
        }
        while (element_pos < local_n && comp(local_data[element_pos], local_splitters[i])) {
            ++element_pos;
        }
        interval_sizes.emplace_back(element_pos);
    }
    interval_sizes.emplace_back(local_n);
    for (size_t i = interval_sizes.size() - 1; i > 0; --i) {
        interval_sizes[i] -= interval_sizes[i - 1];
    }

    std::vector<int> receiving_sizes = comm.alltoall(kamping::send_buf(interval_sizes));

    for (size_t i = interval_sizes.size(); i < comm.size(); ++i) {
        interval_sizes.emplace_back(0);
    }

    local_data =
        comm.alltoallv(kamping::send_buf(local_data), kamping::send_counts(interval_sizes));

    //   if (false && local_data.size() > 1024 * 1024) {
    constexpr bool use_loser_tree = true;
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
        ips4o::sort(local_data.begin(), local_data.end(), comp);
    }
}

} // namespace dsss::mpi

/******************************************************************************/
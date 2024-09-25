#pragma once

#include <algorithm>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"

namespace dsss::mpi_util {

using namespace kamping;

// adapted from: https://github.com/kurpicz/dsss/blob/master/dsss/mpi/distribute_data.hpp
// distributs data such that each process i < comm.size() has n / comm.size() elements
// and the first process has the remaining elements.
template <typename DataType>
std::vector<DataType> distribute_data(std::vector<DataType>& local_data, Communicator<>& comm) {
    size_t num_processes = comm.size();
    size_t cur_local_size = local_data.size();
    size_t total_size = all_reduce_sum(cur_local_size, comm);
    size_t local_size = std::max<size_t>(1, total_size / num_processes);

    size_t local_data_size = local_data.size();
    size_t preceding_size = comm.exscan(send_buf(local_data_size), op(ops::plus<>{}))[0];

    auto get_target_rank = [&](const size_t pos) {
        return std::min(num_processes - 1, pos / local_size);
    };

    std::vector<int> send_cnts(num_processes, 0);
    for (auto cur_rank = get_target_rank(preceding_size);
         local_data_size > 0 && cur_rank < num_processes;
         ++cur_rank) {
        const size_t to_send =
            std::min(((cur_rank + 1) * local_size) - preceding_size, local_data_size);
        send_cnts[cur_rank] = to_send;
        local_data_size -= to_send;
        preceding_size += to_send;
    }
    send_cnts.back() += local_data_size;

    std::vector<DataType> result = comm.alltoallv(send_buf(local_data), send_counts(send_cnts));
    return result;
}

template <typename DataType>
std::vector<DataType> distribute_data_custom(std::vector<DataType>& local_data,
                                             int local_target_size,
                                             Communicator<>& comm) {
    int num_processes = comm.size();
    int local_size = local_data.size();

    KASSERT(all_reduce_sum(local_size, comm) == all_reduce_sum(local_target_size, comm),
            "total and target size don't match");

    std::vector<int> target_sizes = comm.allgather(send_buf(local_target_size));
    std::vector<int> preceding_target_size(num_processes);
    std::exclusive_scan(target_sizes.begin(), target_sizes.end(), target_sizes.begin(), 0);

    int local_data_size = local_data.size();
    int preceding_size = comm.exscan(send_buf(local_size), op(ops::plus<>{}))[0];

    std::vector<int> send_cnts(num_processes, 0);
    for (int cur_rank = 0; cur_rank < num_processes - 1 && local_data_size > 0; cur_rank++) {
        int to_send = std::max(0, target_sizes[cur_rank + 1] - preceding_size);
        to_send = std::min(to_send, local_data_size);
        send_cnts[cur_rank] = to_send;
        local_data_size -= to_send;
        preceding_size += to_send;
    }
    send_cnts.back() += local_data_size;

    std::vector<DataType> result = comm.alltoallv(send_buf(local_data), send_counts(send_cnts));
    return result;
}
} // namespace dsss::mpi_util
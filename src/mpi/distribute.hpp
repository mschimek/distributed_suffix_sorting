#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/reduce.hpp"

namespace dsss::mpi_util {

using namespace kamping;

// adapted from: https://github.com/kurpicz/dsss/blob/master/dsss/mpi/distribute_data.hpp
// distributs data such that each process i < comm.size() has n / comm.size() elements
// and the first process has the remaining elements.
template <typename DataType>
std::vector<DataType> distribute_data(std::vector<DataType>& local_data, Communicator<>& comm) {
    int64_t num_processes = comm.size();
    int64_t cur_local_size = local_data.size();
    int64_t total_size = all_reduce_sum(cur_local_size, comm);
    int64_t local_size = std::max<int64_t>(1, total_size / num_processes);

    int64_t local_data_size = local_data.size();
    int64_t preceding_size = comm.exscan(send_buf(local_data_size), op(ops::plus<>{}))[0];

    auto get_target_rank = [&](const int64_t pos) {
        return std::min(num_processes - 1, pos / local_size);
    };

    std::vector<int64_t> send_cnts(num_processes, 0);
    for (auto cur_rank = get_target_rank(preceding_size);
         local_data_size > 0 && cur_rank < num_processes;
         ++cur_rank) {
        const int64_t to_send =
            std::min(((cur_rank + 1) * local_size) - preceding_size, local_data_size);
        send_cnts[cur_rank] = to_send;
        local_data_size -= to_send;
        preceding_size += to_send;
    }
    send_cnts.back() += local_data_size;

    std::vector<DataType> result = mpi_util::alltoallv_combined(local_data, send_cnts, comm);
    return result;
}

template <typename DataType>
std::vector<DataType> distribute_data_custom(std::vector<DataType>& local_data,
                                             int64_t local_target_size,
                                             Communicator<>& comm) {
    int64_t num_processes = comm.size();
    int64_t local_size = local_data.size();

    KASSERT(all_reduce_sum(local_size, comm) == all_reduce_sum(local_target_size, comm),
            "total and target size don't match");

    std::vector<int64_t> target_sizes = comm.allgather(send_buf(local_target_size));
    std::vector<int64_t> preceding_target_size(num_processes);
    std::exclusive_scan(target_sizes.begin(),
                        target_sizes.end(),
                        preceding_target_size.begin(),
                        int64_t(0));

    int64_t local_data_size = local_data.size();
    int64_t preceding_size = comm.exscan(send_buf(local_size), op(ops::plus<>{}))[0];

    std::vector<int64_t> send_cnts(num_processes, 0);
    for (int64_t cur_rank = 0; cur_rank < num_processes - 1 && local_data_size > 0; cur_rank++) {
        int64_t to_send =
            std::max(int64_t(0), preceding_target_size[cur_rank + 1] - preceding_size);
        to_send = std::min(to_send, local_data_size);
        send_cnts[cur_rank] = to_send;
        local_data_size -= to_send;
        preceding_size += to_send;
    }
    send_cnts.back() += local_data_size;

    std::vector<DataType> result = mpi_util::alltoallv_combined(local_data, send_cnts, comm);
    return result;
}

// local data contains intervals of block_size that belong to one block
// block-i is distributed over all PEs
// distribute data with an alltoall such that the order of the blocks is the same over all PEs
// divide PEs into equal sized groups that will receive one block
template <typename DataType>
std::vector<DataType> transpose_blocks(std::vector<DataType>& local_data,
                                       std::vector<uint64_t> block_size,
                                       Communicator<>& comm) {
    int64_t num_blocks = block_size.size();
    KASSERT(num_blocks <= (int64_t)comm.size());

    KASSERT(local_data.size() == std::accumulate(block_size.begin(), block_size.end(), uint64_t(0)));

    // compute prefix sums
    std::vector<uint64_t> pref_sum_kth_block = comm.exscan(send_buf(block_size), op(ops::plus<>{}));
    std::vector<uint64_t> sum_kth_block =
        comm.allreduce(send_buf(block_size), op(ops::plus<>{}));


    // sort block indices by decreasing size
    std::vector<int64_t> idx_blocks(num_blocks);
    std::iota(idx_blocks.begin(), idx_blocks.end(), 0);
    std::sort(idx_blocks.begin(), idx_blocks.end(), [&](int64_t a, int64_t b) {
        return sum_kth_block[a] > sum_kth_block[b];
    });

    // divide one block amoung #PEs / #blocks
    // remainder is distributed amoung largest blocks
    std::vector<int64_t> num_pe_per_block(num_blocks, comm.size() / num_blocks);
    int64_t rem = comm.size() % num_blocks;
    for (int64_t k = 0; k < rem; k++) {
        int64_t k2 = idx_blocks[k];
        num_pe_per_block[k2]++;
    }

    // assign group of PEs to blocks
    std::vector<int64_t> pe_range(num_blocks + 1, 0);
    std::inclusive_scan(num_pe_per_block.begin(), num_pe_per_block.end(), pe_range.begin() + 1);

    // compute target sizes for alltoall
    std::vector<int64_t> target_size(comm.size(), 0);
    for (int64_t k = 0; k < num_blocks; k++) {
        for (int64_t rank = pe_range[k]; rank < pe_range[k + 1]; rank++) {
            target_size[rank] = sum_kth_block[k] / num_pe_per_block[k];
        }
        target_size[pe_range[k + 1] - 1] += sum_kth_block[k] % num_pe_per_block[k];
    }

    std::vector<int64_t> pred_target_size(comm.size(), 0);
    for (int64_t k = 0; k < num_blocks; k++) {
        std::exclusive_scan(target_size.begin() + pe_range[k],
                            target_size.begin() + pe_range[k + 1],
                            pred_target_size.begin() + pe_range[k],
                            0);
    }

    std::vector<int64_t> send_cnts(comm.size(), 0);
    for (int64_t k = 0; k < num_blocks; k++) {
        int64_t local_data_size = block_size[k];
        int64_t preceding_size = pref_sum_kth_block[k];
        int64_t last_pe = pe_range[k + 1] - 1;
        for (int rank = pe_range[k]; rank < last_pe && local_data_size > 0; rank++) {
            int64_t to_send = std::max(int64_t(0), pred_target_size[rank + 1] - preceding_size);
            to_send = std::min(to_send, local_data_size);
            send_cnts[rank] = to_send;
            local_data_size -= to_send;
            preceding_size += to_send;
        }
        send_cnts[last_pe] += local_data_size;
    }

    int64_t total_send = std::accumulate(send_cnts.begin(), send_cnts.end(), int64_t(0));
    int64_t total_sa = std::accumulate(block_size.begin(), block_size.end(), int64_t(0));
    KASSERT(total_send == total_sa);

    return mpi_util::alltoallv_combined(local_data, send_cnts, comm);
}
} // namespace dsss::mpi_util
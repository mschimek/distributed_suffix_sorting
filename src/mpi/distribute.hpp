#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/reduce.hpp"
#include "util/memory.hpp"

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
                                       std::vector<uint64_t> &block_size,
                                       Communicator<>& comm) {
    int64_t num_blocks = block_size.size();
    KASSERT(num_blocks <= (int64_t)comm.size());

    KASSERT(local_data.size()
            == std::accumulate(block_size.begin(), block_size.end(), uint64_t(0)));

    // compute prefix sums
    std::vector<uint64_t> pref_sum_kth_block = comm.exscan(send_buf(block_size), op(ops::plus<>{}));
    std::vector<uint64_t> sum_kth_block = comm.allreduce(send_buf(block_size), op(ops::plus<>{}));

    // sort block indices by decreasing size
    std::vector<int64_t> idx_blocks(num_blocks);
    std::iota(idx_blocks.begin(), idx_blocks.end(), int64_t(0));
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
                            int64_t(0));
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

/*
Rearranges data that result from space efficient bucket sorting.
Data is distributed equally amoung the PEs.
Additional bookkeeping information (#PEs x #blocks) is required to correctly reorder the received
data locally in an output buffer.
*/
template <typename DataType>
std::vector<DataType> transpose_blocks_balanced(std::vector<DataType>& local_data,
                                                std::vector<uint64_t> &block_size,
                                                Communicator<>& comm) {
    KASSERT(local_data.size()
            == std::accumulate(block_size.begin(), block_size.end(), uint64_t(0)));
    uint64_t num_blocks = block_size.size();

    // compute prefix sums of blocks
    std::vector<uint64_t> pref_sum_kth_block = comm.exscan(send_buf(block_size), op(ops::plus<>{}));
    std::vector<uint64_t> sum_kth_block = comm.allreduce(send_buf(block_size), op(ops::plus<>{}));
    uint64_t total_size = std::accumulate(sum_kth_block.begin(), sum_kth_block.end(), uint64_t(0));

    std::vector<uint64_t> global_pref_sum_block(num_blocks, 0);
    std::exclusive_scan(sum_kth_block.begin(),
                        sum_kth_block.end(),
                        global_pref_sum_block.begin(),
                        uint64_t(0));

    // compute target size
    std::vector<uint64_t> pe_target_sizes(comm.size(), total_size / comm.size());
    uint64_t rem = total_size % comm.size();
    for (uint64_t i = 0; i < rem; i++) {
        pe_target_sizes[i]++;
    }
    uint64_t target_size = pe_target_sizes[comm.rank()];
    KASSERT(total_size == mpi_util::all_reduce_sum(target_size, comm));
    KASSERT(total_size
            == std::accumulate(pe_target_sizes.begin(), pe_target_sizes.end(), uint64_t(0)));

    std::vector<uint64_t> pref_target_sizes(comm.size(), 0);
    std::exclusive_scan(pe_target_sizes.begin(),
                        pe_target_sizes.end(),
                        pref_target_sizes.begin(),
                        uint64_t(0));

    // compute send counts
    std::vector<int64_t> send_cnts(comm.size(), 0);

    // one block of num_blocks sizes per PE
    std::vector<uint64_t> block_size_send(num_blocks * comm.size());

    uint64_t local_data_size = local_data.size();
    uint64_t r = 0;
    for (uint64_t k = 0; k < num_blocks; k++) {
        uint64_t remaining_block_size = block_size[k];
        uint64_t global_index_block = global_pref_sum_block[k] + pref_sum_kth_block[k];
        uint64_t preceding_size = global_index_block;

        // assign current block
        while (r < comm.size() - 1 && remaining_block_size > 0) {
            if (preceding_size < pref_target_sizes[r + 1]) {
                uint64_t elements_left = pref_target_sizes[r + 1] - preceding_size;
                uint64_t to_send = std::min(remaining_block_size, elements_left);
                remaining_block_size -= to_send;
                local_data_size -= to_send;
                send_cnts[r] += to_send;
                preceding_size += to_send;
                block_size_send[r * num_blocks + k] = to_send;
            }
            // same PE might get part of next block
            if (remaining_block_size > 0) {
                r++;
            }
        }
        KASSERT(remaining_block_size == 0ull || r == comm.size() - 1);
        if (r == comm.size() - 1) {
            uint64_t to_send = remaining_block_size;
            remaining_block_size -= to_send;
            local_data_size -= to_send;
            send_cnts[r] += to_send;
            block_size_send[r * num_blocks + k] = to_send;
        }
    }
    KASSERT(local_data_size == 0ull);

    local_data = mpi_util::alltoallv_combined(local_data, send_cnts, comm);
    std::vector<uint64_t> block_size_rcv = comm.alltoall(send_buf(block_size_send));
    KASSERT(local_data.size() == target_size);

    // rearrange data
    std::vector<DataType> output_buffer(local_data.size());
    std::vector<uint64_t> pref_block_size_rcv(block_size_rcv.size() + 1, 0);
    std::inclusive_scan(block_size_rcv.begin(),
                        block_size_rcv.end(),
                        pref_block_size_rcv.begin() + 1);


    // copy each received region in correct order into output buffer
    uint64_t write_index = 0;
    for (uint64_t k = 0; k < num_blocks; k++) {
        for (uint64_t r = 0; r < comm.size(); r++) {
            uint64_t start = pref_block_size_rcv[r * num_blocks + k];
            uint64_t end = pref_block_size_rcv[r * num_blocks + k + 1];
            if (start < end) {
                std::copy(local_data.begin() + start,
                          local_data.begin() + end,
                          output_buffer.begin() + write_index);
                write_index += end - start;
            }
        }
    }

    free_memory(std::move(local_data));
    return output_buffer;
}

template <typename DataType>
std::vector<DataType> transpose_blocks_wrapper(std::vector<DataType>& local_data,
                                               std::vector<uint64_t> &block_size,
                                               Communicator<>& comm,
                                               bool balanced) {
    if (balanced) {
        return transpose_blocks_balanced(local_data, block_size, comm);
    } else {
        return transpose_blocks(local_data, block_size, comm);
    }
}


} // namespace dsss::mpi_util

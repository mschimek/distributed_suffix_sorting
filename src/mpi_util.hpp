#pragma once

#include "kamping/collectives/exscan.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"
#include "kassert/kassert.hpp"
#include "printing.hpp"

namespace mpi_util {

using namespace kamping;

// sends T from processor i to processor i - 1 with blocking
template <typename T>
T shift_left(T& local_value, Communicator<>& comm) {
    int process_rank = comm.rank();
    int num_processes = comm.size();
    T received_value;

    if (process_rank > 0) {
        comm.send(send_buf(local_value), destination(process_rank - 1));
    }
    if (process_rank < num_processes - 1) {
        comm.recv(recv_buf(received_value), recv_count(1), source(process_rank + 1));
    }
    return received_value;
}

// sends T from processor i to processor i - 1 with blocking
template <typename T>
std::vector<T> shift_left(std::vector<T>& local_data, int count, Communicator<>& comm) {
    int process_rank = comm.rank();
    int num_processes = comm.size();

    KASSERT(count <= local_data.size());

    std::vector<T> send_buffer(local_data.begin(), local_data.begin() + count);
    std::vector<T> recv_buffer(count);

    if (process_rank > 0) {
        comm.send(send_buf(send_buffer), destination(process_rank - 1));
    }
    if (process_rank < num_processes - 1) {
        comm.recv(recv_buf(recv_buffer), recv_count(count), source(process_rank + 1));
    }
    return recv_buffer;
}

// sums up local sizes and checks against expected_size
void check_expected_size(size_t expected_size, size_t local_size, Communicator<>& comm) {
    auto total_size = comm.reduce(send_buf(local_size), op(ops::plus<>()));
    if (comm.rank() == 0) {
        KASSERT(total_size.front() == expected_size);
    }
}

template <typename T>
T all_reduce_sum(T& local_data, Communicator<>& comm) {
    // reduce returns result only on root process
    auto total_sum = comm.reduce(send_buf(local_data), op(ops::plus<>()));
    T sum;
    if (comm.rank() == 0) {
        sum = total_sum.front();
    }
    comm.bcast_single(send_recv_buf(sum));
    return sum;
}

// adapted from: https://github.com/kurpicz/dsss/blob/master/dsss/mpi/distribute_data.hpp
// distributs data such thateach process i < comm.size() has n / comm.size() elements and the last process has the remaining elements.
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

} // namespace mpi_util
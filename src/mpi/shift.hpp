#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

namespace dsss::mpi_util {


// sends T from processor i to processor i - 1 with blocking
template <typename T>
T shift_left(T& local_value, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    int process_rank = comm.rank();
    int num_processes = comm.size();
    T received_value{};

    if (process_rank > 0) {
        comm.send(kmp::send_buf(local_value), kmp::destination(process_rank - 1));
    }
    if (process_rank < num_processes - 1) {
        comm.recv(kmp::recv_buf(received_value), kmp::recv_count(1), kmp::source(process_rank + 1));
    }
    return received_value;
}

// sends T from processor i to processor i - 1 with blocking
template <typename T>
std::vector<T> shift_left(std::vector<T>& local_data, int count, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    int process_rank = comm.rank();
    int num_processes = comm.size();

    KASSERT(count <= (int)local_data.size());

    std::vector<T> send_buffer(local_data.begin(), local_data.begin() + count);
    std::vector<T> recv_buffer(count);

    if (process_rank > 0) {
        comm.send(kmp::send_buf(send_buffer), kmp::destination(process_rank - 1));
    }
    if (process_rank < num_processes - 1) {
        comm.recv(kmp::recv_buf(recv_buffer), kmp::recv_count(count), kmp::source(process_rank + 1));
    }
    return recv_buffer;
}

// applies a left shift and pushes entries to vector of receiver
template <typename T>
void shift_entries_left(std::vector<T>& local_data, int count, kamping::Communicator<>& comm) {
    std::vector<T> recv_data = mpi_util::shift_left(local_data, count, comm);
    if (comm.rank() < comm.size() - 1) {
        local_data.insert(local_data.end(), recv_data.begin(), recv_data.end());
    }
}

// sends T from processor i to processor i + 1 with blocking
template <typename T>
T shift_right(T& local_value, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    int process_rank = comm.rank();
    int num_processes = comm.size();
    T received_value{};

    if (process_rank < num_processes - 1) {
        comm.send(kmp::send_buf(local_value), kmp::destination(process_rank + 1));
    }
    if (process_rank > 0) {
        comm.recv(kmp::recv_buf(received_value), kmp::recv_count(1), kmp::source(process_rank - 1));
    }
    return received_value;
}

// sends T from processor i to processor i + 1 with blocking
template <typename T>
std::vector<T> shift_right(std::vector<T>& local_data, int count, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    int process_rank = comm.rank();
    int num_processes = comm.size();

    KASSERT(count <= local_data.size());

    std::vector<T> send_buffer(local_data.begin(), local_data.begin() + count);
    std::vector<T> recv_buffer(count);

    if (process_rank < num_processes - 1) {
        comm.send(kmp::send_buf(send_buffer), kmp::destination(process_rank + 1));
    }
    if (process_rank > 0) {
        comm.recv(kmp::recv_buf(recv_buffer), kmp::recv_count(count), kmp::source(process_rank - 1));
    }
    return recv_buffer;
}


} // namespace dsss::mpi_util

#pragma once

#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

namespace dsss::mpi_util {

template <typename T>
T send_from_to(T data, uint64_t from, uint64_t to, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    KASSERT(from < comm.size());
    KASSERT(to < comm.size());
    KASSERT(from != to);

    T received_value{};

    if (comm.rank() == from) {
        comm.send(kmp::send_buf(data), kmp::destination(to));
    }
    if (comm.rank() == to) {
        comm.recv(kmp::recv_buf(received_value), kmp::recv_count(1), kmp::source(from));
    }
    return received_value;
}

} // namespace dsss::mpi_util

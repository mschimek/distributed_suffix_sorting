#pragma once

#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

namespace dsss::mpi_util {
using namespace kamping;

template <typename T>
T send_from_to(T data, uint64_t from, uint64_t to, Communicator<>& comm) {
    KASSERT(from < comm.size());
    KASSERT(to < comm.size());
    KASSERT(from != to);

    T received_value{};

    if (comm.rank() == from) {
        comm.send(send_buf(data), destination(to));
    }
    if (comm.rank() == to) {
        comm.recv(recv_buf(received_value), recv_count(1), source(from));
    }
    return received_value;
}

} // namespace dsss::mpi_util
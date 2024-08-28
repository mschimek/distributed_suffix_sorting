#pragma once

#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/p2p/irecv.hpp"
#include "kamping/p2p/isend.hpp"

namespace mpi_util {

// sends T from processor i to processor i - 1 without blocking
// assumes recv_buffer to have already the correct size
template <typename T>
void ishift_left(T& send_buffer, T& recv_buffer, kamping::Communicator<>& comm) {
    using namespace kamping;
    int process_rank = comm.rank();
    int num_processes = comm.size();

    if (process_rank > 0) {
        comm.isend(send_buf(send_buffer), destination(process_rank - 1));
    }
    if (process_rank < num_processes - 1) {
        comm.irecv(recv_buf(recv_buffer), source(process_rank + 1));
    }
}

// sums up local sizes and checks against expected_size
void check_expected_size(size_t expected_size, size_t local_size, kamping::Communicator<>& comm) {
    auto total_size =
        comm.reduce(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>()));
    if (comm.rank() == 0) {
        KASSERT(total_size.front() == expected_size);
    }
}

} // namespace mpi_util
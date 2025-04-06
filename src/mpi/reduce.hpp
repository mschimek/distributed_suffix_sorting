#pragma once

#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"

namespace dsss::mpi_util {

using namespace kamping;
template <typename T, typename Operation>
T all_reduce(T& local_data, Operation operation, Communicator<>& comm) {
    // reduce returns result only on root process
    auto combined = comm.reduce(send_buf(local_data), op(operation, ops::commutative));
    T combined_local;
    if (comm.rank() == 0) {
        combined_local = combined.front();
    }
    comm.bcast_single(send_recv_buf(combined_local));
    return combined_local;
}

template <typename T>
T all_reduce_max(T local_data, Communicator<>& comm) {
    return all_reduce(local_data, ops::max<>(), comm);
}

template <typename T>
T all_reduce_max(std::vector<T> const& local_data, Communicator<>& comm) {
    T local_max = *max_element(local_data.begin(), local_data.end());
    return all_reduce_max(local_max, comm);
}

template <typename T>
T all_reduce_min(T local_data, Communicator<>& comm) {
    return all_reduce(local_data, ops::min<>(), comm);
}

template <typename T>
T all_reduce_sum(T local_data, Communicator<>& comm) {
    return all_reduce(local_data, ops::plus<>(), comm);
}

template <typename T>
bool all_reduce_and(T local_data, Communicator<>& comm) {
    int local_bool = local_data;
    return (bool)all_reduce(local_bool, ops::bit_and<>(), comm);
}

template <typename T>
T ex_prefix_sum(T local_data, Communicator<>& comm) {
    auto local_sum = comm.exscan(send_buf(local_data), op(ops::plus<>{}));
    return local_sum.front();
}
} // namespace dsss::mpi_util

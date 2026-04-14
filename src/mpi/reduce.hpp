#pragma once

#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"

namespace dsss::mpi_util {

template <typename T, typename Operation>
T all_reduce(T& local_data, Operation operation, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    // reduce returns result only on root process
    auto combined = comm.reduce(kmp::send_buf(local_data), kmp::op(operation, kamping::ops::commutative));
    T combined_local;
    if (comm.rank() == 0) {
        combined_local = combined.front();
    }
    comm.bcast_single(kmp::send_recv_buf(combined_local));
    return combined_local;
}

template <typename T>
T all_reduce_max(T local_data, kamping::Communicator<>& comm) {
    return all_reduce(local_data, kamping::ops::max<>(), comm);
}

template <typename T>
T all_reduce_max(std::vector<T> const& local_data, kamping::Communicator<>& comm) {
    T local_max = *max_element(local_data.begin(), local_data.end());
    return all_reduce_max(local_max, comm);
}

template <typename T>
T all_reduce_min(T local_data, kamping::Communicator<>& comm) {
    return all_reduce(local_data, kamping::ops::min<>(), comm);
}

template <typename T>
T all_reduce_sum(T local_data, kamping::Communicator<>& comm) {
    return all_reduce(local_data, kamping::ops::plus<>(), comm);
}

template <typename T>
bool all_reduce_and(T local_data, kamping::Communicator<>& comm) {
    int local_bool = local_data;
    return (bool)all_reduce(local_bool, kamping::ops::bit_and<>(), comm);
}

template <typename T>
T ex_prefix_sum(T local_data, kamping::Communicator<>& comm) {
    namespace kmp = kamping::params;
    if (comm.rank() == 0) {
        if (comm.size() > 1) {
            comm.exscan(kmp::send_buf(local_data), kmp::op(kamping::ops::plus<>{}));
        }
        return T{0};
    }
    auto local_sum = comm.exscan(kmp::send_buf(local_data), kmp::op(kamping::ops::plus<>{}));
    return local_sum.front();
}
} // namespace dsss::mpi_util

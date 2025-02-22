// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <tlx/die/core.hpp>

#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/big_type.hpp"
#include "util/printing.hpp"

namespace dsss::mpi_util {

using namespace kamping;

template <typename SendBuf>
auto alltoallv_native(SendBuf&& send_buffer,
                      std::span<int64_t> send_counts,
                      std::span<int64_t> recv_counts,
                      Communicator<>& comm) {
    KASSERT(std::all_of(send_counts.begin(), send_counts.end(), std::in_range<int, int64_t>),
            "all send counts need to fit into an int",
            kamping::assert::normal);
    KASSERT(std::all_of(recv_counts.begin(), recv_counts.end(), std::in_range<int, int64_t>),
            "all recv counts need to fit into an int",
            kamping::assert::normal);

    std::vector<int> send_counts_int{send_counts.begin(), send_counts.end()};
    std::vector<int> recv_counts_int{recv_counts.begin(), recv_counts.end()};
    DBG("call alltoallv native");

    return comm.alltoallv(kamping::send_buf(send_buffer),
                          kamping::send_counts(send_counts_int),
                          kamping::recv_counts(recv_counts_int));
}

template <typename SendBuf>
auto alltoallv_direct(SendBuf&& send_buf,
                      std::span<int64_t> send_counts,
                      std::span<int64_t> recv_counts,
                      Communicator<>& comm) {
    DBG("call alltoallv direct start");

    using DataType = std::remove_reference_t<SendBuf>::value_type;
    std::vector<size_t> send_displs(comm.size()), recv_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    auto const recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType> receive_data(recv_total);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * comm.size());

    DBG("call irecv");

    for (int i = 0; i < comm.size_signed(); ++i) {
        int source = (comm.rank_signed() + (comm.size_signed() - i)) % comm.size_signed();
        if (recv_counts[source] > 0) {
            auto receive_type = mpi_util::get_big_type<DataType>(recv_counts[source]);
            MPI_Irecv(receive_data.data() + recv_displs[source],
                      1,
                      receive_type,
                      source,
                      44227,
                      comm.mpi_communicator(),
                      &requests.emplace_back(MPI_REQUEST_NULL));
        }
    }
    DBG("call isend");

    for (int i = 0; i < comm.size_signed(); ++i) {
        int target = (comm.rank_signed() + i) % comm.size_signed();
        if (send_counts[target] > 0) {
            auto send_type = mpi_util::get_big_type<DataType>(send_counts[target]);
            MPI_Issend(send_buf.data() + send_displs[target],
                       1,
                       send_type,
                       target,
                       44227,
                       comm.mpi_communicator(),
                       &requests.emplace_back(MPI_REQUEST_NULL));
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    DBG("waitall");

    return receive_data;
}

template <typename SendBuf>
auto alltoallv_combined(SendBuf&& send_buffer,
                        std::span<int64_t> send_counts,
                        std::span<int64_t> recv_counts,
                        Communicator<>& comm) {
    DBG("accumlate counts");
    int64_t const send_total = std::accumulate(send_counts.begin(), send_counts.end(), int64_t{0});
    int64_t const recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), int64_t{0});
    int64_t const local_max = std::max<int64_t>(send_total, recv_total);
    int64_t const global_max = comm.allreduce_single(send_buf(local_max), op(ops::max<>{}));

    DBG("local max: " + std::to_string(local_max) + ", global max: " + std::to_string(global_max));

    if (global_max < std::numeric_limits<int>::max()) {
        DBG("using native alltoall");
        return alltoallv_native(send_buffer, send_counts, recv_counts, comm);
    } else {
        DBG("using direct alltoall");
        return alltoallv_direct(send_buffer, send_counts, recv_counts, comm);
    }
    DBG("using direct alltoall");
}

template <typename SendBuf>
auto alltoallv_combined(SendBuf&& send_buffer,
                        std::span<int64_t> send_counts,
                        Communicator<>& comm) {
    DBG("get recv counts");
    auto recv_counts = comm.alltoall(send_buf(send_counts));
    DBG("call alltoallv combined");
    return alltoallv_combined(std::forward<SendBuf>(send_buffer), send_counts, recv_counts, comm);
}

std::vector<bool> alltoallv_packed_bits(std::vector<bool>& bits,
                                        std::vector<int64_t>& send_counts,
                                        std::vector<int64_t>& recv_counts,
                                        kamping::Communicator<>& comm) {
    using PackType = uint64_t;
    constexpr uint64_t BITS = sizeof(PackType) * 8;

    // pack bits into words
    std::vector<int64_t> entries_pe(comm.size(), 0);
    std::vector<PackType> send_masks;
    int64_t idx_bits = 0;
    for (uint64_t r = 0; r < comm.size(); r++) {
        int64_t cnt = 0;
        while (cnt < send_counts[r]) {
            PackType mask = 0;
            for (uint64_t b = 0; b < BITS && cnt < send_counts[r]; b++) {
                if (bits[idx_bits]) {
                    mask |= (1ull << b);
                }
                idx_bits++;
                cnt++;
            }
            send_masks.push_back(mask);
            entries_pe[r]++;
        }
    }

    // send packed bits
    send_masks = mpi_util::alltoallv_combined(send_masks, entries_pe, comm);

    // unpack masks
    std::vector<bool> recv_bits;
    uint64_t mask_idx = 0;
    for (uint64_t r = 0; r < comm.size(); r++) {
        int64_t cnt = 0;
        while (cnt < recv_counts[r]) {
            for (uint64_t b = 0; b < BITS && cnt < recv_counts[r]; b++) {
                bool bit_set = send_masks[mask_idx] & (1ull << b);
                recv_bits.push_back(bit_set);
                cnt++;
            }
            mask_idx++;
        }
    }
    KASSERT(mask_idx == send_masks.size());
    return recv_bits;
}


} // namespace dsss::mpi_util

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "hash/xxhash.hpp"
#include "ips4o.hpp"
#include "ips4o/ips4o_fwd.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/sample_sort_config.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/uint_types.hpp"

namespace dsss::distinguishing_prefix {

using LengthType = uint16_t;
using HashType = uint64_t;
using CanditateIndex = uint32_t;
const static HashType HASH_SEED = 13358259232739045019ull;

template <typename ContiguousIterator>
HashType compute_hash(ContiguousIterator begin, ContiguousIterator end) {
    return xxh::xxhash3<64>(begin, end, HASH_SEED);
}

struct HashAndInt {
    HashType hash;
    uint32_t value;
    bool operator<(const HashAndInt& other) const { return hash < other.hash; }
    bool operator!=(const HashAndInt& other) const { return hash != other.hash; }
    bool operator==(const HashAndInt& other) const { return hash == other.hash; }

    std::string to_string() const {
        return "(" + std::to_string(hash) + ", " + std::to_string(value) + ")";
    }
};

template <typename DataType>
std::vector<HashAndInt> compute_local_hashes(std::vector<CanditateIndex>& candidates,
                                             std::vector<DataType>& local_data,
                                             uint64_t prefix_length) {
    std::vector<HashAndInt> local_hashes(candidates.size());
    std::transform(candidates.begin(),
                   candidates.end(),
                   local_hashes.begin(),
                   [&](CanditateIndex candidate) {
                       HashType hash =
                           compute_hash(local_data[candidate].chars.begin(),
                                        local_data[candidate].chars.begin() + prefix_length);
                       return HashAndInt(hash, candidate);
                   });
    return local_hashes;
}

std::vector<HashType> local_deduplication(std::vector<HashAndInt>& local_hashes) {
    std::vector<HashType> local_unique_hashes(local_hashes.size());
    std::transform(local_hashes.begin(),
                   local_hashes.end(),
                   local_unique_hashes.begin(),
                   [](HashAndInt hash_and_candidate) { return hash_and_candidate.hash; });
    auto it = std::unique(local_unique_hashes.begin(), local_unique_hashes.end());
    local_unique_hashes.resize(std::distance(local_unique_hashes.begin(), it));
    return local_unique_hashes;
}

std::vector<HashAndInt> pair_hash_with_pe(std::vector<HashType>& local_unique_hashes,
                                          std::vector<int64_t>& recv_counts) {
    std::vector<HashAndInt> hash_and_pe;
    hash_and_pe.reserve(local_unique_hashes.size());
    uint64_t k = 0;
    for (uint64_t r = 0; r < recv_counts.size(); r++) {
        for (int64_t i = 0; i < recv_counts[r]; i++) {
            hash_and_pe.emplace_back(local_unique_hashes[k], r);
            k++;
        }
    };
    return hash_and_pe;
}

bool distinct_hashes(std::vector<HashAndInt>& hashes, uint64_t i) {
    if (i > 0 && hashes[i] == hashes[i - 1]) {
        return false;
    }
    if (i + 1 < hashes.size() && hashes[i] == hashes[i + 1]) {
        return false;
    }
    return true;
};


std::vector<bool> get_unique_mask(std::vector<HashAndInt>& hash_and_pe,
                                  std::vector<int64_t>& recv_counts,
                                  kamping::Communicator<>& comm) {
    ips4o::sort(hash_and_pe.begin(), hash_and_pe.end());
    uint64_t recv_elements = std::accumulate(recv_counts.begin(), recv_counts.end(), uint64_t(0));
    std::vector<bool> is_unique(recv_elements, false);
    std::vector<int64_t> index_pe(comm.size(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), index_pe.begin(), int64_t(0));
    for (uint64_t i = 0; i < hash_and_pe.size(); i++) {
        uint64_t pe = hash_and_pe[i].value;
        is_unique[index_pe[pe]++] = distinct_hashes(hash_and_pe, i);
    }
    return is_unique;
}

std::vector<bool> pack_and_send_mask(std::vector<bool>& is_unique,
                                     std::vector<int64_t>& send_counts,
                                     std::vector<int64_t>& recv_counts,
                                     kamping::Communicator<>& comm) {
    // pack response into 64-bit integers
    std::vector<int64_t> entries_pe(comm.size(), 0);
    std::vector<uint64_t> unique_hash_mask;
    int64_t idx = 0;
    for (uint64_t r = 0; r < comm.size(); r++) {
        int64_t k = 0;
        while (k < recv_counts[r]) {
            uint64_t mask = 0;
            for (uint64_t b = 0; b < 64 && k < recv_counts[r]; b++) {
                if (is_unique[idx]) {
                    mask |= (1ull << b);
                }
                idx++;
                k++;
            }
            unique_hash_mask.push_back(mask);
            entries_pe[r]++;
        }
    }

    // send packed bits
    unique_hash_mask = mpi_util::alltoallv_combined(unique_hash_mask, entries_pe, comm);

    // unpack mask
    std::vector<bool> is_hash_unique;
    uint64_t mask_idx = 0;
    for (uint64_t r = 0; r < comm.size(); r++) {
        int64_t cnt_bits = 0;
        while (cnt_bits < send_counts[r]) {
            for (uint64_t b = 0; b < 64 && cnt_bits < send_counts[r]; b++) {
                bool unique = unique_hash_mask[mask_idx] & (1ull << b);
                is_hash_unique.push_back(unique);
                cnt_bits++;
            }
            mask_idx++;
        }
    }
    KASSERT(mask_idx == unique_hash_mask.size());
    return is_hash_unique;
}


void update_candiates_and_prefix_length(std::vector<bool>& hash_is_unique,
                                        std::vector<HashAndInt>& local_hashes,
                                        std::vector<CanditateIndex>& candidates,
                                        std::vector<LengthType>& unique_prefix_length,
                                        LengthType prefix_length,
                                        kamping::Communicator<>& comm) {
    std::vector<CanditateIndex> next_candidates;
    uint64_t idx_candidate = 0;
    uint64_t idx_hash = 0;
    while (idx_hash < hash_is_unique.size()) {
        HashType hash = local_hashes[idx_candidate].hash;
        bool local_unique = idx_candidate + 1 == local_hashes.size()
                            || local_hashes[idx_candidate] != local_hashes[idx_candidate + 1];
        bool is_unique = hash_is_unique[idx_hash] && local_unique;

        while (idx_candidate < candidates.size() && local_hashes[idx_candidate].hash == hash) {
            CanditateIndex candidate = local_hashes[idx_candidate].value;
            if (is_unique) {
                unique_prefix_length[candidate] = prefix_length;
            } else {
                next_candidates.push_back(candidate);
            }
            idx_candidate++;
        }
        idx_hash++;
    }
    std::swap(candidates, next_candidates);
}

template <typename DataType>
std::vector<LengthType> prefix_doubling(kamping::Communicator<>& comm,
                                        std::vector<DataType>& local_data,
                                        LengthType inital_prefix_length = 4) {
    auto& timer = kamping::measurements::timer();

    LengthType prefix_length = inital_prefix_length;
    LengthType max_string_length = local_data[0].chars.size();
    std::vector<LengthType> unique_prefix_length(local_data.size(), max_string_length);

    std::vector<CanditateIndex> candidates(local_data.size());
    std::iota(candidates.begin(), candidates.end(), CanditateIndex(0));

    const HashType LOCAL_HASH_RANGE = std::numeric_limits<HashType>::max() / comm.size();
    std::vector<HashType> hash_splitters(comm.size() - 1, LOCAL_HASH_RANGE);
    std::inclusive_scan(hash_splitters.begin(), hash_splitters.end(), hash_splitters.begin());

    mpi::SampleSortConfig config;
    config.use_binary_search_for_splitters = true;

    while (prefix_length < max_string_length) {
        // compute hashes
        timer.synchronize_and_start("prefix_doubling_compute_hashes");
        std::vector<HashAndInt> local_hashes =
            compute_local_hashes(candidates, local_data, prefix_length);
        timer.stop();

        // remove duplicates
        timer.synchronize_and_start("prefix_doubling_remove_local_duplicates");
        ips4o::sort(local_hashes.begin(), local_hashes.end());
        std::vector<HashType> local_unique_hashes = local_deduplication(local_hashes);
        timer.stop();

        // send hashes
        timer.synchronize_and_start("prefix_doubling_send_hashes");
        std::vector<int64_t> send_counts = dsss::mpi::compute_interval_sizes(local_unique_hashes,
                                                                             hash_splitters,
                                                                             comm,
                                                                             std::less<HashType>{},
                                                                             config);

        std::vector<int64_t> recv_counts = comm.alltoall(kamping::send_buf(send_counts));
        local_unique_hashes = mpi_util::alltoallv_combined(local_unique_hashes, send_counts, comm);
        timer.stop();

        // local duplicate detection
        timer.synchronize_and_start("prefix_doubling_process_incoming_hashes");
        std::vector<HashAndInt> hash_and_pe = pair_hash_with_pe(local_unique_hashes, recv_counts);
        free_memory(std::move(local_unique_hashes));
        std::vector<bool> local_is_unique = get_unique_mask(hash_and_pe, recv_counts, comm);
        free_memory(std::move(hash_and_pe));
        timer.stop();

        // send unique information back to original PE
        timer.synchronize_and_start("prefix_doubling_send_hashes_back");
        std::vector<bool> hash_is_unique =
            mpi_util::alltoallv_packed_bits(local_is_unique, recv_counts, send_counts, comm);
        free_memory(std::move(local_is_unique));
        timer.stop();

        // set prefix length of unique candidates
        timer.synchronize_and_start("prefix_doubling_update_candiates_and_prefix_length");
        update_candiates_and_prefix_length(hash_is_unique,
                                           local_hashes,
                                           candidates,
                                           unique_prefix_length,
                                           prefix_length,
                                           comm);
        timer.stop();

        prefix_length *= 2;
    }

    return unique_prefix_length;
}

}; // namespace dsss::distinguishing_prefix
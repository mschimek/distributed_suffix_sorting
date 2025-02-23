#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "ips4o/ips4o.hpp"
#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"
#include "pdcx/config.hpp"
#include "util/division.hpp"
#include "util/string_util.hpp"

namespace dsss::chunking {

using namespace kamping;
using namespace dcx;

template <typename char_type, typename index_type>
struct Chunking {
    Chunking(Communicator<>& _comm, PDCXLengthInfo& _info, uint64_t _avg_chunks_pe)
        : comm(_comm),
          info(_info) {
        uint64_t local_chars_with_dummy = info.local_chars_with_dummy;
        uint64_t total_chars = mpi_util::all_reduce_sum(local_chars_with_dummy, comm);
        uint64_t avg_local_chars = util::div_ceil(total_chars, comm.size());
        avg_chunks_pe = std::max(uint64_t(1), _avg_chunks_pe);
        avg_chunks_pe = std::min(avg_chunks_pe, local_chars_with_dummy);
        chunk_size = util::div_ceil(avg_local_chars, avg_chunks_pe);
        
        // ensure chunk size is the same for every PE
        chunk_size = mpi_util::all_reduce_min(chunk_size, comm);
        num_local_chunks = util::div_ceil(local_chars_with_dummy, chunk_size);
    }

    Communicator<>& comm;
    PDCXLengthInfo& info;
    uint64_t avg_chunks_pe;
    uint64_t num_local_chunks;
    uint64_t chunk_size;

    struct Chunk {
        index_type start_index;
        uint32_t target_pe;
    };

    uint64_t get_chunk_size() const { return chunk_size; }

    std::vector<Chunk> get_random_chunks(uint64_t seed) {
        std::mt19937 rng(seed + comm.rank());
        std::uniform_int_distribution<uint32_t> dist(0, comm.size() - 1);

        // create chunks
        std::vector<Chunk> chunks;
        chunks.reserve(num_local_chunks);
        for (uint64_t i = 0; i < num_local_chunks; i++) {
            Chunk chunk = {i * chunk_size, dist(rng)};
            chunks.push_back(chunk);
            KASSERT(chunk.start_index < info.local_chars_with_dummy);
        }

        // sort chunks by PE
        ips4o::sort(chunks.begin(), chunks.end(), [&](const Chunk& a, const Chunk& b) {
            if (a.target_pe != b.target_pe) {
                return a.target_pe < b.target_pe;
            }
            return a.start_index < b.start_index;
        });

        return chunks;
    }

    std::vector<int64_t> get_send_counts(std::vector<Chunk>& chunks, int64_t chunk_with_padding) {
        std::vector<int64_t> send_cnt(comm.size(), 0);
        for (auto& chunk: chunks) {
            send_cnt[chunk.target_pe] += chunk_with_padding;
        }
        return send_cnt;
    }

    std::vector<index_type> get_chunk_global_index(std::vector<Chunk>& chunks,
                                                   uint64_t chars_before) {
        std::vector<index_type> chunk_global_index =
            extract_attribute<Chunk, index_type>(chunks, [&](Chunk& c) {
                return index_type(chars_before + c.start_index);
            });
        return chunk_global_index;
    }

    std::vector<char_type> get_chunked_chars(std::vector<Chunk>& chunks,
                                             uint64_t chars_with_padding,
                                             std::vector<char_type>& local_string) {
        KASSERT(num_local_chunks == chunks.size());
        char_type fill_char = char_type(0);
        std::vector<char_type> chunked_chars(num_local_chunks * chars_with_padding, fill_char);
        auto write_it = chunked_chars.begin();
        for (auto& chunk: chunks) {
            auto start = local_string.begin() + chunk.start_index;
            uint64_t len = std::min(chars_with_padding, local_string.size() - chunk.start_index);
            KASSERT(start + len <= local_string.end());
            std::copy_n(start, len, write_it);
            write_it += chars_with_padding;
            KASSERT(write_it <= chunked_chars.end());
        }
        KASSERT(write_it == chunked_chars.end());
        return chunked_chars;
    }

    std::vector<index_type> get_chunked_ranks(std::vector<Chunk>& chunks,
                                              uint64_t ranks_with_padding,
                                              std::vector<index_type>& local_ranks,
                                              auto get_ranks_pos,
                                              index_type padding_rank) {
        KASSERT(num_local_chunks == chunks.size());
        std::vector<index_type> chunked_ranks(num_local_chunks * ranks_with_padding, padding_rank);
        auto write_it = chunked_ranks.begin();
        for (auto& chunk: chunks) {
            uint64_t first_rank = get_ranks_pos(local_ranks, chunk.start_index);
            auto start = local_ranks.begin() + first_rank;
            uint64_t len = std::min(ranks_with_padding, local_ranks.size() - first_rank);
            KASSERT(start + len <= local_ranks.end());
            std::copy_n(start, len, write_it);
            write_it += ranks_with_padding;
            KASSERT(write_it <= chunked_ranks.end());
        }
        KASSERT(write_it == chunked_ranks.end());
        return chunked_ranks;
    }

    std::vector<uint32_t> get_chunk_sizes(std::vector<Chunk>& chunks,
                                          uint64_t local_chars_with_dummy) {
        std::vector<uint32_t> chunk_sizes;
        for (auto& chunk: chunks) {
            uint32_t size = std::min(chunk_size, (local_chars_with_dummy - chunk.start_index));
            chunk_sizes.push_back(size);
        }
        return chunk_sizes;
    }
};
} // namespace dsss::chunking
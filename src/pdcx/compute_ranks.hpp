#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <kamping/measurements/counter.hpp>

#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/p2p.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "pdcx/chunking.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/packing.hpp"
#include "pdcx/redistribute.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/space_efficient_sort.hpp"
#include "pdcx/statistics.hpp"
#include "util/division.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"

//******* Start Phase 2: Construct Ranks  ********

namespace dsss::dcx {

using namespace kamping;

template <typename char_type, typename index_type, typename DC>
struct DCRankIndex {
    index_type rank;
    index_type index;
    bool unique;

    static bool cmp_by_index(const DCRankIndex& a, const DCRankIndex& b) {
        return a.index < b.index;
    }
    static bool cmp_by_rank(const DCRankIndex& a, const DCRankIndex& b) { return a.rank < b.rank; }
    static bool cmp_mod_div(const DCRankIndex& a, const DCRankIndex& b) {
        const int a_mod = a.index % DC::X;
        const int b_mod = b.index % DC::X;
        if (a_mod != b_mod) {
            return a_mod < b_mod;
        }
        return a.index / DC::X < b.index / DC::X;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << "(" << rank << "," << index << "," << unique << ")";
        return ss.str();
    }
};

template <typename char_type, typename index_type, typename DC, typename SampleString>
struct LexicographicRankPhase {
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using BucketMappingType = SpaceEfficientSort<char_type, index_type, DC>::BucketMappingType;

    Communicator<>& comm;
    PDCXConfig const& config;
    PDCXLengthInfo& info;
    static constexpr uint32_t X = DC::X;

    LexicographicRankPhase(Communicator<>& _comm, PDCXConfig const& _config, PDCXLengthInfo& _info)
        : comm(_comm),
          config(_config),
          info(_info) {}

    // shift one sample left be able to compute rank of last element
    void shift_samples_left(std::vector<SampleString>& local_samples) const {
        // adds a dummy sample for last process
        KASSERT(local_samples.size() >= 1u);
        SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
        local_samples.push_back(recv_sample);
        local_samples.shrink_to_fit();
    }

    std::vector<RankIndex>
    compute_lexicographic_ranks(std::vector<SampleString>& local_samples) const {
        std::vector<RankIndex> local_ranks;

        // exclude sample from process i + 1
        uint64_t num_ranks = local_samples.size() - 1;
        local_ranks.reserve(num_ranks);

        // compute local ranks
        uint64_t prev_rank = 0;
        for (uint64_t i = 0; i < num_ranks; i++) {
            KASSERT(i + 1 < local_samples.size());
            local_ranks.emplace_back(index_type(prev_rank), local_samples[i].index, false);
            uint64_t changed = local_samples[i].chars != local_samples[i + 1].chars ? 1 : 0;
            prev_rank += changed;
        }

        // shift ranks by 1 + prefix sum
        uint64_t ranks_before = mpi_util::ex_prefix_sum(prev_rank, comm);
        std::for_each(local_ranks.begin(), local_ranks.end(), [&](RankIndex& x) {
            x.rank += index_type(1 + ranks_before);
        });
        return local_ranks;
    }

    void flag_unique_ranks(std::vector<RankIndex>& local_ranks) const {
        KASSERT(local_ranks.size() >= 2u);
        uint64_t num_ranks = local_ranks.size();
        index_type rank_before = mpi_util::shift_right(local_ranks.back().rank, comm);
        index_type rank_after = mpi_util::shift_left(local_ranks.front().rank, comm);
        auto distinct = [](index_type a, index_type b, index_type c) { return a != b && a != c; };

        // first and last element
        index_type first_rank = local_ranks[0].rank;
        index_type second_rank = local_ranks[1].rank;
        index_type second_to_last_rank = local_ranks[num_ranks - 2].rank;
        index_type last_rank = local_ranks[num_ranks - 1].rank;
        local_ranks[0].unique = distinct(first_rank, rank_before, second_rank);
        local_ranks[num_ranks - 1].unique = distinct(last_rank, second_to_last_rank, rank_after);

        for (uint64_t i = 1; i < num_ranks - 1; i++) {
            index_type rank = local_ranks[i].rank;
            index_type prev_rank = local_ranks[i - 1].rank;
            index_type next_rank = local_ranks[i + 1].rank;
            local_ranks[i].unique = distinct(rank, prev_rank, next_rank);
        }
    }

    // create lexicographic ranks and flag unique ranks
    // sideeffect: shifts one sample from next PE to local_string
    std::vector<RankIndex>
    create_lexicographic_ranks(std::vector<SampleString>& local_samples) const {
        shift_samples_left(local_samples);
        std::vector<RankIndex> local_ranks = compute_lexicographic_ranks(local_samples);
        flag_unique_ranks(local_ranks);
        return local_ranks;
    }

    void process_bucket_samples(auto& phase1,
                                std::vector<SampleString>& samples,
                                std::vector<RankIndex>& concat_rank_buckets,
                                std::vector<uint64_t>& received_size,
                                SampleString& prev_sample,
                                uint64_t bucket_nr,
                                bool use_packing) {
        auto& timer = measurements::timer();
        // Phase 1: sort dc-samples
        timer.synchronize_and_start("phase_01_02_sort_samples");
        phase1.sort_samples(samples, use_packing);
        timer.stop();

        // Phase 2: compute lexicographic ranks
        timer.synchronize_and_start("phase_01_02_process_samples");
        redistribute_if_imbalanced(samples, config.min_imbalance, comm);
        shift_samples_left(samples);

        if (bucket_nr != 0) {
            SampleString first_sample =
                mpi_util::send_from_to(samples.front(), 0, comm.size() - 1, comm);
            if (comm.rank() == comm.size() - 1) {
                // last PE compared samples with Padding --> change is always 1
                // if there was no change revert to 0
                concat_rank_buckets.back().rank -= prev_sample.chars == first_sample.chars ? 1 : 0;
            }
        }
        // skip padding sample
        prev_sample = samples[samples.size() - 2];

        // exclude sample from process i + 1
        uint64_t num_ranks = samples.size() - 1;

        // only store changes
        uint64_t last_changed = 0;
        for (uint64_t i = 0; i < num_ranks; i++) {
            last_changed = samples[i].chars != samples[i + 1].chars ? 1 : 0;
            concat_rank_buckets.emplace_back(index_type(last_changed), samples[i].index, false);
        }

        received_size.push_back(num_ranks);
        samples.clear();
        timer.stop();
    }

    void compute_ranks_from_changes(std::vector<RankIndex>& local_ranks) {
        // compute local ranks
        uint64_t prev_rank = 0;
        for (uint64_t i = 0; i < local_ranks.size(); i++) {
            uint64_t change = local_ranks[i].rank;
            local_ranks[i].rank = index_type(prev_rank);
            prev_rank += change;
        }

        // shift ranks by 1 + prefix sum
        uint64_t ranks_before = mpi_util::ex_prefix_sum(prev_rank, comm);
        std::for_each(local_ranks.begin(), local_ranks.end(), [&](RankIndex& x) {
            x.rank += index_type(1 + ranks_before);
        });
    }

    std::vector<RankIndex> create_ranks_space_efficient(auto& phase1,
                                                        std::vector<char_type>& local_string,
                                                        const uint64_t num_buckets,
                                                        const bool use_packing = false) {
        using SpaceEfficient = SpaceEfficientSort<char_type, index_type, DC>;
        using Splitter = typename SpaceEfficient::Splitter;
        const uint32_t X = DC::X;

        auto& timer = measurements::timer();
        SpaceEfficient space_efficient(comm, config);
        double packing_ratio = use_packing ? config.packing_ratio : 1;

        auto materialize_sample = [&](uint64_t i) {
            return phase1.materialize_sample(local_string, i, packing_ratio);
        };

        // determine bucket splitters
        std::vector<Splitter> bucket_splitter =
            space_efficient.random_sample_splitters(info.local_chars, num_buckets, local_string);

        // assign dc-substrings to blocks
        std::vector<uint64_t> bucket_sizes(num_buckets, 0);
        std::vector<BucketMappingType> sample_to_bucket(local_string.size(), num_buckets);
        KASSERT(num_buckets <= std::numeric_limits<BucketMappingType>::max());

        uint64_t offset = info.chars_before % X;
        uint64_t _local_sample_size = 0;
        // add dummy sample for last PE
        for (uint64_t i = 0; i < info.local_chars_with_dummy; i++) {
            uint64_t m = (i + offset) % X;
            if (is_in_dc<DC>(m)) {
                _local_sample_size++;
                BucketMappingType block_id = num_buckets - 1;
                for (uint64_t j = 0; j < num_buckets - 1; j++) {
                    if (cmp_index_substring(local_string, i, bucket_splitter[j], X - 1)) {
                        block_id = j;
                        break;
                    }
                }
                bucket_sizes[block_id]++;
                sample_to_bucket[i] = block_id;
            }
        }
        KASSERT(_local_sample_size == info.local_sample_size);

        std::vector<SampleString> samples;
        std::vector<RankIndex> concat_rank_buckets;
        std::vector<uint64_t> received_size;
        received_size.reserve(num_buckets);

        // log imbalance of buckets
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, info.total_sample_size, comm);
        get_stats_instance().bucket_imbalance_samples.push_back(bucket_imbalance);
        get_local_stats_instance().input_bucket_imbalance_dcx_samples.push_back(
            get_max_local_imbalance(bucket_sizes, info.total_sample_size, comm.size()));


        SampleString prev_sample;
        // sorting in each round one blocks of materialized samples
        for (uint64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_01_02_space_efficient_sort_collect_buckets");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);
            for (uint64_t idx = 0; idx < info.local_chars_with_dummy; idx++) {
                if (sample_to_bucket[idx] == k) {
                    index_type index = index_type(info.chars_before + idx);
                    auto chars = materialize_sample(idx);
                    samples.push_back(SampleString(std::move(chars), index));
                }
            }
            timer.stop();
            KASSERT(bucket_sizes[k] == samples.size());
            process_bucket_samples(phase1,
                                   samples,
                                   concat_rank_buckets,
                                   received_size,
                                   prev_sample,
                                   k,
                                   use_packing);
        }
        KASSERT(mpi_util::all_reduce_sum(concat_rank_buckets.size(), comm)
                == info.total_sample_size);
        double bucket_imbalance_received =
            get_imbalance_bucket(received_size, info.total_sample_size, comm);
        get_stats_instance().bucket_imbalance_samples_received.push_back(bucket_imbalance_received);
        get_local_stats_instance().output_bucket_imbalance_dcx_samples.push_back(get_max_local_imbalance(received_size, info.total_sample_size, comm.size()));

        timer.synchronize_and_start("phase_01_02_space_efficient_sort_alltoall");
        std::vector<RankIndex> local_ranks =
            mpi_util::transpose_blocks_wrapper(concat_rank_buckets,
                                               received_size,
                                               comm,
                                               config.rearrange_buckets_balanced);
        timer.stop();

        compute_ranks_from_changes(local_ranks);
        flag_unique_ranks(local_ranks);
        return local_ranks;
    }

    std::vector<RankIndex>
    create_ranks_space_efficient_chunking(auto& phase1,
                                          std::vector<char_type>& local_string,
                                          const uint64_t num_buckets,
                                          const bool use_packing = false) {
        using SpaceEfficient = SpaceEfficientSort<char_type, index_type, DC>;
        using Splitter = typename SpaceEfficient::Splitter;

        auto& timer = measurements::timer();

        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        SpaceEfficient space_efficient(comm, config);

        // determine bucket splitters
        std::vector<Splitter> bucket_splitter =
            space_efficient.random_sample_splitters(info.local_chars, num_buckets, local_string);

        // chunking
        timer.synchronize_and_start("phase_01_02_chunking_create_chunks");
        chunking::Chunking<char_type, index_type> chunking(comm, info, config.avg_chunks_pe);
        using Chunk = chunking::Chunking<char_type, index_type>::Chunk;
        std::vector<Chunk> chunks = chunking.get_random_chunks(config.seed);
        get_stats_instance().chunk_sizes_phase1.push_back(chunking.get_chunk_size());
        get_local_stats_instance().chunk_sizes_phase1.push_back(chunking.get_chunk_size());

        // add padding to be able to materialize last suffix in chunk
        uint64_t chars_with_padding = chunking.get_chunk_size() + char_packing_ratio * X;

        // store global index of beginning of each chunk
        std::vector<char_type> chunked_chars =
            chunking.get_chunked_chars(chunks, chars_with_padding, local_string);
        std::vector<index_type> chunk_global_index =
            chunking.get_chunk_global_index(chunks, info.chars_before);
        std::vector<uint32_t> chunk_sizes =
            chunking.get_chunk_sizes(chunks, info.local_chars_with_dummy);
        std::vector<int64_t> send_cnt_chars = chunking.get_send_counts(chunks, chars_with_padding);
        std::vector<int64_t> send_cnt_index = chunking.get_send_counts(chunks, 1);
        timer.stop();

        // sanity checks
        KASSERT(std::accumulate(send_cnt_chars.begin(), send_cnt_chars.end(), int64_t(0))
                == (int64_t)chunked_chars.size());
        KASSERT(std::accumulate(send_cnt_index.begin(), send_cnt_index.end(), int64_t(0))
                == (int64_t)chunk_global_index.size());

        // exchange linearized data
        timer.synchronize_and_start("phase_01_02_chunking_alltoall");
        chunked_chars = mpi_util::alltoallv_combined(chunked_chars, send_cnt_chars, comm);
        chunk_global_index = mpi_util::alltoallv_combined(chunk_global_index, send_cnt_index, comm);
        chunk_sizes = mpi_util::alltoallv_combined(chunk_sizes, send_cnt_index, comm);
        uint64_t received_chunks = chunk_global_index.size();
        timer.stop();

        timer.synchronize_and_start("phase_01_02_chunking_mapping");
        // assign dc-substrings to blocks
        std::vector<uint64_t> bucket_sizes(num_buckets, 0);
        std::vector<BucketMappingType> sample_to_bucket(chunked_chars.size(), num_buckets);
        KASSERT(num_buckets <= std::numeric_limits<BucketMappingType>::max());

        uint64_t materialized_samples = 0;
        for (uint64_t i = 0; i < received_chunks; i++) {
            uint64_t start_chunk = i * chars_with_padding;
            uint64_t global_index_chunk = chunk_global_index[i];
            for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                BucketMappingType block_id = num_buckets - 1;
                uint64_t suffix_start = start_chunk + j;
                uint64_t global_index = global_index_chunk + j;
                uint64_t m = global_index % X;
                if (is_in_dc<DC>(m)) {
                    auto cmp = [&](int64_t k) {
                        return cmp_index_substring(chunked_chars, suffix_start, bucket_splitter[k]);
                    };
                    block_id = util::binary_search(0, num_buckets - 1, cmp);
                    bucket_sizes[block_id]++;
                    sample_to_bucket[suffix_start] = block_id;
                    materialized_samples++;
                }
            }
        }
        timer.stop();
        uint64_t total_materialized_samples = mpi_util::all_reduce_sum(materialized_samples, comm);
        KASSERT(total_materialized_samples == info.total_sample_size);

        // log imbalance of buckets
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, info.total_sample_size, comm);
        get_stats_instance().bucket_imbalance_samples.push_back(bucket_imbalance);
        report_on_root("--> Randomized Bucket Imbalance " + std::to_string(bucket_imbalance),
                       comm,
                       info.recursion_depth);


        std::vector<SampleString> samples;
        std::vector<RankIndex> concat_rank_buckets;
        std::vector<uint64_t> received_size;
        received_size.reserve(num_buckets);
        SampleString prev_sample;

        auto materialize_sample = [&](uint64_t i) {
            return phase1.materialize_sample(chunked_chars, i, char_packing_ratio);
        };

        // sorting in each round one blocks of materialized samples
        for (uint64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_01_02_chunkingcollect_buckets");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);

            for (uint64_t i = 0; i < received_chunks; i++) {
                uint64_t start_chunk = i * chars_with_padding;
                uint64_t global_index_chunk = chunk_global_index[i];
                for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                    uint64_t char_pos = start_chunk + j;
                    uint64_t global_index = global_index_chunk + j;
                    if (sample_to_bucket[char_pos] == k) {
                        auto chars = materialize_sample(char_pos);
                        samples.push_back(SampleString(std::move(chars), global_index));
                    }
                }
            }

            timer.stop();
            KASSERT(bucket_sizes[k] == samples.size());
            process_bucket_samples(phase1,
                                   samples,
                                   concat_rank_buckets,
                                   received_size,
                                   prev_sample,
                                   k,
                                   use_packing);
        }
        KASSERT(mpi_util::all_reduce_sum(concat_rank_buckets.size(), comm)
                == info.total_sample_size);
        double bucket_imbalance_received =
            get_imbalance_bucket(received_size, info.total_sample_size, comm);
        get_stats_instance().bucket_imbalance_samples_received.push_back(bucket_imbalance_received);
        report_on_root("--> Randomized Bucket Imbalance Received "
                           + std::to_string(bucket_imbalance_received),
                       comm,
                       info.recursion_depth);


        timer.synchronize_and_start("phase_01_02_space_efficient_sort_alltoall");
        std::vector<RankIndex> local_ranks =
            mpi_util::transpose_blocks_wrapper(concat_rank_buckets,
                                               received_size,
                                               comm,
                                               config.rearrange_buckets_balanced);
        timer.stop();

        compute_ranks_from_changes(local_ranks);
        flag_unique_ranks(local_ranks);
        return local_ranks;
    }
};

} // namespace dsss::dcx

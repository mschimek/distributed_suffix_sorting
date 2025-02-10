#pragma once

#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/p2p.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/packing.hpp"
#include "pdcx/redistribute.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/space_efficient_sort.hpp"
#include "pdcx/statistics.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"


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

    Communicator<>& comm;
    PDCXLengthInfo& info;

    LexicographicRankPhase(Communicator<>& _comm, PDCXLengthInfo& _info)
        : comm(_comm),
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
        DBG("shift samples");
        shift_samples_left(local_samples);
        DBG("compute ranks");
        std::vector<RankIndex> local_ranks = compute_lexicographic_ranks(local_samples);
        DBG("flag unique ranks");
        flag_unique_ranks(local_ranks);
        return local_ranks;
    }

    std::vector<RankIndex>
    create_ranks_space_efficient(auto& phase1,
                                 std::vector<char_type>& local_string,
                                 const uint64_t num_buckets,
                                 const bool use_packing = false) {
        using SpaceEfficient = SpaceEfficientSort<char_type, index_type, DC>;
        using Splitter = typename SpaceEfficient::Splitter;
        const uint32_t X = DC::X;

        auto& timer = measurements::timer();

        // CharPacking<char_type, X + 1> packing(info.largest_char);
        PDCXConfig config = phase1.config;
        SpaceEfficient space_efficient(comm, config);

        auto materialize_sample = [&](uint64_t i) {
            return phase1.materialize_sample(local_string, i);
        };


        DBG("determine bucket splitters");

        // determine bucket splitters
        std::vector<Splitter> bucket_splitter =
            space_efficient.random_sample_splitters(info.local_chars, num_buckets, local_string);

        // assign dc-substrings to blocks
        std::vector<uint64_t> bucket_sizes(num_buckets, 0);
        std::vector<uint8_t> sample_to_bucket(local_string.size(), num_buckets);
        KASSERT(num_buckets <= 255ull);

        uint64_t offset = info.chars_before % X;
        uint64_t _local_sample_size = 0;
        // add dummy sample for last PE
        for (uint64_t i = 0; i < info.local_chars_with_dummy; i++) {
            uint64_t m = (i + offset) % X;
            if (is_in_dc<DC>(m)) {
                _local_sample_size++;
                uint8_t block_id = num_buckets - 1;
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

        SampleString prev_sample;
        // sorting in each round one blocks of materialized samples
        for (uint64_t k = 0; k < num_buckets; k++) {
            DBG("bucket round " + std::to_string(k));

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

            DBG("sort bucket samples");
            // print_concatenated_size(samples, comm, "samples size before sort");

            // Phase 1: sort dc-samples
            phase1.sort_samples(samples);

            // print_concatenated_size(samples, comm, "samples size after sort");
            DBG("redist");


            // Phase 2: compute lexicographic ranks
            redistribute_if_imbalanced(samples, config.min_imbalance, comm);
            shift_samples_left(samples);

            if (k != 0) {
                SampleString first_sample =
                    mpi_util::send_from_to(samples.front(), 0, comm.size() - 1, comm);
                if (comm.rank() == comm.size() - 1) {
                    // last PE compared samples with Padding --> change is always 1
                    // if there was no change revert to 0
                    concat_rank_buckets.back().rank -=
                        prev_sample.chars == first_sample.chars ? 1 : 0;
                }
            }
            // skip padding sample
            prev_sample = samples[samples.size() - 2];

            // exclude sample from process i + 1
            uint64_t num_ranks = samples.size() - 1;
            concat_rank_buckets.reserve(concat_rank_buckets.size() + num_ranks);

            // only store changes
            uint64_t last_changed = 0;
            for (uint64_t i = 0; i < num_ranks; i++) {
                last_changed = samples[i].chars != samples[i + 1].chars ? 1 : 0;
                concat_rank_buckets.emplace_back(index_type(last_changed), samples[i].index, false);
            }

            received_size.push_back(num_ranks);
            samples.clear();
        }
        KASSERT(mpi_util::all_reduce_sum(concat_rank_buckets.size(), comm)
                == info.total_sample_size);
        double bucket_imbalance_received =
            get_imbalance_bucket(received_size, info.total_sample_size, comm);
        get_stats_instance().bucket_imbalance_samples_received.push_back(bucket_imbalance_received);


        DBG("transpose blocks");

        timer.synchronize_and_start("phase_01_02_space_efficient_sort_alltoall");
        std::vector<RankIndex> local_ranks =
            mpi_util::transpose_blocks_wrapper(concat_rank_buckets,
                                               received_size,
                                               comm,
                                               config.rearrange_buckets_balanced);
        timer.stop();

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

        DBG("flag ranks");

        flag_unique_ranks(local_ranks);
        return local_ranks;
    }
};

} // namespace dsss::dcx
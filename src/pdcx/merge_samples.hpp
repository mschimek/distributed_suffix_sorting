#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>

#include "ips4o.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "mpi/stats.hpp"
#include "pdcx/compute_ranks.hpp"
#include "pdcx/config.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/space_efficient_sort.hpp"
#include "pdcx/statistics.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/sample_sort_strings.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"


namespace dsss::dcx {

using namespace kamping;

//******* Start Phase 4: Merge Suffixes  ********

template <typename char_type, typename index_type, typename DC>
struct DCMergeSamples {
    // for string sorter
    using CharType = char_type;
    const CharType* cbegin_chars() const { return chars.data(); }
    const CharType* cend_chars() const { return chars.data() + DC::X; }
    std::string get_string() { return to_string(); }

    DCMergeSamples() {
        index = 0;
        chars.fill(0);
        ranks.fill(0);
    }
    DCMergeSamples(std::array<char_type, DC::X>&& _chars,
                   std::array<index_type, DC::D>&& _ranks,
                   index_type _index)
        : chars(_chars),
          ranks(_ranks),
          index(_index) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << "((" << (uint64_t)chars[0];
        for (uint i = 1; i < DC::X - 1; i++) {
            ss << " " << (uint64_t)chars[i];
        }
        ss << ") (" << ranks[0];
        for (uint i = 1; i < DC::D; i++) {
            ss << " " << ranks[i];
        }
        ss << ") " << index << ")";
        return ss.str();
    }
    bool operator<(const DCMergeSamples& b) const {
        index_type i1 = index % DC::X;
        index_type i2 = b.index % DC::X;
        auto [d, r1, r2] = DC::cmpDepthRanks[i1][i2];

        // compare first d chars
        for (int k = 0; k < d; k++) {
            if (chars[k] != b.chars[k]) {
                return chars[k] < b.chars[k];
            }
        }

        // tie breaking using ranks
        return ranks[r1] < b.ranks[r2];
    }

    // X - 1 chars + 0
    std::array<char_type, DC::X> chars;
    std::array<index_type, DC::D> ranks;
    index_type index;
};

template <typename char_type, typename index_type, typename DC>
struct MergeSamplePhase {
    using SampleString = DCSampleString<char_type, index_type, DC>;
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using MergeSamples = DCMergeSamples<char_type, index_type, DC>;
    using LcpType = SeqStringSorterWrapper::LcpType;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;
    PDCXConfig& config;
    mpi::SortingWrapper& atomic_sorter;
    dsss::SeqStringSorterWrapper& string_sorter;
    SpaceEfficientSort<char_type, index_type, DC> space_efficient_sort;

    MergeSamplePhase(Communicator<>& _comm,
                     PDCXConfig& _config,
                     mpi::SortingWrapper& _atomic_sorter,
                     dsss::SeqStringSorterWrapper& _string_sorter)
        : comm(_comm),
          config(_config),
          atomic_sorter(_atomic_sorter),
          string_sorter(_string_sorter),
          space_efficient_sort(comm, config) {}

    // shift ranks left to access overlapping ranks
    void shift_ranks_left(std::vector<RankIndex>& local_ranks) const {
        mpi_util::shift_entries_left(local_ranks, D, comm);
        local_ranks.shrink_to_fit();
    }

    // add dummy padding that is sorted at the end
    void push_padding(std::vector<RankIndex>& local_ranks, index_type total_chars) const {
        if (comm.rank() == comm.size() - 1) {
            RankIndex padding(0, total_chars, false);
            std::fill_n(std::back_inserter(local_ranks), D, padding);
            local_ranks.shrink_to_fit();
        }
    }

    uint64_t get_ranks_pos(std::vector<RankIndex>& local_ranks,
                           int64_t local_index,
                           int64_t chars_before) const {
        int64_t block_nr =
            std::max(int64_t(0), (local_index / X) - 1); // block of one periodic sample
        int64_t rank_pos = block_nr * D;                 // estimate of position
        while (local_index > (int64_t)local_ranks[rank_pos].index - chars_before) {
            rank_pos++;
            KASSERT(rank_pos < (int64_t)local_ranks.size());
        }
        return rank_pos;
    }

    MergeSamples materialize_merge_sample(std::vector<char_type>& local_string,
                                          std::vector<RankIndex>& local_ranks,
                                          uint64_t chars_before,
                                          uint64_t char_pos,
                                          uint64_t rank_pos) const {
        KASSERT(char_pos + X - 2 < local_string.size());
        KASSERT(rank_pos + D - 1 < local_ranks.size());
        std::array<char_type, X> chars;
        std::array<index_type, D> ranks;
        for (uint32_t i = 0; i < X - 1; i++) {
            chars[i] = local_string[char_pos + i];
        }
        // strings must be terminated with 0
        chars.back() = 0;
        for (uint32_t i = 0; i < D; i++) {
            ranks[i] = local_ranks[rank_pos + i].rank;
        }
        uint64_t global_index = char_pos + chars_before;
        return MergeSamples(std::move(chars), std::move(ranks), global_index);
    }
    MergeSamples materialize_merge_sample_at(std::vector<char_type>& local_string,
                                             std::vector<RankIndex>& local_ranks,
                                             uint64_t chars_before,
                                             uint64_t local_index) const {
        uint64_t rank_pos = get_ranks_pos(local_ranks, local_index, chars_before);
        return materialize_merge_sample(local_string,
                                        local_ranks,
                                        chars_before,
                                        local_index,
                                        rank_pos);
    }

    // materialize all substrings of length X - 1 and corresponding D ranks
    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      uint64_t chars_before,
                                                      uint64_t chars_at_proc) const {
        uint64_t rank_pos = 0;
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(chars_at_proc);

        // for each index in local string
        for (uint64_t local_index = 0; local_index < chars_at_proc; local_index++) {
            // find next index in difference cover
            while (local_index > local_ranks[rank_pos].index - chars_before) {
                rank_pos++;
                KASSERT(rank_pos < local_ranks.size());
            }
            MergeSamples sample = materialize_merge_sample(local_string,
                                                           local_ranks,
                                                           chars_before,
                                                           local_index,
                                                           rank_pos);
            merge_samples.push_back(sample);
        }
        return merge_samples;
    }

    // sort merge samples using substrings and rank information
    void sort_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_sort_merge_samples");
        atomic_sorter.sort(merge_samples, std::less<>{});
        timer.stop();
    }

    std::vector<LcpType> string_sort_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        bool output_lcps = config.use_lcps_tie_breaking;
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_sort_merge_samples");
        std::vector<LcpType> lcps = mpi::sample_sort_strings(merge_samples,
                                                             comm,
                                                             string_sorter,
                                                             config.sample_sort_config,
                                                             output_lcps);
        timer.stop();
        return lcps;
    }

    void tie_break_ranks(std::vector<MergeSamples>& merge_samples, std::vector<LcpType>& lcps) {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_string_tie_breaking");

        // assuming that chars are not split by sample sorter
        auto cmp_rank = [](const MergeSamples& a, const MergeSamples& b) {
            index_type i1 = a.index % DC::X;
            index_type i2 = b.index % DC::X;
            auto [d, r1, r2] = DC::cmpDepthRanks[i1][i2];
            return a.ranks[r1] < b.ranks[r2];
        };

        int64_t local_max_segment = 0;
        int64_t local_sum_segment = 0;
        int64_t local_num_segment = 0;
        if (config.use_lcps_tie_breaking) {
            KASSERT(lcps.size() == merge_samples.size());
        }

        // sort each segement with the same chars by rank
        int64_t start = 0;
        int64_t end = 0;
        for (int64_t i = 0; i < (int64_t)merge_samples.size() - 1; i++) {
            bool segment_ended;
            if (config.use_lcps_tie_breaking) {
                // for some reason LCPs can be larger than X - 1?
                // segment_ended = lcps[i + 1] != DC::X - 1;
                segment_ended = lcps[i + 1] < DC::X - 1;
                KASSERT(segment_ended == (merge_samples[i].chars != merge_samples[i + 1].chars),
                        std::to_string(lcps[i + 1]) + " " + std::to_string(DC::X - 1) + " "
                            + merge_samples[i].to_string() + " "
                            + merge_samples[i + 1].to_string());

            } else {
                segment_ended = merge_samples[i].chars != merge_samples[i + 1].chars;
            }
            if (segment_ended) {
                local_num_segment++;
                end = i + 1;
                ips4o::sort(merge_samples.begin() + start, merge_samples.begin() + end, cmp_rank);
                local_sum_segment += end - start;
                local_max_segment = std::max(local_max_segment, end - start);
                start = end;
            }
        }

        end = merge_samples.size();
        local_sum_segment += end - start;
        local_max_segment = std::max(local_max_segment, end - start);
        local_num_segment += end != start;

        if (merge_samples.size() > 1) {
            ips4o::sort(merge_samples.begin() + start, merge_samples.end(), cmp_rank);
        }

        int64_t total_segments = mpi_util::all_reduce_sum(local_num_segment, comm);
        int64_t sum_segments = mpi_util::all_reduce_sum(local_sum_segment, comm);
        int64_t max_segments = mpi_util::all_reduce_max(local_max_segment, comm);
        double avg_len = total_segments == 0 ? 0 : (double)sum_segments / total_segments;
        get_stats_instance().avg_segment.push_back(avg_len);
        get_stats_instance().max_segment.push_back(max_segments);
        timer.stop();
    }

    // extract SA from merge samples
    std::vector<index_type> extract_SA(std::vector<MergeSamples>& merge_samples) const {
        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);
        return local_SA;
    }

    std::vector<index_type> space_effient_sort_SA(
        std::vector<char_type>& local_string,
        std::vector<RankIndex>& local_ranks,
        uint64_t chars_before,
        uint64_t local_chars,
        std::vector<typename SampleString::SampleStringLetters>& global_splitters) {
        auto& timer = measurements::timer();

        using SA = std::vector<index_type>;
        int64_t num_buckets = global_splitters.size() + 1;

        auto [bucket_sizes, sample_to_bucket] =
            space_efficient_sort.compute_sample_to_block_mapping(local_string,
                                                                 local_chars,
                                                                 global_splitters);
        std::vector<MergeSamples> samples;
        SA concat_sa_buckets;
        std::vector<uint64_t> sa_bucket_size;
        sa_bucket_size.reserve(num_buckets);

        /*
            // log all bucket sizes
            auto all_buckets = comm.gatherv(kamping::send_buf(bucket_sizes));
            // phase 4 stats are logged in deepest level first
            std::reverse(all_buckets.begin(), all_buckets.end());
            stats.bucket_sizes.insert(stats.bucket_sizes.end(),
                                      all_buckets.begin(),
                                      all_buckets.end());
        */
        // log imbalance of buckets
        uint64_t largest_bucket = mpi_util::all_reduce_max(bucket_sizes, comm);
        uint64_t total_chars = mpi_util::all_reduce_sum(local_chars, comm);
        double avg_buckets = (double)total_chars / (num_buckets * comm.size());
        double bucket_imbalance = ((double)largest_bucket / avg_buckets) - 1.0;
        get_stats_instance().bucket_imbalance_merging.push_back(bucket_imbalance);

        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_04_space_efficient_sort_collect_bucket");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);
            for (uint64_t idx = 0; idx < local_chars; idx++) {
                if (sample_to_bucket[idx] == k) {
                    MergeSamples sample =
                        materialize_merge_sample_at(local_string, local_ranks, chars_before, idx);
                    samples.push_back(sample);
                }
            }
            timer.stop();
            KASSERT(bucket_sizes[k] == samples.size());

            if (config.balance_blocks_space_efficient_sort) {
                timer.synchronize_and_start("phase_04_space_efficient_sort_balance_buckets");
                samples = mpi_util::distribute_data(samples, comm);
                timer.stop();
            }

            if (config.use_string_sort) {
                auto lcps = string_sort_merge_samples(samples);
                tie_break_ranks(samples, lcps);
            } else {
                sort_merge_samples(samples);
            }

            // extract SA of block
            for (auto& sample: samples) {
                concat_sa_buckets.push_back(sample.index);
            }
            sa_bucket_size.push_back(samples.size());
            samples.clear();
        }

        timer.synchronize_and_start("phase_04_space_efficient_sort_alltoall");
        SA local_SA = mpi_util::transpose_blocks(concat_sa_buckets, sa_bucket_size, comm);
        timer.stop();

        return local_SA;
    }
};


} // namespace dsss::dcx
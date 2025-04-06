#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
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
#include "pdcx/packing.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/space_efficient_sort.hpp"
#include "pdcx/statistics.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/sample_sort_strings.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/char_container.hpp"
#include "util/binary_search.hpp"
#include "util/division.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"

namespace dsss::dcx {

using namespace kamping;

//******* Start Phase 4: Merge Suffixes  ********

template <typename char_type,
          typename index_type,
          typename DC,
          typename CharContainer = CharArray<char_type, DC::X>>
struct DCMergeSamples {
    // for string sorter
    using CharType = char_type;
    const CharType* cbegin_chars() const { return chars.cbegin_chars(); }
    const CharType* cend_chars() const { return chars.cend_chars(); }
    std::string get_string() { return to_string(); }

    // to separately send chars, ranks and index
    using CharContainerType = CharContainer;
    struct NonCharData {
        std::array<index_type, DC::D> ranks;
        index_type index;
    };

    NonCharData get_non_char_data() const { return {ranks, index}; }

    DCMergeSamples() : chars(CharContainer()), ranks(), index(0) { ranks.fill(0); }
    DCMergeSamples(CharContainer&& _chars,
                   std::array<index_type, DC::D>&& _ranks,
                   index_type _index)
        : chars(_chars),
          ranks(_ranks),
          index(_index) {}

    DCMergeSamples(CharContainer&& _chars, NonCharData&& _non_chars)
        : chars(_chars),
          ranks(std::move(_non_chars.ranks)),
          index(std::move(_non_chars.index)) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << "((" << chars.to_string();
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

        if constexpr (CharContainer::IS_PACKED) {
            // compare multiple characters at once with packed integers
            if (chars != b.chars)
                return (chars < b.chars);
        } else {
            // compare first d chars
            for (uint32_t k = 0; k < d; k++) {
                if (chars.at(k) != b.chars.at(k)) {
                    return chars.at(k) < b.chars.at(k);
                }
            }
        }

        // tie breaking using ranks
        KASSERT(r1 < ranks.size());
        KASSERT(r2 < ranks.size());
        return ranks[r1] < b.ranks[r2];
    }

    static bool cmp_by_chars_and_ranks(const DCMergeSamples& a, const DCMergeSamples& b) {
        return a < b;
    }
    static bool cmp_by_chars(const DCMergeSamples& a, const DCMergeSamples& b) {
        return a.chars < b.chars;
    }


    // X - 1 chars + 0
    CharContainer chars;
    std::array<index_type, DC::D> ranks;
    index_type index;
};

template <typename char_type,
          typename index_type,
          typename DC,
          typename CharContainer = CharArray<char_type, DC::X>>
struct MergeSamplePhase {
    using SampleString = DCSampleString<char_type, index_type, DC>;
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using MergeSamples = DCMergeSamples<char_type, index_type, DC, CharContainer>;
    using LcpType = SeqStringSorterWrapper::LcpType;
    using SplitterType = MergeSamples;
    using CharContainerType = CharContainer;
    using RankContainerType = std::array<index_type, DC::D>;
    using BucketMappingType = SpaceEfficientSort<char_type, index_type, DC>::BucketMappingType;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;
    PDCXConfig const& config;
    PDCXLengthInfo& info;
    mpi::SortingWrapper& atomic_sorter;
    dsss::SeqStringSorterWrapper& string_sorter;
    SpaceEfficientSort<char_type, index_type, DC> space_efficient_sort;

    MergeSamplePhase(Communicator<>& _comm,
                     PDCXConfig const& _config,
                     PDCXLengthInfo& _info,
                     mpi::SortingWrapper& _atomic_sorter,
                     dsss::SeqStringSorterWrapper& _string_sorter)
        : comm(_comm),
          config(_config),
          info(_info),
          atomic_sorter(_atomic_sorter),
          string_sorter(_string_sorter),
          space_efficient_sort(comm, config) {}

    // shift ranks left to access overlapping ranks
    void shift_ranks_left(std::vector<index_type>& local_ranks) const {
        mpi_util::shift_entries_left(local_ranks, D, comm);
        local_ranks.shrink_to_fit();
    }

    // add dummy padding that is sorted at the end
    void push_padding(std::vector<index_type>& local_ranks) const {
        if (comm.rank() == comm.size() - 1) {
            index_type padding = 0;
            std::fill_n(std::back_inserter(local_ranks), D, padding);
            local_ranks.shrink_to_fit();
        }
    }

    auto get_splitter_comparator() const {
        return config.use_robust_tie_break ? MergeSamples::cmp_by_chars_and_ranks
                                           : MergeSamples::cmp_by_chars;
    }

    inline uint64_t get_global_ranks_pos(int64_t global_index) const {
        uint64_t block_nr = global_index / X;
        uint64_t start_block = block_nr * D;
        uint64_t rem = global_index % X;
        uint64_t offset = DC::NEXT_RANK[rem];
        uint64_t global_rank_pos = start_block + offset;
        return global_rank_pos;
    }

    inline uint64_t get_ranks_pos(int64_t local_index) const {
        uint64_t global_index = local_index + info.chars_before;
        uint64_t global_rank_pos = get_global_ranks_pos(global_index);
        uint64_t local_rank_pos = global_rank_pos - info.samples_before;
        return local_rank_pos;
    }

    CharContainer materialize_characters(std::vector<char_type>& local_string,
                                         uint64_t char_pos,
                                         double char_packing_ratio = 1) const {
        KASSERT(char_pos + X - 2 < local_string.size());
        return CharContainer(local_string.begin() + char_pos,
                             local_string.begin() + char_pos + char_packing_ratio * X - 1);
    }

    std::array<index_type, D> materialize_ranks(std::vector<index_type>& local_ranks,
                                                uint64_t rank_pos) const {
        KASSERT(rank_pos + D - 1 < local_ranks.size());
        auto start = local_ranks.begin() + rank_pos;
        std::array<index_type, D> ranks;
        std::copy(start, start + D, ranks.begin());
        return ranks;
    }

    MergeSamples materialize_merge_sample(std::vector<char_type>& local_string,
                                          std::vector<index_type>& local_ranks,
                                          uint64_t char_pos,
                                          uint64_t rank_pos,
                                          auto materialize_chars) const {
        CharContainer chars = materialize_chars(local_string, char_pos);
        std::array<index_type, D> ranks = materialize_ranks(local_ranks, rank_pos);
        uint64_t global_index = char_pos + info.chars_before;
        return MergeSamples(std::move(chars), std::move(ranks), global_index);
    }
    MergeSamples materialize_merge_sample_at(std::vector<char_type>& local_string,
                                             std::vector<index_type>& local_ranks,
                                             uint64_t local_index,
                                             auto materialize_chars) const {
        uint64_t rank_pos = get_ranks_pos(local_index);
        return materialize_merge_sample(local_string,
                                        local_ranks,
                                        local_index,
                                        rank_pos,
                                        materialize_chars);
    }

    // materialize all substrings of length X - 1 and corresponding D ranks
    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<index_type>& local_ranks,
                                                      const bool use_packing = false) const {
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(info.local_chars);

        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };

        // for each index in local string
        uint64_t rank_pos = get_ranks_pos(0);
        for (uint64_t local_index = 0; local_index < info.local_chars; local_index++) {
            MergeSamples sample = materialize_merge_sample(local_string,
                                                           local_ranks,
                                                           local_index,
                                                           rank_pos,
                                                           materialize_chars);
            merge_samples.push_back(sample);
            rank_pos += DC::IN_DC[sample.index % DC::X];
        }
        return merge_samples;
    }

    void tie_break_ranks(std::vector<MergeSamples>& merge_samples,
                         std::vector<LcpType>& lcps) const {
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

        // sort each segement with the same chars by rank
        int64_t start = 0;
        int64_t end = 0;
        for (int64_t i = 0; i < (int64_t)merge_samples.size() - 1; i++) {
            bool segment_ended = merge_samples[i].chars != merge_samples[i + 1].chars;
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
    }

    // sort merge samples using substrings and rank information
    void atomic_sort_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        atomic_sorter.sort(merge_samples, std::less<>{});
    }

    std::vector<LcpType> string_sort_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        bool output_lcps = false;
        std::vector<LcpType> lcps = mpi::sample_sort_strings(merge_samples,
                                                             comm,
                                                             string_sorter,
                                                             config.sample_sort_config,
                                                             output_lcps);
        return lcps;
    }

    void string_sort_tie_break_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        std::vector<LcpType> lcps;
        auto tie_break = [&](std::vector<MergeSamples>& merge_samples) {
            tie_break_ranks(merge_samples, lcps);
        };
        sample_sort_strings_tie_break(merge_samples,
                                      comm,
                                      string_sorter,
                                      tie_break,
                                      config.sample_sort_config);
    }

    void sort_merge_samples(std::vector<MergeSamples>& merge_samples, bool use_packing) const {
        auto& timer = measurements::timer();
        const bool use_string_sort = config.use_string_sort && !use_packing;
        const bool use_tie_break = config.use_string_sort_tie_breaking_phase4;
        if (use_string_sort && !use_tie_break) {
            timer.synchronize_and_start("phase_04_sort_merge_samples");
            auto lcps = string_sort_merge_samples(merge_samples);
            timer.stop();

            timer.synchronize_and_start("phase_04_string_tie_breaking");
            tie_break_ranks(merge_samples, lcps);
            timer.stop();

        } else if (use_string_sort && use_tie_break) {
            timer.synchronize_and_start("phase_04_sort_merge_samples");
            string_sort_tie_break_merge_samples(merge_samples);
            timer.stop();
        } else {
            timer.synchronize_and_start("phase_04_sort_merge_samples");
            atomic_sort_merge_samples(merge_samples);
            timer.stop();
        }
    }

    // extract SA from merge samples
    std::vector<index_type> extract_SA(std::vector<MergeSamples>& merge_samples) const {
        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);
        return local_SA;
    }

    std::vector<index_type> space_effient_sort_SA(std::vector<char_type>& local_string,
                                                  std::vector<index_type>& local_ranks,
                                                  std::vector<SplitterType>& global_splitters,
                                                  bool use_packing = false) {
        auto& timer = measurements::timer();

        using SA = std::vector<index_type>;
        int64_t num_buckets = global_splitters.size() + 1;

        CharPacking<char_type, X> packing(info.largest_char);

        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };
        auto get_element_at = [&](index_type i) {
            return materialize_merge_sample_at(local_string, local_ranks, i, materialize_chars);
        };
        std::vector<uint64_t> bucket_sizes;
        std::vector<BucketMappingType> sample_to_bucket;
        if (config.use_robust_tie_break) {
            auto cmp_splitter = MergeSamples::cmp_by_chars_and_ranks;
            std::tie(bucket_sizes, sample_to_bucket) =
                space_efficient_sort.compute_sample_to_block_mapping(get_element_at,
                                                                     cmp_splitter,
                                                                     info.local_chars,
                                                                     global_splitters);
        } else {
            auto get_kth_splitter_at = [&](uint64_t splitter_nr, uint64_t i) {
                return global_splitters[splitter_nr].chars.at(i);
            };
            std::tie(bucket_sizes, sample_to_bucket) =
                space_efficient_sort.compute_sample_to_block_mapping(local_string,
                                                                     info.local_chars,
                                                                     num_buckets,
                                                                     get_kth_splitter_at);
        }


        std::vector<MergeSamples> samples;
        SA concat_sa_buckets;
        std::vector<uint64_t> sa_bucket_size;
        sa_bucket_size.reserve(num_buckets);

        // log imbalance of buckets
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging.push_back(bucket_imbalance);
        report_on_root("--> Bucket Imbalance " + std::to_string(bucket_imbalance),
                       comm,
                       info.recursion_depth);
        get_local_stats_instance().input_all_bucket_sizes_merging_all.insert(
            get_local_stats_instance().input_all_bucket_sizes_merging_all.end(),
            bucket_sizes.begin(),
            bucket_sizes.end());
        get_local_stats_instance().input_max_bucket_merging_all.push_back(
            get_max_local_bucket(bucket_sizes));
        get_local_stats_instance().input_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(bucket_sizes, info.total_chars, comm.size()));


        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_04_space_efficient_sort_collect_bucket");

            // collect samples falling into kth block
            uint64_t rank_pos = get_ranks_pos(0);
            samples.reserve(bucket_sizes[k]);
            for (uint64_t idx = 0; idx < info.local_chars; idx++) {
                if (sample_to_bucket[idx] == k) {
                    MergeSamples sample = materialize_merge_sample(local_string,
                                                                   local_ranks,
                                                                   idx,
                                                                   rank_pos,
                                                                   materialize_chars);
                    samples.push_back(sample);
                }
                uint64_t global_index = info.chars_before + idx;
                rank_pos += DC::IN_DC[global_index % X];
            }

            timer.stop();
            KASSERT(bucket_sizes[k] == samples.size());

            if (config.balance_blocks_space_efficient_sort) {
                timer.synchronize_and_start("phase_04_space_efficient_sort_balance_buckets");
                samples = mpi_util::distribute_data(samples, comm);
                timer.stop();
            }

            sort_merge_samples(samples, use_packing);

            // extract SA of block
            for (auto& sample: samples) {
                concat_sa_buckets.push_back(sample.index);
            }
            sa_bucket_size.push_back(samples.size());
            samples.clear();
        }
        // log imbalance of received suffixes
        double bucket_imbalance_received =
            get_imbalance_bucket(sa_bucket_size, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging_received.push_back(bucket_imbalance_received);
        report_on_root("--> Bucket Imbalance Received " + std::to_string(bucket_imbalance_received),
                       comm,
                       info.recursion_depth);
        get_local_stats_instance().output_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(sa_bucket_size, info.total_chars, comm.size()));

        timer.synchronize_and_start("phase_04_space_efficient_sort_alltoall");
        SA local_SA = mpi_util::transpose_blocks_wrapper(concat_sa_buckets,
                                                         sa_bucket_size,
                                                         comm,
                                                         config.rearrange_buckets_balanced);
        timer.stop();

        return local_SA;
    }

    struct ChunkedData {
        std::vector<char_type> chunked_chars;
        std::vector<index_type> chunked_ranks;
        std::vector<index_type> chunk_global_index;
        std::vector<uint32_t> chunk_sizes;
        uint64_t chars_with_padding;
        uint64_t ranks_with_padding;

        std::tuple<std::vector<char_type>&,
                   std::vector<index_type>&,
                   std::vector<index_type>&,
                   std::vector<uint32_t>&>
        get_chunked_data_ref() {
            return {chunked_chars, chunked_ranks, chunk_global_index, chunk_sizes};
        }
    };

    ChunkedData compute_chunked_data(std::vector<char_type>& local_string,
                                     std::vector<index_type>& local_ranks,
                                     std::vector<SplitterType>& global_splitters,
                                     double char_packing_ratio) {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_space_effient_sort_chunking_create_chunks");
        chunking::Chunking<char_type, index_type> chunking(comm, info, config.avg_chunks_pe);
        using Chunk = chunking::Chunking<char_type, index_type>::Chunk;
        std::vector<Chunk> chunks = chunking.get_random_chunks(config.seed);
        get_stats_instance().chunk_sizes_phase4.push_back(chunking.get_chunk_size());
        get_local_stats_instance().chunk_sizes_phase4.push_back(chunking.get_chunk_size());

        // add padding to be able to materialize last suffix in chunk
        uint64_t chars_with_padding = chunking.get_chunk_size() + char_packing_ratio * X;
        uint64_t num_dc_samples = util::div_ceil(chunking.get_chunk_size(), X) * D + 1;
        uint64_t ranks_with_padding = num_dc_samples + D - 1;
        index_type padding_rank = std::numeric_limits<index_type>::max();

        auto get_ranks_positions = [&](std::vector<index_type> const& local_ranks,
                                       uint64_t chunk_start) { return get_ranks_pos(chunk_start); };

        std::vector<char_type> chunked_chars =
            chunking.get_chunked_chars(chunks, chars_with_padding, local_string);
        std::vector<index_type> chunked_ranks = chunking.get_chunked_ranks(chunks,
                                                                           ranks_with_padding,
                                                                           local_ranks,
                                                                           get_ranks_positions,
                                                                           padding_rank);
        std::vector<index_type> chunk_global_index =
            chunking.get_chunk_global_index(chunks, info.chars_before);
        std::vector<uint32_t> chunk_sizes = chunking.get_chunk_sizes(chunks, info.local_chars);
        std::vector<int64_t> send_cnt_chars = chunking.get_send_counts(chunks, chars_with_padding);
        std::vector<int64_t> send_cnt_ranks = chunking.get_send_counts(chunks, ranks_with_padding);
        std::vector<int64_t> send_cnt_index = chunking.get_send_counts(chunks, 1);

        // sanity checks
        KASSERT(std::accumulate(send_cnt_chars.begin(), send_cnt_chars.end(), 0)
                == (int64_t)chunked_chars.size());
        KASSERT(std::accumulate(send_cnt_ranks.begin(), send_cnt_ranks.end(), 0)
                == (int64_t)chunked_ranks.size());
        KASSERT(std::accumulate(send_cnt_index.begin(), send_cnt_index.end(), 0)
                == (int64_t)chunk_global_index.size());

        free_memory(std::move(local_ranks));
        timer.stop();

        // exchange linearized data
        timer.synchronize_and_start("phase_04_space_effient_sort_chunking_alltoall_chunks");
        chunked_chars = mpi_util::alltoallv_combined(chunked_chars, send_cnt_chars, comm);
        chunked_ranks = mpi_util::alltoallv_combined(chunked_ranks, send_cnt_ranks, comm);
        chunk_global_index = mpi_util::alltoallv_combined(chunk_global_index, send_cnt_index, comm);
        chunk_sizes = mpi_util::alltoallv_combined(chunk_sizes, send_cnt_index, comm);
        timer.stop();

        timer.synchronize_and_start("phase_04_space_effient_sort_chunking_mapping");
        uint64_t received_chunks = chunk_global_index.size();
        KASSERT(chunked_chars.size() == received_chunks * chars_with_padding);
        KASSERT(chunked_ranks.size() == received_chunks * ranks_with_padding);
        KASSERT(chunk_sizes.size() == received_chunks);

        return {chunked_chars,
                chunked_ranks,
                chunk_global_index,
                chunk_sizes,
                chars_with_padding,
                ranks_with_padding};
    }

    struct BucketMapping {
        std::vector<uint64_t> bucket_sizes;
        std::vector<BucketMappingType> sample_to_bucket;
    };

    BucketMapping compute_bucket_mapping(ChunkedData& chunked_data,
                                         std::vector<SplitterType>& global_splitters,
                                         auto materialize_chars) {
        auto [chunked_chars, chunked_ranks, chunk_global_index, chunk_sizes] =
            chunked_data.get_chunked_data_ref();

        uint64_t chars_with_padding = chunked_data.chars_with_padding;
        uint64_t ranks_with_padding = chunked_data.ranks_with_padding;
        uint64_t received_chunks = chunk_global_index.size();

        // compute bucket sizes and mapping
        int64_t num_buckets = global_splitters.size() + 1;
        std::vector<uint64_t> bucket_sizes(num_buckets, 0);
        std::vector<BucketMappingType> sample_to_bucket(chunked_chars.size(), num_buckets);
        KASSERT(num_buckets < std::numeric_limits<BucketMappingType>::max());

        uint64_t num_materialized_samples = 0;
        for (uint64_t i = 0; i < received_chunks; i++) {
            uint64_t start_chunk = i * chars_with_padding;
            uint64_t rank_pos = i * ranks_with_padding;
            uint64_t global_index_chunk = chunk_global_index[i];
            for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                BucketMappingType block_id = num_buckets - 1;
                uint64_t suffix_start = start_chunk + j;
                uint64_t global_index = global_index_chunk + j;
                if (config.use_robust_tie_break) {
                    MergeSamples sample = materialize_merge_sample(chunked_chars,
                                                                   chunked_ranks,
                                                                   suffix_start,
                                                                   rank_pos,
                                                                   materialize_chars);
                    sample.index = global_index;
                    auto cmp = [&](int64_t k) { return sample < global_splitters[k]; };
                    block_id = util::binary_search(0, num_buckets - 1, cmp);

                } else {
                    auto cmp = [&](int64_t k) {
                        for (uint64_t i = suffix_start; i < suffix_start + X - 1; i++) {
                            char_type c = chunked_chars[i];
                            if (c != global_splitters[k].chars.at(i - suffix_start)) {
                                return c < global_splitters[k].chars.at(i - suffix_start);
                            }
                        }
                        return false;
                    };
                    block_id = util::binary_search(0, num_buckets - 1, cmp);
                }

                bucket_sizes[block_id]++;
                sample_to_bucket[suffix_start] = block_id;
                num_materialized_samples++;
                rank_pos += DC::IN_DC[global_index % X];
            }
        }
        KASSERT(mpi_util::all_reduce_sum(num_materialized_samples, comm) == info.total_chars);
        return {bucket_sizes, sample_to_bucket};
    }

    std::vector<index_type>
    space_effient_sort_chunking_SA(std::vector<char_type>& local_string,
                                   std::vector<index_type>& local_ranks,
                                   std::vector<SplitterType>& global_splitters,
                                   bool use_packing = false) {
        using SA = std::vector<index_type>;
        auto& timer = measurements::timer();

        int64_t num_buckets = global_splitters.size() + 1;
        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };
        ChunkedData chunked_data =
            compute_chunked_data(local_string, local_ranks, global_splitters, char_packing_ratio);
        auto [chunked_chars, chunked_ranks, chunk_global_index, chunk_sizes] =
            chunked_data.get_chunked_data_ref();
        uint64_t chars_with_padding = chunked_data.chars_with_padding;
        uint64_t ranks_with_padding = chunked_data.ranks_with_padding;
        uint64_t received_chunks = chunk_global_index.size();

        BucketMapping bucket_mapping =
            compute_bucket_mapping(chunked_data, global_splitters, materialize_chars);
        std::vector<uint64_t>& bucket_sizes = bucket_mapping.bucket_sizes;
        std::vector<BucketMappingType>& sample_to_bucket = bucket_mapping.sample_to_bucket;

        // log imbalance
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging.push_back(bucket_imbalance);
        report_on_root("--> Randomized Bucket Imbalance " + std::to_string(bucket_imbalance),
                       comm,
                       info.recursion_depth);

        get_local_stats_instance().input_all_bucket_sizes_merging_all.insert(
            get_local_stats_instance().input_all_bucket_sizes_merging_all.end(),
            bucket_sizes.begin(),
            bucket_sizes.end());
        get_local_stats_instance().input_max_bucket_merging_all.push_back(
            get_max_local_bucket(bucket_sizes));
        get_local_stats_instance().input_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(bucket_sizes, info.total_chars, comm.size()));
        timer.stop();

        std::vector<MergeSamples> samples;
        SA concat_sa_buckets;

        // size estimate, in best case, need only one reallocation at the end
        uint64_t estimated_size = (info.total_chars / comm.size()) * 1.03;
        concat_sa_buckets.reserve(estimated_size);

        std::vector<uint64_t> sa_bucket_size;
        sa_bucket_size.reserve(num_buckets);

        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_04_space_effient_sort_chunking_collect_bucket");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);

            for (uint64_t i = 0; i < received_chunks; i++) {
                uint64_t start_chunk = i * chars_with_padding;
                uint64_t rank_pos = i * ranks_with_padding;
                uint64_t global_index_chunk = chunk_global_index[i];
                for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                    uint64_t char_pos = start_chunk + j;
                    uint64_t global_index = global_index_chunk + j;
                    if (sample_to_bucket[char_pos] == k) {
                        auto merge_sample = materialize_merge_sample(chunked_chars,
                                                                     chunked_ranks,
                                                                     char_pos,
                                                                     rank_pos,
                                                                     materialize_chars);
                        // above function does not compute global_index the right way
                        merge_sample.index = global_index;
                        samples.push_back(merge_sample);
                    }
                    // increment rank position index when we skipped over a DC-element
                    rank_pos += DC::IN_DC[global_index % X];
                }
            }
            timer.stop();
            KASSERT(samples.size() == bucket_sizes[k]);

            sort_merge_samples(samples, use_packing);

            timer.start("phase_04_space_effient_sort_wait_after_sort");
            comm.barrier();
            timer.stop();

            // reserve exact size, in case estimation was to low
            uint64_t new_size = concat_sa_buckets.size() + samples.size();
            if (new_size > concat_sa_buckets.capacity()) {
                concat_sa_buckets.reserve(new_size);
            }

            // extract SA of block
            for (auto& sample: samples) {
                concat_sa_buckets.push_back(sample.index);
            }

            sa_bucket_size.push_back(samples.size());
            samples.clear();
        }

        // ensure memory is freed
        free_memory(std::move(samples));
        free_memory(std::move(chunked_chars));
        free_memory(std::move(chunked_ranks));
        free_memory(std::move(chunk_global_index));
        free_memory(std::move(chunk_sizes));
        free_memory(std::move(chunked_data));
        free_memory(std::move(bucket_mapping));

        // log imbalance of received suffixes
        double bucket_imbalance_received =
            get_imbalance_bucket(sa_bucket_size, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging_received.push_back(bucket_imbalance_received);
        report_on_root("--> Randomized Bucket Imbalance Received "
                           + std::to_string(bucket_imbalance_received),
                       comm,
                       info.recursion_depth);
        get_local_stats_instance().output_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(sa_bucket_size, info.total_chars, comm.size()));

        if (info.recursion_depth == 0) {
            get_stats_instance().phase_04_sa_size =
                comm.allgather(kamping::send_buf(concat_sa_buckets.size()));
            get_stats_instance().phase_04_sa_capacity =
                comm.allgather(kamping::send_buf(concat_sa_buckets.capacity()));
        }

        timer.synchronize_and_start("phase_04_space_efficient_sort_chunking_alltoall");
        SA local_SA = mpi_util::transpose_blocks_wrapper(concat_sa_buckets,
                                                         sa_bucket_size,
                                                         comm,
                                                         config.rearrange_buckets_balanced);
        timer.stop();
        return local_SA;
    }

    // store the IDs of suffix in the same memory as the SA
    // assumes that suffixes after sorting are well balanced
    std::vector<index_type>
    space_effient_sort_chunking_SA_compressed(std::vector<char_type>& local_string,
                                              std::vector<index_type>& local_ranks,
                                              std::vector<SplitterType>& global_splitters,
                                              bool use_packing = false) {
        using SA = std::vector<index_type>;
        auto& timer = measurements::timer();

        int64_t num_buckets = global_splitters.size() + 1;
        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };
        ChunkedData chunked_data =
            compute_chunked_data(local_string, local_ranks, global_splitters, char_packing_ratio);
        auto [chunked_chars, chunked_ranks, chunk_global_index, chunk_sizes] =
            chunked_data.get_chunked_data_ref();
        uint64_t chars_with_padding = chunked_data.chars_with_padding;
        uint64_t ranks_with_padding = chunked_data.ranks_with_padding;
        uint64_t received_chunks = chunk_global_index.size();

        BucketMapping bucket_mapping =
            compute_bucket_mapping(chunked_data, global_splitters, materialize_chars);
        std::vector<uint64_t>& bucket_sizes = bucket_mapping.bucket_sizes;
        std::vector<BucketMappingType>& sample_to_bucket = bucket_mapping.sample_to_bucket;


        // store suffixes to be materialized in SA
        uint64_t num_suffixes =
            std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), uint64_t(0));
        uint64_t avg_size = (info.total_chars / comm.size());
        uint64_t min_size = std::max(avg_size, num_suffixes);
        uint64_t estimated_size = min_size * 1.25 + 10000;
        SA concat_sa_buckets(estimated_size, 0);
        uint64_t offset_sa_idx = estimated_size - num_suffixes;
        std::vector<uint64_t> bucket_idx(num_buckets, 0);
        std::exclusive_scan(bucket_sizes.begin(),
                            bucket_sizes.end(),
                            bucket_idx.begin(),
                            uint64_t(0));

        constexpr uint64_t PACK_MASK = (1ull << 20) - 1;
        KASSERT(*std::max_element(chunk_sizes.begin(), chunk_sizes.end()) <= PACK_MASK);
        KASSERT(received_chunks <= PACK_MASK);
        for (uint64_t i = 0; i < received_chunks; i++) {
            uint64_t start_chunk = i * chars_with_padding;
            for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                uint64_t char_pos = start_chunk + j;
                uint64_t b = sample_to_bucket[char_pos];
                uint64_t idx = offset_sa_idx + bucket_idx[b]++;
                uint64_t packed_chunk_and_idx = (i << 20ull) | j;
                KASSERT(idx < concat_sa_buckets.size());
                concat_sa_buckets[idx] = index_type(packed_chunk_and_idx);
            }
        }
        free_memory(std::move(bucket_mapping.sample_to_bucket));

        // log imbalance
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging.push_back(bucket_imbalance);
        report_on_root("--> Randomized Bucket Imbalance " + std::to_string(bucket_imbalance),
                       comm,
                       info.recursion_depth);
        get_local_stats_instance().input_all_bucket_sizes_merging_all.insert(
            get_local_stats_instance().input_all_bucket_sizes_merging_all.end(),
            bucket_sizes.begin(),
            bucket_sizes.end());
        get_local_stats_instance().input_max_bucket_merging_all.push_back(
            get_max_local_bucket(bucket_sizes));
        get_local_stats_instance().input_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(bucket_sizes, info.total_chars, comm.size()));
        timer.stop();

        std::vector<MergeSamples> samples;
        std::vector<uint64_t> sa_bucket_size;
        sa_bucket_size.reserve(num_buckets);
        uint64_t sa_write_idx = 0;

        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_04_space_effient_sort_chunking_collect_bucket");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);
            for (uint64_t i = 0; i < bucket_sizes[k]; i++) {
                uint64_t packed_chunk_and_idx = concat_sa_buckets[offset_sa_idx + i];
                uint64_t chunk = packed_chunk_and_idx >> 20ull;
                uint64_t idx = packed_chunk_and_idx & PACK_MASK;

                uint64_t char_pos = chunk * chars_with_padding + idx;
                uint64_t global_index_chunk = chunk_global_index[chunk];
                uint64_t global_index = global_index_chunk + idx;

                uint64_t global_rank_pos = get_global_ranks_pos(global_index);
                uint64_t rank_pos = global_rank_pos - get_global_ranks_pos(global_index_chunk);
                rank_pos += chunk * ranks_with_padding;

                auto merge_sample = materialize_merge_sample(chunked_chars,
                                                             chunked_ranks,
                                                             char_pos,
                                                             rank_pos,
                                                             materialize_chars);
                // above function does not compute global_index the right way
                merge_sample.index = global_index;
                samples.push_back(merge_sample);
            }
            offset_sa_idx += bucket_sizes[k];
            timer.stop();
            KASSERT(samples.size() == bucket_sizes[k]);

            sort_merge_samples(samples, use_packing);

            timer.start("phase_04_space_effient_sort_wait_after_sort");
            comm.barrier();
            timer.stop();

            bool ok = sa_write_idx + samples.size() < offset_sa_idx;
            bool all_ok = mpi_util::all_reduce_and(ok, comm);
            if (!all_ok) {
                report_on_root("ERROR: sa_write_idx + samples.size() >= offset_sa_idx. Use normal "
                               "chunking variant.",
                               comm,
                               info.recursion_depth);
                exit(1);
            }

            // extract SA of block
            for (auto& sample: samples) {
                KASSERT(sa_write_idx < concat_sa_buckets.size());
                concat_sa_buckets[sa_write_idx++] = sample.index;
            }

            sa_bucket_size.push_back(samples.size());
            samples.clear();
        }
        // ensure memory is freed
        free_memory(std::move(samples));
        free_memory(std::move(chunked_chars));
        free_memory(std::move(chunked_ranks));
        free_memory(std::move(chunk_global_index));
        free_memory(std::move(chunk_sizes));
        free_memory(std::move(chunked_data));
        free_memory(std::move(bucket_mapping));

        // delete last entries that do not belong to SA
        uint64_t num_idx =
            std::accumulate(sa_bucket_size.begin(), sa_bucket_size.end(), uint64_t(0));
        concat_sa_buckets.resize(num_idx);

        // log imbalance of received suffixes
        double bucket_imbalance_received =
            get_imbalance_bucket(sa_bucket_size, info.total_chars, comm);
        get_stats_instance().bucket_imbalance_merging_received.push_back(bucket_imbalance_received);
        report_on_root("--> Randomized Bucket Imbalance Received "
                           + std::to_string(bucket_imbalance_received),
                       comm,
                       info.recursion_depth);
        get_local_stats_instance().output_bucket_imbalance_merging_all.push_back(
            get_max_local_imbalance(sa_bucket_size, info.total_chars, comm.size()));

        if (info.recursion_depth == 0) {
            get_stats_instance().phase_04_sa_size =
                comm.allgather(kamping::send_buf(concat_sa_buckets.size()));
            get_stats_instance().phase_04_sa_capacity =
                comm.allgather(kamping::send_buf(concat_sa_buckets.capacity()));
        }

        timer.synchronize_and_start("phase_04_space_efficient_sort_chunking_alltoall");

        SA local_SA = mpi_util::transpose_blocks_wrapper(concat_sa_buckets,
                                                         sa_bucket_size,
                                                         comm,
                                                         config.rearrange_buckets_balanced);

        timer.stop();
        return local_SA;
    }

}; // namespace dsss::dcx


} // namespace dsss::dcx

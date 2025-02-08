#pragma once

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

    DCMergeSamples() : chars(CharContainer()), ranks(), index(0) { ranks.fill(0); }
    DCMergeSamples(CharContainer&& _chars,
                   std::array<index_type, DC::D>&& _ranks,
                   index_type _index)
        : chars(_chars),
          ranks(_ranks),
          index(_index) {}

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
        return ranks[r1] < b.ranks[r2];
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

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;
    PDCXConfig& config;
    PDCXLengthInfo& info;
    mpi::SortingWrapper& atomic_sorter;
    dsss::SeqStringSorterWrapper& string_sorter;
    SpaceEfficientSort<char_type, index_type, DC> space_efficient_sort;

    MergeSamplePhase(Communicator<>& _comm,
                     PDCXConfig& _config,
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
    void shift_ranks_left(std::vector<RankIndex>& local_ranks) const {
        mpi_util::shift_entries_left(local_ranks, D, comm);
        local_ranks.shrink_to_fit();
    }

    // add dummy padding that is sorted at the end
    void push_padding(std::vector<RankIndex>& local_ranks) const {
        if (comm.rank() == comm.size() - 1) {
            RankIndex padding(0, info.total_chars, false);
            std::fill_n(std::back_inserter(local_ranks), D, padding);
            local_ranks.shrink_to_fit();
        }
    }

    uint64_t get_ranks_pos(std::vector<RankIndex>& local_ranks, int64_t local_index) const {
        // does not need index, maybe can remove index
        uint64_t global_index = local_index + info.chars_before;
        uint64_t block_nr = global_index / X;
        uint64_t start_block = block_nr * D;
        uint64_t rem = global_index % X;
        uint64_t offset = DC::NEXT_RANK[rem];
        uint64_t global_rank_pos = start_block + offset;
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

    std::array<index_type, D> materialize_ranks(std::vector<RankIndex>& local_ranks,
                                                uint64_t rank_pos) const {
        KASSERT(rank_pos + D - 1 < local_ranks.size());
        std::array<index_type, D> ranks;
        for (uint32_t i = 0; i < D; i++) {
            ranks[i] = local_ranks[rank_pos + i].rank;
        }
        return ranks;
    }

    MergeSamples materialize_merge_sample(std::vector<char_type>& local_string,
                                          std::vector<RankIndex>& local_ranks,
                                          uint64_t char_pos,
                                          uint64_t rank_pos,
                                          auto materialize_chars) const {
        CharContainer chars = materialize_chars(local_string, char_pos);
        std::array<index_type, D> ranks = materialize_ranks(local_ranks, rank_pos);
        uint64_t global_index = char_pos + info.chars_before;
        return MergeSamples(std::move(chars), std::move(ranks), global_index);
    }
    MergeSamples materialize_merge_sample_at(std::vector<char_type>& local_string,
                                             std::vector<RankIndex>& local_ranks,
                                             uint64_t local_index,
                                             auto materialize_chars) const {
        uint64_t rank_pos = get_ranks_pos(local_ranks, local_index);
        return materialize_merge_sample(local_string,
                                        local_ranks,
                                        local_index,
                                        rank_pos,
                                        materialize_chars);
    }

    // materialize all substrings of length X - 1 and corresponding D ranks
    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      const bool use_packing = false) const {
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(info.local_chars);

        CharPacking<char_type, X> packing(info.largest_char + 1);
        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };

        // for each index in local string
        for (uint64_t local_index = 0; local_index < info.local_chars; local_index++) {
            MergeSamples sample = materialize_merge_sample_at(local_string,
                                                              local_ranks,
                                                              local_index,
                                                              materialize_chars);
            merge_samples.push_back(sample);
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
        bool output_lcps = config.use_lcps_tie_breaking;
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

    void sort_merge_samples(std::vector<MergeSamples>& merge_samples) const {
        auto& timer = measurements::timer();
        if (config.use_string_sort && !config.use_string_sort_tie_breaking) {
            timer.synchronize_and_start("phase_04_sort_merge_samples");
            auto lcps = string_sort_merge_samples(merge_samples);
            timer.stop();

            timer.synchronize_and_start("phase_04_string_tie_breaking");
            tie_break_ranks(merge_samples, lcps);
            timer.stop();

        } else if (config.use_string_sort && config.use_string_sort_tie_breaking) {
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

    std::vector<index_type>
    space_effient_sort_SA(std::vector<char_type>& local_string,
                          std::vector<RankIndex>& local_ranks,
                          std::vector<typename SampleString::SampleStringLetters>& global_splitters,
                          bool use_packing = false) {
        auto& timer = measurements::timer();

        using SA = std::vector<index_type>;
        int64_t num_buckets = global_splitters.size() + 1;

        CharPacking<char_type, X> packing(info.largest_char);

        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };

        auto [bucket_sizes, sample_to_bucket] =
            space_efficient_sort.compute_sample_to_block_mapping(local_string,
                                                                 info.local_chars,
                                                                 global_splitters);
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

        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < num_buckets; k++) {
            timer.synchronize_and_start("phase_04_space_efficient_sort_collect_bucket");

            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);
            for (uint64_t idx = 0; idx < info.local_chars; idx++) {
                if (sample_to_bucket[idx] == k) {
                    MergeSamples sample = materialize_merge_sample_at(local_string,
                                                                      local_ranks,
                                                                      idx,
                                                                      materialize_chars);
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

            sort_merge_samples(samples);

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


        timer.synchronize_and_start("phase_04_space_efficient_sort_alltoall");
        SA local_SA = mpi_util::transpose_blocks(concat_sa_buckets, sa_bucket_size, comm);
        timer.stop();

        return local_SA;
    }

    std::vector<index_type> space_effient_sort_chunking_SA(
        std::vector<char_type>& local_string,
        std::vector<RankIndex>& local_ranks,
        std::vector<typename SampleString::SampleStringLetters>& global_splitters,
        bool use_packing = false) {
        auto& timer = measurements::timer();
        using SA = std::vector<index_type>;

        // randomized chunking
        timer.synchronize_and_start("phase_04_space_effient_sort_chunking_create_chunks");

        CharPacking<char_type, X> packing(info.largest_char);

        double char_packing_ratio = use_packing ? config.packing_ratio : 1;
        auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
            return materialize_characters(local_string, char_pos, char_packing_ratio);
        };

        struct Chunk {
            index_type start_index;
            uint32_t target_pe;
        };
        std::vector<int64_t> send_cnt_chars(comm.size(), 0);
        std::vector<int64_t> send_cnt_ranks(comm.size(), 0);
        std::vector<int64_t> send_cnt_index(comm.size(), 0);

        uint64_t total_chars = mpi_util::all_reduce_sum(info.local_chars, comm);
        uint64_t chunk_size = config.num_randomized_chunks;
        uint64_t num_local_chunks = util::div_ceil(info.local_chars, chunk_size);

        // add padding to be able to materialize last suffix in chunk
        uint64_t num_dc_samples = util::div_ceil(chunk_size, X) * D + 1;
        uint64_t chars_with_padding = chunk_size + char_packing_ratio * X - 1;
        uint64_t ranks_with_padding = num_dc_samples + D - 1;

        std::mt19937 rng(config.seed + comm.rank());
        std::uniform_int_distribution<uint32_t> dist(0, comm.size() - 1);

        DBG("create chunks");

        // create chunks in evenly spaced intervals
        std::vector<Chunk> chunks;
        chunks.reserve(num_local_chunks);
        for (uint64_t i = 0; i < num_local_chunks; i++) {
            Chunk chunk = {i * chunk_size, dist(rng)};
            chunks.push_back(chunk);
            send_cnt_chars[chunk.target_pe] += chars_with_padding;
            send_cnt_ranks[chunk.target_pe] += ranks_with_padding;
            send_cnt_index[chunk.target_pe] += 1;
        }


        // sort chunks by PE
        ips4o::sort(chunks.begin(), chunks.end(), [&](const Chunk& a, const Chunk& b) {
            return a.target_pe < b.target_pe;
        });

        // store global index of beginning of each chunk
        std::vector<index_type> chunk_global_index =
            extract_attribute<Chunk, index_type>(chunks, [&](Chunk& c) {
                return index_type(info.chars_before + c.start_index);
            });

        // linearize data
        std::vector<char_type> chunked_chars;
        std::vector<RankIndex> chunked_ranks;
        std::vector<uint32_t> chunk_sizes;
        chunked_chars.reserve(chars_with_padding * num_local_chunks);
        chunked_ranks.reserve(ranks_with_padding * num_local_chunks);
        chunk_sizes.reserve(num_local_chunks);

        char_type fill_char = char_type(0);
        uint64_t padding_rank = total_chars + 1;

        DBG("fill chunks");
        for (auto& chunk: chunks) {
            // chars
            uint64_t start = chunk.start_index;
            uint64_t limit = std::min(start + chars_with_padding, local_string.size());
            for (uint64_t i = start; i < limit; i++) {
                chunked_chars.push_back(local_string[i]);
            }
            for (uint64_t i = 0; i < chars_with_padding - (limit - start); i++) {
                chunked_chars.push_back(fill_char);
            }
            uint32_t size = std::min(chunk_size, (info.local_chars - start));
            chunk_sizes.push_back(size);

            // ranks
            uint64_t first_rank = get_ranks_pos(local_ranks, start);
            limit = std::min(first_rank + ranks_with_padding, local_ranks.size());
            for (uint64_t i = first_rank; i < limit; i++) {
                chunked_ranks.push_back(local_ranks[i]);
            }
            for (uint64_t i = 0; i < ranks_with_padding - (limit - first_rank); i++) {
                chunked_ranks.push_back({padding_rank, padding_rank, false});
            }
        }

        // sanity checks
        KASSERT(chunked_chars.size() == chars_with_padding * num_local_chunks);
        KASSERT(chunked_ranks.size() == ranks_with_padding * num_local_chunks);
        KASSERT(std::accumulate(send_cnt_chars.begin(), send_cnt_chars.end(), 0)
                == (int64_t)chunked_chars.size());
        KASSERT(std::accumulate(send_cnt_ranks.begin(), send_cnt_ranks.end(), 0)
                == (int64_t)chunked_ranks.size());
        KASSERT(std::accumulate(send_cnt_index.begin(), send_cnt_index.end(), 0)
                == (int64_t)chunk_global_index.size());

        free_memory(local_ranks);
        timer.stop();

        DBG("alltoall chunks");

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

        // compute bucket sizes and mapping
        int64_t num_buckets = global_splitters.size() + 1;
        std::vector<uint64_t> bucket_sizes(num_buckets, 0);
        std::vector<uint8_t> sample_to_bucket(chunked_chars.size(), num_buckets);

        DBG("compute bucket mapping");

        uint64_t num_materialized_samples = 0;
        for (uint64_t i = 0; i < received_chunks; i++) {
            uint64_t start_chunk = i * chars_with_padding;
            for (uint64_t j = 0; j < chunk_sizes[i]; j++) {
                uint8_t block_id = num_buckets - 1;
                uint64_t suffix_start = start_chunk + j;
                for (int64_t k = 0; k < num_buckets - 1; k++) {
                    auto cmp = [&]() {
                        for (uint64_t i = suffix_start; i < suffix_start + X - 1; i++) {
                            char_type c = chunked_chars[i];
                            if (c != global_splitters[k][i - suffix_start]) {
                                return c < global_splitters[k][i - suffix_start];
                            }
                        }
                        return false;
                    };

                    if (cmp()) {
                        block_id = k;
                        break;
                    }
                }
                bucket_sizes[block_id]++;
                sample_to_bucket[suffix_start] = block_id;
                num_materialized_samples++;
            }
        }
        KASSERT(mpi_util::all_reduce_sum(num_materialized_samples, comm) == total_chars);

        // log imbalance
        double bucket_imbalance = get_imbalance_bucket(bucket_sizes, total_chars, comm);
        get_stats_instance().bucket_imbalance_merging.push_back(bucket_imbalance);
        report_on_root("--> Randomized Bucket Imbalance " + std::to_string(bucket_imbalance),
                       comm,
                       info.recursion_depth);
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
                for (uint64_t j = 0; j < chunk_size; j++) {
                    uint64_t char_pos = start_chunk + j;
                    if (sample_to_bucket[char_pos] == k) {
                        uint64_t global_index = global_index_chunk + j;

                        // increment rank position index
                        while (global_index > (uint64_t)chunked_ranks[rank_pos].index) {
                            rank_pos++;
                            KASSERT(rank_pos < chunked_ranks.size(),
                                    std::to_string(global_index) + " > "
                                        + std::to_string(chunked_ranks[rank_pos - 1].index));
                        }
                        auto merge_sample = materialize_merge_sample(chunked_chars,
                                                                     chunked_ranks,
                                                                     char_pos,
                                                                     rank_pos,
                                                                     materialize_chars);

                        // above function does not compute global_index the right way
                        merge_sample.index = global_index;
                        samples.push_back(merge_sample);
                    }
                }
            }
            timer.stop();


            DBG("sort merge samples " + std::to_string(k));
            sort_merge_samples(samples);


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

        // ensure memory of samples is freed
        free_memory(samples);

        // more vectors freed
        free_memory(chunks);
        free_memory(sample_to_bucket);
        free_memory(chunked_chars);
        free_memory(chunked_ranks);
        free_memory(chunk_global_index);
        free_memory(chunk_sizes);

        // log imbalance of received suffixes
        double bucket_imbalance_received = get_imbalance_bucket(sa_bucket_size, total_chars, comm);
        get_stats_instance().bucket_imbalance_merging_received.push_back(bucket_imbalance_received);
        report_on_root("--> Randomized Bucket Imbalance Received "
                           + std::to_string(bucket_imbalance_received),
                       comm,
                       info.recursion_depth);

        if (info.recursion_depth == 0) {
            get_stats_instance().phase_04_sa_size =
                comm.allgather(kamping::send_buf(concat_sa_buckets.size()));
            get_stats_instance().phase_04_sa_capacity =
                comm.allgather(kamping::send_buf(concat_sa_buckets.capacity()));
        }

        DBG("transpose blocks");

        timer.synchronize_and_start("phase_04_space_efficient_sort_chunking_alltoall");
        SA local_SA = mpi_util::transpose_blocks(concat_sa_buckets, sa_bucket_size, comm);
        timer.stop();


        // print_concatenated(bucket_sizes,
        //                    comm,
        //                    "bucket_sizes_" + std::to_string(info.recursion_depth));
        // print_concatenated(sa_bucket_size,
        //                    comm,
        //                    "sa_bucket_size_" + std::to_string(info.recursion_depth));

        return local_SA;
    }
}; // namespace dsss::dcx


} // namespace dsss::dcx
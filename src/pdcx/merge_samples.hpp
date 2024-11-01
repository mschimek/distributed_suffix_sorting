#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ips4o.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "mpi/stats.hpp"
#include "pdcx/compute_ranks.hpp"
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
    const CharType* cend_chars() const { return chars.data() + DC::X - 1; }
    std::string get_string() { return to_string(); }

    DCMergeSamples() {
        index = 0;
        chars.fill(0);
        ranks.fill(0);
    }
    DCMergeSamples(std::array<char_type, DC::X - 1>&& _chars,
                   std::array<index_type, DC::D>&& _ranks,
                   index_type _index)
        : chars(_chars),
          ranks(_ranks),
          index(_index) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << "((" << (uint64_t)chars[0];
        for (uint i = 1; i < DC::X - 1; i++) {
            ss << " " << (uint64_t) chars[i];
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

    std::array<char_type, DC::X - 1> chars;
    std::array<index_type, DC::D> ranks;
    index_type index;
};

template <typename char_type, typename index_type, typename DC>
struct MergeSamplePhase {
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using MergeSamples = DCMergeSamples<char_type, index_type, DC>;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;

    MergeSamplePhase(Communicator<>& _comm) : comm(_comm) {}

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
        std::array<char_type, X - 1> chars;
        std::array<index_type, D> ranks;
        for (uint32_t i = 0; i < X - 1; i++) {
            chars[i] = local_string[char_pos + i];
        }
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
    void sort_merge_samples(std::vector<MergeSamples>& merge_samples,
                            mpi::SortingWrapper& atomic_sorter) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_sort_merge_samples");
        atomic_sorter.sort(merge_samples, std::less<>{});
        timer.stop();
    }

    void string_sort_merge_samples(std::vector<MergeSamples>& merge_samples,
                                   dsss::SeqStringSorterWrapper& string_sorter,
                                   mpi::SampleSortConfig &config) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_sort_merge_samples");
        mpi::sample_sort_strings(merge_samples, comm, string_sorter, config);
        timer.stop();
    }

    void tie_break_ranks(std::vector<MergeSamples>& merge_samples) {
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
            // TODO use lcps
            if (merge_samples[i].chars != merge_samples[i + 1].chars) {
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

        report_on_root("finished segments", comm);
        int64_t total_segments = mpi_util::all_reduce_sum(local_num_segment, comm);
        int64_t sum_segments = mpi_util::all_reduce_sum(local_sum_segment, comm);
        int64_t max_segments = mpi_util::all_reduce_max(local_max_segment, comm);
        double avg_len = (double)sum_segments / total_segments;
        get_stats_instance().avg_segment.push_back(avg_len);
        get_stats_instance().max_segment.push_back(max_segments);
    }

    // extract SA from merge samples
    std::vector<index_type> extract_SA(std::vector<MergeSamples>& merge_samples) const {
        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);
        return local_SA;
    }

    std::vector<uint64_t>
    compute_bucket_sizes(std::vector<char_type>& local_string,
                         uint64_t local_chars,
                         std::vector<std::array<char_type, X>>& global_splitters) {
        int64_t blocks = global_splitters.size() + 1;
        std::vector<uint64_t> bucket_sizes(blocks, 0);
        for (uint64_t i = 0; i < local_chars; i++) {
            bool found = false;
            for (int64_t j = 0; j < blocks - 1; j++) {
                if (cmp_index_substring(local_string, i, global_splitters[j])) {
                    bucket_sizes[j]++;
                    found = true;
                    break;
                }
            }
            bucket_sizes.back() += !found;
        }
        return bucket_sizes;
    }


    // TODO: test, space efficient construction
    std::vector<index_type>
    space_effient_sort_SA(std::vector<char_type>& local_string,
                          std::vector<RankIndex>& local_ranks,
                          std::vector<uint64_t>& bucket_sizes,
                          uint64_t chars_before,
                          uint64_t local_chars,
                          std::vector<std::array<char_type, X>>& global_splitters,
                          mpi::SortingWrapper& atomic_sorter) {
        using SA = std::vector<index_type>;
        int64_t blocks = global_splitters.size() + 1;

        std::vector<MergeSamples> samples;
        SA concat_sa_blocks;
        std::vector<uint64_t> sa_block_size;
        std::vector<bool> processed(local_chars, false);
        sa_block_size.reserve(blocks);

        // TODO: maybe discard ranks that are not used anymore?
        // sorting in each round one blocks of materialized samples
        for (int64_t k = 0; k < blocks; k++) {
            // collect samples falling into kth block
            samples.reserve(bucket_sizes[k]);
            for (uint64_t idx = 0; idx < local_chars; idx++) {
                if (!processed[idx]
                    && ((k == blocks - 1)
                        || cmp_index_substring(local_string, idx, global_splitters[k]))) {
                    MergeSamples sample =
                        materialize_merge_sample_at(local_string, local_ranks, chars_before, idx);
                    processed[idx] = true;
                    samples.push_back(sample);
                }
            }
            KASSERT(bucket_sizes[k] == samples.size());
            sort_merge_samples(samples, atomic_sorter);

            // extract SA of block
            for (auto& sample: samples) {
                concat_sa_blocks.push_back(sample.index);
            }
            sa_block_size.push_back(samples.size());
            samples.clear();
        }

        std::vector<uint64_t> pref_sum_kth_block =
            comm.exscan(send_buf(sa_block_size), op(ops::plus<>{}));
        std::vector<uint64_t> sum_kth_block =
            comm.reduce(send_buf(sa_block_size), op(ops::plus<>{}));

        if (comm.rank() == 0) {
            comm.bcast(send_recv_buf(sum_kth_block));
        } else {
            sum_kth_block = comm.bcast(send_recv_buf((alloc_new<std::vector<uint64_t>>)));
        }
        KASSERT(blocks <= (int64_t)comm.size());

        // sort block indices by decreasing size
        std::vector<int64_t> idx_blocks(blocks);
        std::iota(idx_blocks.begin(), idx_blocks.end(), 0);
        std::sort(idx_blocks.begin(), idx_blocks.end(), [&](int64_t a, int64_t b) {
            return sum_kth_block[a] > sum_kth_block[b];
        });

        // divide one block amoung #PEs / #blocks
        // remainder is distributed amoung largest blocks
        std::vector<int64_t> num_pe_per_block(blocks, comm.size() / blocks);
        int64_t rem = comm.size() % blocks;
        for (int64_t k = 0; k < rem; k++) {
            int64_t k2 = idx_blocks[k];
            num_pe_per_block[k2]++;
        }

        std::vector<int64_t> pe_range(blocks + 1, 0);
        std::inclusive_scan(num_pe_per_block.begin(), num_pe_per_block.end(), pe_range.begin() + 1);

        std::vector<int64_t> target_size(comm.size(), 0);
        for (int64_t k = 0; k < blocks; k++) {
            for (int64_t rank = pe_range[k]; rank < pe_range[k + 1]; rank++) {
                target_size[rank] = sum_kth_block[k] / num_pe_per_block[k];
            }
            target_size[pe_range[k + 1] - 1] += sum_kth_block[k] % num_pe_per_block[k];
        }

        std::vector<int64_t> pred_target_size(comm.size(), 0);
        for (int64_t k = 0; k < blocks; k++) {
            std::exclusive_scan(target_size.begin() + pe_range[k],
                                target_size.begin() + pe_range[k + 1],
                                pred_target_size.begin() + pe_range[k],
                                0);
        }

        std::vector<int64_t> send_cnts(comm.size(), 0);
        for (int64_t k = 0; k < blocks; k++) {
            int64_t local_data_size = sa_block_size[k];
            int64_t preceding_size = pref_sum_kth_block[k];
            int64_t last_pe = pe_range[k + 1] - 1;
            for (int rank = pe_range[k]; rank < last_pe && local_data_size > 0; rank++) {
                int64_t to_send = std::max(int64_t(0), pred_target_size[rank + 1] - preceding_size);
                to_send = std::min(to_send, local_data_size);
                send_cnts[rank] = to_send;
                local_data_size -= to_send;
                preceding_size += to_send;
            }
            send_cnts[last_pe] += local_data_size;
        }

        int64_t total_send = std::accumulate(send_cnts.begin(), send_cnts.end(), int64_t(0));
        int64_t total_sa = std::accumulate(sa_block_size.begin(), sa_block_size.end(), int64_t(0));
        KASSERT(total_send == total_sa);

        SA local_SA = mpi_util::alltoallv_combined(concat_sa_blocks, send_cnts, comm);
        return local_SA;
    }
};


} // namespace dsss::dcx
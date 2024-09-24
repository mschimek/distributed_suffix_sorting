#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "kassert/kassert.hpp"
#include "mpi_util.hpp"
#include "printing.hpp"
#include "sort.hpp"
#include "uint_types.hpp"
#include "util.hpp"

namespace dsss::dcx {

using namespace kamping;

struct Statistics {
    Statistics() : max_depth(0) {}
    void reset() {
        max_depth = 0;
        string_sizes.clear();
        local_string_sizes.clear();
        highest_ranks.clear();
        char_type_used.clear();
        discarding_reduction.clear();
    }

    int max_depth;
    std::vector<uint64_t> local_string_sizes;
    std::vector<uint64_t> string_sizes;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
    std::vector<double> discarding_reduction;
};

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

template <typename char_type, typename index_type, typename DC>
class PDCX {
    struct SampleString {
        SampleString() {
            letters.fill(0);
            index = 0;
        }
        SampleString(std::array<char_type, DC::X> _letters, index_type _index)
            : letters(_letters),
              index(_index) {}

        bool operator<(const SampleString& other) const {
            for (uint i = 0; i < DC::X; i++) {
                if (letters[i] != other.letters[i]) {
                    return letters[i] < other.letters[i];
                }
            }
            return index < other.index;
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "(" << letters[0];
            for (uint i = 1; i < DC::X; i++) {
                ss << " " << letters[i];
            }
            ss << ") " << index;
            return ss.str();
        }

        std::array<char_type, DC::X> letters;
        index_type index;
    };

    struct RankIndex {
        index_type rank;
        index_type index;
        bool unique;

        static bool cmp_by_index(const RankIndex& a, const RankIndex& b) {
            return a.index < b.index;
        }
        static bool cmp_by_rank(const RankIndex& a, const RankIndex& b) { return a.rank < b.rank; }
        static bool cmp_mod_div(const RankIndex& a, const RankIndex& b) {
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

    struct MergeSamples {
        MergeSamples() {
            index = 0;
            chars.fill(0);
            ranks.fill(0);
        }
        MergeSamples(std::array<char_type, DC::X - 1> _chars,
                     std::array<index_type, DC::D> _ranks,
                     index_type _index)
            : chars(_chars),
              ranks(_ranks),
              index(_index) {}

        std::string to_string() const {
            std::stringstream ss;
            ss << "((" << chars[0];
            for (uint i = 1; i < DC::X - 1; i++) {
                ss << " " << chars[i];
            }
            ss << ") (" << ranks[0];
            for (uint i = 1; i < DC::D; i++) {
                ss << " " << ranks[i];
            }
            ss << ") " << index << ")";
            return ss.str();
        }
        bool operator<(const MergeSamples& b) const {
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

public:
    PDCX(Communicator<>& _comm)
        : comm(_comm),
          timer(measurements::timer()),
          stats(get_stats_instance()),
          recursion_depth(0) {}

    // maps the index i from a recursive dcx call back to the global index
    index_type map_back(index_type idx,
                        index_type total_chars,
                        std::array<index_type, DC::D + 1>& samples_before) {
        // find interval into which index belongs
        for (uint i = 0; i < DC::D; i++) {
            if (idx < samples_before[i + 1]) {
                index_type d = DC::DC[i];
                index_type k = idx - samples_before[i];
                return DC::X * k + d;
            }
        }
        KASSERT(false);
        return 0;
    }

    template <typename T>
    void free_memory(std::vector<T>& vec) {
        vec.clear();
        vec.shrink_to_fit();
    }

    void add_padding(std::vector<char_type>& local_data) {
        char_type padding = char_type(0);
        if (comm.rank() == comm.size() - 1) {
            std::fill_n(std::back_inserter(local_data), DC::X, padding);
        }
    }
    void remove_padding(std::vector<char_type>& local_data) {
        if (comm.rank() == comm.size() - 1) {
            local_data.resize(local_data.size() - DC::X);
        }
    }

    // revert changes made to local string by left shift
    void clean_up(std::vector<char_type>& local_string) {
        if (comm.rank() < comm.size() - 1) {
            local_string.resize(local_string.size() - DC::X + 1);
        }
        remove_padding(local_string);
    }

    bool is_in_dc(uint64_t idx) const {
        // simple scan sufficient for small difference cover
        for (auto d: DC::DC) {
            if (idx == d) {
                return true;
            }
        }
        return false;
    }

    // computes how many chars are at position with a remainder
    std::array<uint64_t, DC::X> compute_num_pos_mod(uint64_t total_chars) const {
        constexpr uint64_t X = DC::X;
        std::array<uint64_t, X> num_pos_mod;
        num_pos_mod.fill(0);
        for (uint64_t i = 0; i < X; i++) {
            num_pos_mod[i] = (total_chars + X - 1 - i) / X;
        }
        return num_pos_mod;
    }

    // compute samples strings of length X according to difference cover DC::DC
    std::vector<SampleString> compute_sample_strings(std::vector<char_type>& local_string,
                                                     index_type chars_before,
                                                     index_type total_chars) {
        constexpr uint32_t X = DC::X;
        constexpr uint32_t D = DC::D;

        // compute local sample strings
        std::vector<SampleString> local_samples;
        uint64_t size_estimate = ((local_string.size() + X - 1) / X) * D;
        local_samples.reserve(size_estimate);

        uint64_t offset = chars_before % X;
        for (uint64_t i = 0; i + X - 1 < local_string.size(); i++) {
            uint64_t m = (i + offset) % X;
            if (is_in_dc(m)) {
                index_type index = chars_before + i;
                std::array<char_type, X> letters;
                for (uint k = 0; k < X; k++) {
                    letters[k] = local_string[i + k];
                }
                local_samples.push_back(SampleString(letters, index));
            }
        }
        // last process adds a dummy sample if remainder of some differrence cover element aligns
        // with the string length

        return local_samples;
    }

    void flag_unique_ranks(std::vector<RankIndex>& local_ranks) {
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

    // create lexicographic ranks using a prefix sum
    std::vector<RankIndex> compute_lexicographic_ranks(std::vector<SampleString>& local_samples) {
        std::vector<RankIndex> local_ranks;

        // exclude sample from process i + 1
        uint64_t num_ranks = local_samples.size() - 1;
        local_ranks.reserve(num_ranks);

        // compute local ranks
        uint64_t prev_rank = 0;
        for (uint64_t i = 0; i < num_ranks; i++) {
            KASSERT(i + 1 < local_samples.size());
            local_ranks.emplace_back(prev_rank, local_samples[i].index, false);
            uint64_t changed = local_samples[i].letters != local_samples[i + 1].letters ? 1 : 0;
            prev_rank += changed;
        }

        // shift ranks by 1 + prefix sum
        uint64_t ranks_before = mpi_util::ex_prefix_sum(prev_rank, comm);
        std::for_each(local_ranks.begin(), local_ranks.end(), [&](RankIndex& x) {
            x.rank += 1 + ranks_before;
        });
        return local_ranks;
    }


    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      uint64_t chars_before,
                                                      uint64_t chars_at_proc) {
        constexpr uint32_t X = DC::X;
        constexpr uint32_t D = DC::D;
        uint64_t pos = 0;
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(chars_at_proc);

        // for each index in local string
        for (uint64_t local_index = 0; local_index < chars_at_proc; local_index++) {
            // find next index in difference cover
            while (local_index > local_ranks[pos].index - chars_before) {
                pos++;
                KASSERT(pos < local_ranks.size());
            }
            uint64_t global_index = local_index + chars_before;

            KASSERT(local_index + X - 2 < local_string.size());
            KASSERT(pos + D - 1 < local_ranks.size());

            std::array<char_type, X - 1> chars;
            std::array<index_type, D> ranks;
            for (uint32_t i = 0; i < X - 1; i++) {
                chars[i] = local_string[local_index + i];
            }
            for (uint32_t i = 0; i < D; i++) {
                ranks[i] = local_ranks[pos + i].rank;
            }
            merge_samples.emplace_back(chars, ranks, global_index);
        }
        return merge_samples;
    }

    void dispatch_recursive_call(std::vector<RankIndex>& local_ranks,
                                 uint64_t local_sample_size,
                                 uint64_t last_rank,
                                 uint64_t total_chars,
                                 std::array<index_type, DC::D + 1>& samples_before) {
        auto map_back_func = [&](index_type sa_i) {
            return map_back(sa_i, total_chars, samples_before);
        };
        if (total_chars <= 80) {
            // continue with sequential algorithm
            sequential_sa_and_local_ranks(local_ranks, local_sample_size, map_back_func);
        } else {
            // pick smallest data type that will fit
            if (last_rank <= std::numeric_limits<uint8_t>::max()) {
                handle_recursive_call<uint8_t>(local_ranks,
                                               local_sample_size,
                                               total_chars,
                                               map_back_func);
            } else if (last_rank <= std::numeric_limits<uint16_t>::max()) {
                handle_recursive_call<uint16_t>(local_ranks,
                                                local_sample_size,
                                                total_chars,
                                                map_back_func);
            } else if (last_rank <= std::numeric_limits<uint32_t>::max()) {
                handle_recursive_call<uint32_t>(local_ranks,
                                                local_sample_size,
                                                total_chars,
                                                map_back_func);
            } else if (last_rank <= std::numeric_limits<uint32_t>::max()) {
                handle_recursive_call<dsss::uint40>(local_ranks,
                                                    local_sample_size,
                                                    total_chars,
                                                    map_back_func);
            } else {
                print_on_root("Max Rank input size that can be handled is 2^40", comm);
            }
            // handle_recursive_call<uint32_t>(local_ranks,
            //                                 local_sample_size,
            //                                 total_chars,
            //                                 map_back_func);
        }
    }

    // sequential SACA and sequential computation ranks computation on root process
    void sequential_sa_and_local_ranks(std::vector<RankIndex>& local_ranks,
                                       uint64_t local_sample_size,
                                       auto map_back_func) {
        std::vector<RankIndex> global_ranks = comm.gatherv(send_buf(local_ranks));
        std::vector<index_type> SA;
        if (comm.rank() == 0) {
            std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_mod_div);
            auto get_rank = [](RankIndex& r) -> index_type { return r.rank; };
            std::vector<index_type> ranks =
                extract_attribute<RankIndex, index_type>(global_ranks, get_rank);

            // TODO: better sequential SACA
            SA = slow_suffixarray<index_type, index_type>(ranks);
            global_ranks.clear();

            for (uint64_t i = 0; i < SA.size(); i++) {
                index_type global_index = map_back_func(SA[i]);
                global_ranks.emplace_back(i + 1, global_index);
            }
            std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_by_index);
        }

        local_ranks.clear();
        local_ranks = mpi_util::distribute_data_custom(global_ranks, local_sample_size, comm);
    }

    std::vector<index_type> pdc3(std::vector<char_type>& local_string);

    template <typename new_char_type>
    void handle_recursive_call(std::vector<RankIndex>& local_ranks,
                               uint64_t local_sample_size,
                               uint64_t total_chars,
                               auto map_back_func) {
        // sort by (mod X, div X)
        timer.synchronize_and_start("phase_03_sort_mod_div");
        mpi::sort(local_ranks, RankIndex::cmp_mod_div, comm);
        timer.stop();
        KASSERT(local_ranks.size() >= 2u); // can happen for small inputs

        uint64_t after_discarding = num_ranks_after_discarding(local_ranks);
        double reduction = ((double)after_discarding / total_chars);
        stats.discarding_reduction.push_back(reduction);

        // TODO make this a config parameter
        double discarding_threshold = 0.0;
        bool use_discarding = reduction <= discarding_threshold;
        if (use_discarding) {
            recursive_call_with_discarding<new_char_type>(local_ranks,
                                                          local_sample_size,
                                                          after_discarding);
        } else {
            recursive_call_direct<new_char_type>(local_ranks, local_sample_size, map_back_func);
        }
    }

    template <typename new_char_type>
    void recursive_call_direct(std::vector<RankIndex>& local_ranks,
                               uint64_t local_sample_size,
                               auto map_back_func) {
        auto get_rank = [](RankIndex& r) -> new_char_type { return r.rank; };
        std::vector<new_char_type> recursive_string =
            extract_attribute<RankIndex, new_char_type>(local_ranks, get_rank);

        recursive_string =
            mpi_util::distribute_data_custom(recursive_string, local_sample_size, comm);

        // free memory of ranks
        free_memory(local_ranks);

        // TODO: flexible selection of DC
        // create new instance of PDC3 with templates of new char type size
        PDCX<new_char_type, index_type, DC> rec_pdcx(comm);

        // memory of SA is counted in recursive call
        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;
        free_memory(recursive_string);

        auto index_function = [&](index_type index, index_type sa_at_i) {
            index_type global_index = map_back_func(sa_at_i);
            index_type rank = 1 + index;
            bool unique = true; // does not matter here
            return RankIndex(rank, global_index, unique);
        };
        local_ranks = mpi_util::zip_with_index<index_type, RankIndex>(SA, index_function, comm);
        free_memory(SA);

        timer.synchronize_and_start("phase_03_sort_ranks_index");
        mpi::sort(local_ranks, RankIndex::cmp_by_index, comm);
        timer.stop();
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
    }

    uint64_t num_ranks_after_discarding(std::vector<RankIndex>& local_ranks) {
        // all ranks can be dropped that are unique and are not needed to determine a not unique
        // rank

        // for simplicity we always keep the first element
        // and don't do shifts for the first and last elements
        uint64_t count_discarded = 0;
        for (uint64_t i = 1; i < local_ranks.size(); i++) {
            bool is_unique = local_ranks[i].unique;
            bool prev_is_unique = local_ranks[i - 1].unique;
            count_discarded += is_unique && prev_is_unique;
        }
        return local_ranks.size() - count_discarded;
    }

    template <typename new_char_type>
    void recursive_call_with_discarding(std::vector<RankIndex>& local_ranks,
                                        uint64_t local_sample_size,
                                        uint64_t after_discarding) {
        KASSERT(local_ranks.size() >= 2u);

        // build recursive string and discard ranks
        std::vector<new_char_type> recursive_string;
        std::vector<index_type> orginal_index;
        std::vector<bool> pos_unique;
        recursive_string.reserve(after_discarding);
        orginal_index.reserve(after_discarding);
        pos_unique.reserve(after_discarding);

        // always keep first element
        recursive_string.push_back(local_ranks[0].rank);
        orginal_index.push_back(local_ranks[0].index);
        pos_unique.push_back(local_ranks[0].unique);
        for (uint64_t i = 1; i < local_ranks.size(); i++) {
            bool is_unique = local_ranks[i].unique;
            bool prev_is_unique = local_ranks[i - 1].unique;
            bool can_drop = is_unique && prev_is_unique;
            if (!can_drop) {
                recursive_string.push_back(local_ranks[i].rank);
                orginal_index.push_back(local_ranks[i].index);
                pos_unique.push_back(is_unique);
            }
        }

        timer.synchronize_and_start("phase_03_sort_ranks_by_ranks");
        mpi::sort(local_ranks, RankIndex::cmp_by_rank, comm);
        timer.stop();
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);

        // recursive call
        PDCX<new_char_type, index_type, DC> rec_pdcx(comm);
        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> reduced_SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;
        free_memory(recursive_string);

        // invert reduced SA to get ranks
        struct IndexRank {
            index_type index, rank;

            std::string to_string() const {
                return "(" + std::to_string(index) + ", " + std::to_string(rank) + ")";
            }
        };
        auto index_function = [&](uint64_t idx, index_type sa_index) {
            return IndexRank{sa_index, index_type(1 + idx)};
        };
        auto cmp_index_sa = [](const IndexRank& l, const IndexRank& r) {
            return l.index < r.index;
        };
        std::vector<IndexRank> ranks_sa =
            mpi_util::zip_with_index<index_type, IndexRank>(reduced_SA, index_function, comm);

        free_memory(reduced_SA);

        timer.synchronize_and_start("phase_03_sort_index_sa");
        mpi::sort(ranks_sa, cmp_index_sa, comm);
        timer.stop();

        // get ranks of recursive string that was generated locally on this PE
        ranks_sa = mpi_util::distribute_data_custom(ranks_sa, after_discarding, comm);

        uint64_t local_not_unique_red = std::count(pos_unique.begin(), pos_unique.end(), false);
        std::vector<IndexRank> new_ranks;
        new_ranks.reserve(local_not_unique_red);
        for (uint64_t i = 0; i < after_discarding; i++) {
            if (!pos_unique[i]) {
                KASSERT(i < orginal_index.size());
                KASSERT(i < ranks_sa.size());
                uint64_t org_idx = orginal_index[i];
                uint64_t new_rank = ranks_sa[i].rank;
                new_ranks.emplace_back(org_idx, new_rank);
            }
        }
        free_memory(orginal_index);
        free_memory(pos_unique);
        free_memory(ranks_sa);

        auto cmp_new_ranks = [&](auto a, auto b) { return a.rank < b.rank; };

        timer.synchronize_and_start("phase_03_sort_new_ranks");
        mpi::sort(new_ranks, cmp_new_ranks, comm);
        timer.stop();

        auto cnt_not_unique = [](uint64_t sum, RankIndex& r) { return sum + !r.unique; };
        uint64_t local_new_ranks =
            std::accumulate(local_ranks.begin(), local_ranks.end(), 0, cnt_not_unique);
        new_ranks = mpi_util::distribute_data_custom(new_ranks, local_new_ranks, comm);

        // update order of indices
        uint64_t local_rank_size = local_ranks.size();
        uint64_t ranks_before = mpi_util::ex_prefix_sum(local_rank_size, comm);
        uint64_t index_new_ranks = 0;
        for (uint64_t i = 0; i < local_ranks.size(); i++) {
            bool unique = local_ranks[i].unique;
            if (!unique) {
                KASSERT(index_new_ranks < new_ranks.size());
                local_ranks[i].index = new_ranks[index_new_ranks++].index;
            }
            local_ranks[i].rank = ranks_before + i + 1;
        }

        timer.synchronize_and_start("phase_03_sort_ranks_index");
        mpi::sort(local_ranks, RankIndex::cmp_by_index, comm);
        timer.stop();
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
    }

    std::vector<index_type> compute_sa(std::vector<char_type>& local_string) {
        timer.synchronize_and_start("pdcx");

        const int process_rank = comm.rank();
        constexpr uint32_t X = DC::X;
        constexpr uint32_t D = DC::D;

        //******* Start Phase 0: Preparation  ********
        timer.synchronize_and_start("phase_00_preparation");

        // figure out lengths of the other strings
        auto chars_at_proc = comm.allgather(send_buf(local_string.size()));
        uint64_t total_chars = std::accumulate(chars_at_proc.begin(), chars_at_proc.end(), 0);

        // number of chars before processor i
        std::vector<uint64_t> chars_before(comm.size());
        std::exclusive_scan(chars_at_proc.begin(), chars_at_proc.end(), chars_before.begin(), 0);

        // number of positions with mod X = d
        std::array<uint64_t, X> num_at_mod = compute_num_pos_mod(total_chars);
        const uint64_t rem = total_chars % X;
        bool added_dummy = is_in_dc(rem);
        num_at_mod[rem] += added_dummy;

        // inclusive prefix sum to compute map back
        std::array<index_type, DC::D + 1> samples_before;
        samples_before[0] = 0;
        for (uint i = 1; i < D + 1; i++) {
            uint d = DC::DC[i - 1];
            samples_before[i] = samples_before[i - 1] + num_at_mod[d];
        }
        index_type num_samples = samples_before.back();
        add_padding(local_string);

        // logging
        stats.max_depth = std::max(stats.max_depth, recursion_depth);
        stats.string_sizes.push_back(total_chars);
        stats.local_string_sizes.push_back(local_string.size());
        stats.char_type_used.push_back(8 * sizeof(char_type));
        timer.stop();

        // solve sequentially on root to avoid corner cases with empty PEs
        if (total_chars <= comm.size() * 10) {
            remove_padding(local_string);
            std::vector<char_type> global_string = comm.gatherv(send_buf(local_string));
            timer.stop(); // pdcx
            if (comm.rank() == 0) {
                return slow_suffixarray<char_type, index_type>(global_string);
            } else {
                return std::vector<index_type>();
            }
        }
        //******* End Phase 0: Preparation  ********

        //******* Start Phase 1: Construct Samples  ********
        timer.synchronize_and_start("phase_01_samples");

        mpi_util::shift_entries_left(local_string, X - 1, comm);
        local_string.shrink_to_fit();

        std::vector<SampleString> local_samples =
            compute_sample_strings(local_string, chars_before[process_rank], total_chars);

        index_type local_sample_size = local_samples.size();
        index_type total_sample_size = mpi_util::all_reduce_sum(local_sample_size, comm);
        KASSERT(total_sample_size == num_samples);

        timer.synchronize_and_start("phase_01_sort_local_samples");
        mpi::sort(local_samples, std::less<>{}, comm);
        timer.stop();
        local_samples.shrink_to_fit();
        timer.stop();
        //******* End Phase 1: Construct Samples  ********

        //******* Start Phase 2: Construct Ranks  ********
        timer.synchronize_and_start("phase_02_ranks");

        // adds a dummy sample for last process
        KASSERT(local_string.size() >= 1u);
        SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
        local_samples.push_back(recv_sample);
        local_samples.shrink_to_fit();

        std::vector<RankIndex> local_ranks = compute_lexicographic_ranks(local_samples);
        free_memory(local_samples);
        flag_unique_ranks(local_ranks);

        index_type last_rank = local_ranks.empty() ? index_type(0) : local_ranks.back().rank;
        comm.bcast_single(send_recv_buf(last_rank), root(comm.size() - 1));
        stats.highest_ranks.push_back(last_rank);
        bool chars_distinct = last_rank >= total_sample_size;

        timer.stop();
        //******* End Phase 2: Construct Ranks  ********

        //******* Start Phase 3: Recursive Call  ********
        timer.synchronize_and_start("phase_03_recursion");

        if (chars_distinct) {
            timer.synchronize_and_start("phase_03_sort_index_base");
            mpi::sort(local_ranks, RankIndex::cmp_by_index, comm);
            timer.stop();

            local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
            local_ranks.shrink_to_fit();

        } else {
            dispatch_recursive_call(local_ranks,
                                    local_sample_size,
                                    last_rank,
                                    total_chars,
                                    samples_before);
        }
        timer.stop();
        //******* End Phase 3: Recursive Call  ********

        //******* Start Phase 4: Merge Suffixes  ********
        timer.synchronize_and_start("phase_04_merge");

        mpi_util::shift_entries_left(local_ranks, D, comm);
        local_ranks.shrink_to_fit();

        if (comm.rank() == comm.size() - 1) {
            RankIndex padding(0, total_chars, false);
            std::fill_n(std::back_inserter(local_ranks), D, padding);
            local_ranks.shrink_to_fit();
        }

        std::vector<MergeSamples> merge_samples =
            construct_merge_samples(local_string,
                                    local_ranks,
                                    chars_before[process_rank],
                                    chars_at_proc[process_rank]);

        free_memory(local_ranks);

        timer.synchronize_and_start("phase_04_sort_merge_samples");
        mpi::sort(merge_samples, std::less<>{}, comm);
        timer.stop();

        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);

        timer.stop();
        //******* End Phase 4: Merge Suffixes  ********

        clean_up(local_string);

        timer.stop(); // pdcx

        return local_SA;
    }

    void report_time() {
        comm.barrier();
        // timer.aggregate_and_print(measurements::SimpleJsonPrinter<>{});
        timer.aggregate_and_print(measurements::FlatPrinter{});
        std::cout << "\n";
        comm.barrier();
    }

    void report_stats() {
        comm.barrier();
        if (comm.rank() == comm.size() - 1) {
            std::cout << "\nStatistics:\n";
            std::cout << "algo=DC" << DC::X << std::endl;
            std::cout << "num_proc=" << comm.size() << std::endl;
            std::cout << "max_depth=" << stats.max_depth << std::endl;
            std::cout << "string_sizes=";
            print_vector(stats.string_sizes, ",");
            std::cout << "highest_ranks=";
            print_vector(stats.highest_ranks, ",");
            std::cout << "char_type_bits=";
            print_vector(stats.char_type_used, ",");
            std::cout << "char_type_bits=";
            print_vector(stats.discarding_reduction, ",");
            std::cout << "\n";
        }
        comm.barrier();
    }

    void reset() {
        stats.reset();
        recursion_depth = 0;
        timer.clear();
    }

    constexpr static bool DBG = false;
    constexpr static bool use_recursion = true;

    Communicator<>& comm;
    measurements::Timer<Communicator<>>& timer;
    Statistics& stats;
    int recursion_depth;
};

} // namespace dsss::dcx
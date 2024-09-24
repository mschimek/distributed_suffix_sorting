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
    }

    int max_depth;
    std::vector<uint64_t> local_string_sizes;
    std::vector<uint64_t> string_sizes;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
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

        static bool cmp_by_index(const RankIndex& a, const RankIndex& b) {
            return a.index < b.index;
        }
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
            ss << "(" << rank << " " << index << ")";
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

    void add_padding(std::vector<char_type>& local_data) {
        constexpr char_type padding = 0;
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
            local_ranks.emplace_back(prev_rank, local_samples[i].index);
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
        if (total_chars <= 80) {
            // continue with sequential algorithm
            sequential_sa_and_local_ranks(local_ranks,
                                          local_sample_size,
                                          total_chars,
                                          samples_before);
        } else {
            if (last_rank <= std::numeric_limits<uint8_t>::max()) {
                handle_recursive_call<uint8_t>(local_ranks,
                                               local_sample_size,
                                               total_chars,
                                               samples_before);
            } else if (last_rank <= std::numeric_limits<uint16_t>::max()) {
                handle_recursive_call<uint16_t>(local_ranks,
                                                local_sample_size,
                                                total_chars,
                                                samples_before);
            } else if (last_rank <= std::numeric_limits<uint32_t>::max()) {
                handle_recursive_call<uint32_t>(local_ranks,
                                                local_sample_size,
                                                total_chars,
                                                samples_before);
            } else {
                handle_recursive_call<uint64_t>(local_ranks,
                                                local_sample_size,
                                                total_chars,
                                                samples_before);
            }
            // handle_recursive_call<uint32_t>(local_ranks,
            //                                 local_sample_size,
            //                                 total_chars,
            //                                 samples_before);
        }
    }

    // sequential SACA and sequential computation ranks computation on root process
    void sequential_sa_and_local_ranks(std::vector<RankIndex>& local_ranks,
                                       uint64_t local_sample_size,
                                       uint64_t total_chars,
                                       std::array<index_type, DC::D + 1>& samples_before) {
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
                index_type global_index = map_back(SA[i], total_chars, samples_before);
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
                               std::array<index_type, DC::D + 1>& samples_before) {
        // sort by (mod X, div X)
        timer.synchronize_and_start("phase_03_sort_mod_div");
        mpi::sort(local_ranks, RankIndex::cmp_mod_div, comm);
        timer.stop();
        KASSERT(local_ranks.size() >= 2u); // can happen for small inputs

        auto get_rank = [](RankIndex& r) -> new_char_type { return r.rank; };
        std::vector<new_char_type> recursive_string =
            extract_attribute<RankIndex, new_char_type>(local_ranks, get_rank);

        recursive_string =
            mpi_util::distribute_data_custom(recursive_string, local_sample_size, comm);

        // free memory of ranks
        local_ranks.clear();
        local_ranks.shrink_to_fit();

        // TODO: flexible selection of DC
        // create new instance of PDC3 with templates of new char type size
        PDCX<new_char_type, index_type, DC> rec_pdcx(comm);

        // memory of SA is counted in recursive call
        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;

        recursive_string.clear();
        recursive_string.shrink_to_fit();

        uint64_t local_SA_size = SA.size();
        uint64_t elements_before = mpi_util::ex_prefix_sum(local_SA_size, comm);

        local_ranks.reserve(SA.size());
        for (uint64_t i = 0; i < SA.size(); i++) {
            index_type global_index = map_back(SA[i], total_chars, samples_before);
            index_type rank = 1 + i + elements_before;
            local_ranks.emplace_back(rank, global_index);
        }
        SA.clear();
        SA.shrink_to_fit();

        timer.synchronize_and_start("phase_02_sort_ranks_index");
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

        // free memory of samples
        local_samples.clear();
        local_samples.shrink_to_fit();

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
            RankIndex padding(0, total_chars);
            std::fill_n(std::back_inserter(local_ranks), D, padding);
            local_ranks.shrink_to_fit();
        }

        std::vector<MergeSamples> merge_samples =
            construct_merge_samples(local_string,
                                    local_ranks,
                                    chars_before[process_rank],
                                    chars_at_proc[process_rank]);

        // free memory of local_ranks
        local_ranks.clear();
        local_ranks.shrink_to_fit();

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
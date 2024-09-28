#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "difference_cover.hpp"
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
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/memory_monitor.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"

namespace dsss::dc3 {

using namespace kamping;

struct Statistics {
    void reset() {
        max_depth = 0;
        string_sizes.clear();
        local_string_sizes.clear();
        highest_ranks.clear();
        char_type_used.clear();
    }

    int max_depth = 0;
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

template <typename char_type, typename index_type>
class PDC3 {
    struct SampleString {
        SampleString() : letters({0, 0, 0}), index(0) {}
        SampleString(std::array<char_type, 3> _letters, index_type _index)
            : letters(_letters),
              index(_index) {}

        bool operator<(const SampleString& other) const {
            for (uint i = 0; i < letters.size(); i++) {
                if (letters[i] != other.letters[i]) {
                    return letters[i] < other.letters[i];
                }
            }
            return index < other.index;
        }

        std::string to_string() const {
            std::stringstream ss;
            auto [a, b, c] = letters;
            ss << "(" << a << "," << b << "," << c << ") " << index;
            return ss.str();
        }

        std::array<char_type, 3> letters;
        index_type index;
    };

    struct RankIndex {
        index_type rank;
        index_type index;

        static bool cmp_by_index(const RankIndex& a, const RankIndex& b) {
            return a.index < b.index;
        }
        static bool cmp_mod_div_3(const RankIndex& a, const RankIndex& b) {
            const int a_mod3 = a.index % 3;
            const int b_mod3 = b.index % 3;
            if (a_mod3 != b_mod3) {
                return a_mod3 < b_mod3;
            }
            return a.index / 3 < b.index / 3;
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "(" << rank << " " << index << ")";
            return ss.str();
        }
    };

    struct MergeSamples {
        MergeSamples() : chars({0, 0}), ranks({0, 0}), index(0) {}
        MergeSamples(
            char_type char1, char_type char2, index_type rank1, index_type rank2, index_type idx)
            : chars({char1, char2}),
              ranks({rank1, rank2}),
              index(idx) {}

        std::string to_string() {
            std::stringstream ss;
            ss << "((" << chars[0] << " " << chars[1] << ") (" << ranks[0] << " " << ranks[1]
               << ") " << index << ")";
            return ss.str();
        }
        bool operator<(const MergeSamples& b) const {
            index_type i1 = index % 3;
            index_type i2 = b.index % 3;
            auto [d, r1, r2] = dcx::DC3Param::cmpDepthRanks[i1][i2];

            // compare first d chars
            for (int k = 0; k < d; k++) {
                if (chars[k] != b.chars[k]) {
                    return chars[k] < b.chars[k];
                }
            }

            // tie breaking using ranks
            return ranks[r1] < b.ranks[r2];
        }

        std::array<char_type, 2> chars;
        std::array<index_type, 2> ranks;
        index_type index;
    };

public:
    PDC3(Communicator<>& _comm)
        : atomic_sorter(_comm),
          comm(_comm),
          timer(measurements::timer()),
          memory_monitor(monitor::get_monitor_instance()),
          stats(get_stats_instance()) {
        atomic_sorter.set_sorter(mpi::AtomicSorters::SampleSort);
    }

    // maps the index i from a recursive dc3 call back to the global index
    index_type map_back(index_type i, index_type n) {
        index_type n0 = (n + 2) / 3;
        return i < n0 ? 3 * i + 1 : 3 * (i - n0) + 2;
    }

    void add_padding(std::vector<char_type>& local_data) {
        constexpr char_type padding = 0;
        if (comm.rank() == comm.size() - 1) {
            local_data.push_back(padding);
            local_data.push_back(padding);
            local_data.push_back(padding);
        }
    }
    void remove_padding(std::vector<char_type>& local_data) {
        if (comm.rank() == comm.size() - 1) {
            local_data.pop_back();
            local_data.pop_back();
            local_data.pop_back();
        }
    }

    void clean_up(std::vector<char_type>& local_string) {
        // revert changes made to local string by left shift
        if (comm.rank() < comm.size() - 1) {
            local_string.pop_back();
            local_string.pop_back();
        }
        remove_padding(local_string);
    }

    // compute samples strings of length 3 according to difference cover {1, 2}
    std::vector<SampleString> compute_sample_strings(std::vector<char_type>& local_string,
                                                     index_type chars_before,
                                                     index_type total_chars) {
        // position of first mod 0 character
        uint64_t offset = chars_before % 3;
        offset = (3 - offset) % 3;

        // compute local sample strings
        std::vector<SampleString> local_samples;
        uint64_t size_estimate = ((local_string.size() + 2) / 3) * 2;
        local_samples.reserve(size_estimate);

        for (uint64_t i = 0; i + 2 < local_string.size(); i++) {
            if (i % 3 != offset) {
                index_type index = chars_before + i;
                std::array<char_type, 3> letters = {local_string[i],
                                                    local_string[i + 1],
                                                    local_string[i + 2]};
                local_samples.push_back(SampleString(letters, index));
            }
        }
        // keep a padding sample for last process of in case n mod 3 == 1
        std::array<char_type, 3> padding = {0, 0, 0};
        const bool is_last_rank = comm.rank() == comm.size() - 1;
        if (is_last_rank && total_chars % 3 != 1 && local_samples.back().letters == padding) {
            local_samples.pop_back();
        }
        memory_monitor.add_memory(local_samples, "samples");
        return local_samples;
    }

    // create lexicographic ranks using a prefix sum
    std::vector<RankIndex> compute_lexicographic_ranks(std::vector<SampleString>& local_samples,
                                                       uint64_t num_samples) {
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
        memory_monitor.add_memory(local_ranks, "ranks");
        return local_ranks;
    }

    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      uint64_t chars_before,
                                                      uint64_t chars_at_proc) {
        uint64_t pos = 0;
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(chars_at_proc);

        // for each index in local string
        for (uint64_t local_index = 0; local_index < chars_at_proc; local_index++) {
            // find next position in difference cover
            while (local_index > local_ranks[pos].index - chars_before) {
                pos++;
                KASSERT(pos < local_ranks.size());
            }
            uint64_t global_index = local_index + chars_before;

            KASSERT(local_index + 1 < local_string.size());
            KASSERT(pos + 1 < local_ranks.size());

            char_type char1 = local_string[local_index];
            char_type char2 = local_string[local_index + 1];
            index_type rank1 = local_ranks[pos].rank;
            index_type rank2 = local_ranks[pos + 1].rank;
            merge_samples.emplace_back(char1, char2, rank1, rank2, global_index);
        }
        memory_monitor.add_memory(merge_samples, "merge_samples");
        return merge_samples;
    }

    void dispatch_recursive_call(std::vector<RankIndex>& local_ranks,
                                 uint64_t local_sample_size,
                                 uint64_t last_rank,
                                 uint64_t total_chars) {
        if (total_chars <= 4 * comm.size()) {
            // continue with sequential algorithm
            timer.start("sequential_SA");
            sequential_sa_and_local_ranks(local_ranks, local_sample_size, total_chars);
            timer.stop();
        } else {
            if (last_rank <= std::numeric_limits<uint8_t>::max()) {
                handle_recursive_call<uint8_t>(local_ranks, local_sample_size, total_chars);
            } else if (last_rank <= std::numeric_limits<uint16_t>::max()) {
                handle_recursive_call<uint16_t>(local_ranks, local_sample_size, total_chars);
            } else if (last_rank <= std::numeric_limits<uint32_t>::max()) {
                handle_recursive_call<uint32_t>(local_ranks, local_sample_size, total_chars);
            } else {
                handle_recursive_call<uint64_t>(local_ranks, local_sample_size, total_chars);
            }
            // handle_recursive_call<uint32_t>(local_ranks, local_sample_size, total_chars);
        }
    }

    // sequential SACA and sequential computation ranks computation on root process
    void sequential_sa_and_local_ranks(std::vector<RankIndex>& local_ranks,
                                       uint64_t local_sample_size,
                                       uint64_t total_chars) {
        std::vector<RankIndex> global_ranks = comm.gatherv(send_buf(local_ranks));
        std::vector<index_type> SA;
        if (comm.rank() == 0) {
            std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_mod_div_3);
            auto get_rank = [](RankIndex& r) -> index_type { return r.rank; };
            std::vector<index_type> ranks =
                extract_attribute<RankIndex, index_type>(global_ranks, get_rank);

            // TODO: better sequential SACA
            SA = slow_suffixarray<index_type, index_type>(ranks);
            global_ranks.clear();
            for (uint64_t i = 0; i < SA.size(); i++) {
                index_type global_index = map_back(SA[i], total_chars);
                global_ranks.emplace_back(i + 1, global_index);
            }
            std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_by_index);
        }

        local_ranks.clear();
        memory_monitor.remove_memory(local_ranks, "seq_dist_ranks");
        local_ranks = mpi_util::distribute_data_custom(global_ranks, local_sample_size, comm);
        memory_monitor.add_memory(local_ranks, "seq_dist_ranks");
    }

    std::vector<index_type> pdc3(std::vector<char_type>& local_string);

    template <typename new_char_type>
    void handle_recursive_call(std::vector<RankIndex>& local_ranks,
                               uint64_t local_sample_size,
                               uint64_t total_chars) {
        // sort by (mod 3, div 3)
        memory_monitor.remove_memory(local_ranks, "ranks_mod_div_3");
        timer.start("sort_mod_div_3");
        atomic_sorter.sort(local_ranks, RankIndex::cmp_mod_div_3);
        timer.stop();
        memory_monitor.add_memory(local_ranks, "ranks_mod_div_3");
        KASSERT(local_ranks.size() >= 2u); // can happen for small inputs


        auto get_rank = [](RankIndex& r) -> new_char_type { return r.rank; };
        std::vector<new_char_type> recursive_string =
            extract_attribute<RankIndex, new_char_type>(local_ranks, get_rank);
        memory_monitor.add_memory(recursive_string, "recursive_string");

        // free memory of ranks
        memory_monitor.remove_memory(local_ranks, "ranks");
        local_ranks.clear();
        local_ranks.shrink_to_fit();

        // create new instance of PDC3 with templates of new char type size
        PDC3<new_char_type, index_type> rec_pdc3(comm);

        // memory of SA is counted in recursive call
        recursion_depth++;
        rec_pdc3.recursion_depth = recursion_depth;
        std::vector<index_type> SA = rec_pdc3.compute_sa(recursive_string);
        recursion_depth--;

        memory_monitor.remove_memory(recursive_string, "recursive_string");
        recursive_string.clear();
        recursive_string.shrink_to_fit();

        uint64_t local_SA_size = SA.size();
        uint64_t elements_before = mpi_util::ex_prefix_sum(local_SA_size, comm);

        local_ranks.reserve(SA.size());
        for (uint64_t i = 0; i < SA.size(); i++) {
            index_type global_index = map_back(SA[i], total_chars);
            index_type rank = 1 + i + elements_before;
            local_ranks.emplace_back(rank, global_index);
        }
        memory_monitor.add_memory(local_ranks, "ranks");
        memory_monitor.remove_memory(SA, "SA delete");
        SA.clear();
        SA.shrink_to_fit();

        memory_monitor.remove_memory(local_ranks, "ranks_sort");
        timer.start("sort_ranks_index");
        atomic_sorter.sort(local_ranks, RankIndex::cmp_by_index);
        timer.stop();
        memory_monitor.add_memory(local_ranks, "ranks_sort");

        memory_monitor.remove_memory(local_ranks, "ranks_dist");
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
        memory_monitor.add_memory(local_ranks, "ranks_dist");
    }

    std::vector<index_type> compute_sa(std::vector<char_type>& local_string) {
        timer.start("pdc3");

        const int process_rank = comm.rank();

        // figure out lengths of the other strings
        auto chars_at_proc = comm.allgather(send_buf(local_string.size()));
        uint64_t total_chars = std::accumulate(chars_at_proc.begin(), chars_at_proc.end(), 0);

        // number of chars before processor i
        std::vector<uint64_t> chars_before(comm.size());
        std::exclusive_scan(chars_at_proc.begin(), chars_at_proc.end(), chars_before.begin(), 0);

        // n0 + n2 to account for possible dummy sample of n1
        uint64_t num_samples = (total_chars + 2) / 3 + total_chars / 3;

        add_padding(local_string);
        memory_monitor.add_memory(local_string, "string");

        // logging
        stats.max_depth = std::max(stats.max_depth, recursion_depth);
        stats.string_sizes.push_back(total_chars);
        stats.local_string_sizes.push_back(local_string.size());
        stats.char_type_used.push_back(8 * sizeof(char_type));

        memory_monitor.remove_memory(local_string, "shift_string");
        mpi_util::shift_entries_left(local_string, 2, comm);
        local_string.shrink_to_fit();
        memory_monitor.add_memory(local_string, "shift_string");

        std::vector<SampleString> local_samples =
            compute_sample_strings(local_string, chars_before[process_rank], total_chars);


        const uint64_t local_sample_size = local_samples.size();

        memory_monitor.remove_memory(local_samples, "samples_sort");
        timer.start("sort_local_samples");
        atomic_sorter.sort(local_samples, std::less<>{});
        timer.stop();

        local_samples.shrink_to_fit();
        memory_monitor.add_memory(local_samples, "samples_sort");

        // can happen for small inputs
        KASSERT(local_samples.size() > 0u);

        // adds a dummy sample for last process
        KASSERT(local_string.size() >= 1u);
        memory_monitor.remove_memory(local_samples, "shift_samples");
        SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
        local_samples.push_back(recv_sample);
        local_samples.shrink_to_fit();
        memory_monitor.add_memory(local_samples, "shift_samples");

        std::vector<RankIndex> local_ranks =
            compute_lexicographic_ranks(local_samples, num_samples);

        // free memory of samples
        memory_monitor.remove_memory(local_samples, "samples");
        local_samples.clear();
        local_samples.shrink_to_fit();

        uint64_t last_rank = local_ranks.empty() ? 0 : local_ranks.back().rank;
        comm.bcast_single(send_recv_buf(last_rank), root(comm.size() - 1));
        stats.highest_ranks.push_back(last_rank);
        bool chars_distinct = last_rank >= num_samples;


        if (chars_distinct) {
            memory_monitor.remove_memory(local_ranks, "ranks_sort_base");
            timer.start("sort_local_ranks_index_base");
            atomic_sorter.sort(local_ranks, RankIndex::cmp_by_index);
            timer.stop();
            memory_monitor.add_memory(local_ranks, "ranks_sort_base");

            memory_monitor.remove_memory(local_ranks, "ranks_dist_base");
            local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
            local_ranks.shrink_to_fit();
            memory_monitor.add_memory(local_ranks, "ranks_dist_base");

        } else {
            dispatch_recursive_call(local_ranks, local_sample_size, last_rank, total_chars);
        }

        memory_monitor.remove_memory(local_ranks, "shift_left_ranks");
        mpi_util::shift_entries_left(local_ranks, 2, comm);
        local_ranks.shrink_to_fit();
        memory_monitor.add_memory(local_ranks, "shift_left_ranks");

        // add two paddings with rank 0
        if (comm.rank() == comm.size() - 1) {
            memory_monitor.remove_memory(local_ranks, "padding");
            local_ranks.push_back(RankIndex(0, total_chars));
            local_ranks.push_back(RankIndex(0, total_chars));
            local_ranks.shrink_to_fit();
            memory_monitor.add_memory(local_ranks, "padding");
        }

        std::vector<MergeSamples> merge_samples =
            construct_merge_samples(local_string,
                                    local_ranks,
                                    chars_before[process_rank],
                                    chars_at_proc[process_rank]);

        // free memory of local_ranks
        memory_monitor.remove_memory(local_ranks, "ranks");
        local_ranks.clear();
        local_ranks.shrink_to_fit();

        memory_monitor.remove_memory(merge_samples, "merge_samples_sort");
        timer.start("sort_merge_samples");
        atomic_sorter.sort(merge_samples, std::less<>{});
        timer.stop();
        memory_monitor.add_memory(merge_samples, "merge_samples_sort");

        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);
        memory_monitor.add_memory(local_SA, "SA");

        clean_up(local_string);

        memory_monitor.remove_memory(merge_samples, "merge_samples");

        timer.stop(); // pdc3

        return local_SA;
    }

    void report_time() {
        comm.barrier();
        // timer.aggregate_and_print(measurements::SimpleJsonPrinter<>{});
        timer.aggregate_and_print(measurements::FlatPrinter{});
        std::cout << "\n";
        comm.barrier();
    }

    void report_memory(bool print_history = false) {
        comm.barrier();
        if (print_history) {
            std::string msg = "History \n" + memory_monitor.history_mb_to_string() + "\n";
            print_result_on_root(msg, comm);
        }

        monitor::MemoryKey peak_memory = memory_monitor.get_peak_memory();
        std::string msg2 = "Memory peak: " + peak_memory.to_string_mb();
        print_result(msg2, comm);


        uint64_t local_string_bytes = stats.local_string_sizes.front() * sizeof(char_type);
        double blow_up_factor = (double)peak_memory.get_memory_bytes() / local_string_bytes;
        std::stringstream ss;
        ss << "Blow up factor: " << std::fixed << std::setprecision(2) << blow_up_factor << "\n";
        print_result(ss.str(), comm);

        comm.barrier();
    }

    void report_stats() {
        comm.barrier();
        if (comm.rank() == comm.size() - 1) {
            std::cout << "\nStatistics:\n";
            std::cout << "max depth: " << stats.max_depth << std::endl;
            std::cout << "string sizes: ";
            print_vector(stats.string_sizes);
            std::cout << "highest rank: ";
            print_vector(stats.highest_ranks);
            std::cout << "char type bits: ";
            print_vector(stats.char_type_used);
            std::cout << "\n";
        }
        comm.barrier();
    }

    void reset() {
        memory_monitor.reset();
        stats.reset();
        recursion_depth = 0;
        timer.clear();
    }

    constexpr static bool DBG = false;
    constexpr static bool use_recursion = true;

    mpi::SortingWrapper atomic_sorter;

    Communicator<>& comm;
    measurements::Timer<Communicator<>>& timer;
    monitor::MemoryMonitor& memory_monitor;
    Statistics& stats;
    int recursion_depth;
};

} // namespace dsss::dc3
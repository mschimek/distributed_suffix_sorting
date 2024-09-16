#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kassert/kassert.hpp"
#include "memory_monitor.hpp"
#include "mpi_util.hpp"
#include "printing.hpp"
#include "sa_check.hpp"
#include "sort.hpp"
#include "util.hpp"

namespace dsss::dc3 {

using namespace kamping;
struct SampleString {
    SampleString() : letters({0, 0, 0}), index(0) {}
    SampleString(std::array<int, 3> _letters, int _index) : letters(_letters), index(_index) {}

    bool operator<(const SampleString& other) const {
        for (uint i = 0; i < letters.size(); i++) {
            if (letters[i] != other.letters[i]) {
                return letters[i] < other.letters[i];
            }
        }
        return index < other.index;
    }

    std::array<int, 3> letters;
    int index;
};

std::ostream& operator<<(std::ostream& os, const SampleString& sample) {
    auto [a, b, c] = sample.letters;
    os << "(" << a << "," << b << "," << c << ") " << sample.index;
    return os;
}

struct RankIndex {
    int rank;
    int index;
};

bool cmp_by_index(const RankIndex& a, const RankIndex& b) { return a.index < b.index; }
bool cmp_mod_div_3(const RankIndex& a, const RankIndex& b) {
    const int a_mod3 = a.index % 3;
    const int b_mod3 = b.index % 3;
    if (a_mod3 != b_mod3) {
        return a_mod3 < b_mod3;
    }
    return a.index / 3 < b.index / 3;
}

std::ostream& operator<<(std::ostream& os, const RankIndex& rank_index) {
    os << "(" << rank_index.rank << " " << rank_index.index << ")";
    return os;
}

struct DC3Param {
    static const unsigned int X = 3;
    static const unsigned int D = 2;

    static const unsigned int DC[D];
    static const int cmpDepthRanks[X][X][3];
};

const unsigned int DC3Param::DC[] = {1, 2};

const int DC3Param::cmpDepthRanks[3][3][3] = {
    {{1, 0, 0}, {1, 0, 1}, {2, 1, 1}},
    {{1, 1, 0}, {0, 0, 0}, {0, 0, 0}},
    {{2, 1, 1}, {0, 0, 0}, {0, 0, 0}},
};

struct MergeSamples {
    MergeSamples() : chars({0, 0}), ranks({0, 0}), index(0) {}
    MergeSamples(int char1, int char2, int rank1, int rank2, int idx)
        : chars({char1, char2}),
          ranks({rank1, rank2}),
          index(idx) {}

    std::array<int, 2> chars;
    std::array<int, 2> ranks;
    int index;
};

std::ostream& operator<<(std::ostream& os, const MergeSamples& m) {
    os << "((" << m.chars[0] << " " << m.chars[1] << ") (" << m.ranks[0] << " " << m.ranks[1]
       << ") " << m.index << ")";
    return os;
}

bool operator<(const MergeSamples& a, const MergeSamples& b) {
    int i1 = a.index % 3;
    int i2 = b.index % 3;
    auto [d, r1, r2] = DC3Param::cmpDepthRanks[i1][i2];

    // compare first d chars
    for (int k = 0; k < d; k++) {
        if (a.chars[k] != b.chars[k]) {
            return a.chars[k] < b.chars[k];
        }
    }

    // tie breaking using ranks
    return a.ranks[r1] < b.ranks[r2];
}

class PDC3 {
public:
    PDC3(Communicator<>& _comm) : comm(_comm), recursion_depth(0) {}

    // maps the index i from a recursive dc3 call back to the global index
    int map_back(int i, int n) {
        int n0 = (n + 2) / 3;
        return i < n0 ? 3 * i + 1 : 3 * (i - n0) + 2;
    }

    void add_padding(std::vector<int>& local_data) {
        constexpr int padding = 0;
        if (comm.rank() == comm.size() - 1) {
            local_data.push_back(padding);
            local_data.push_back(padding);
            local_data.push_back(padding);
        }
    }
    void remove_padding(std::vector<int>& local_data) {
        if (comm.rank() == comm.size() - 1) {
            local_data.pop_back();
            local_data.pop_back();
            local_data.pop_back();
        }
    }

    void clean_up(std::vector<int>& local_string) {
        // revert changes made to local string by left shift
        if (comm.rank() < comm.size() - 1) {
            local_string.pop_back();
            local_string.pop_back();
        }
        remove_padding(local_string);
    }

    // compute samples strings of length 3 according to difference cover {1, 2}
    std::vector<SampleString>
    compute_sample_strings(std::vector<int>& local_string, int chars_before, int total_chars) {
        // position of first mod 0 character
        uint offset = chars_before % 3;
        offset = (3 - offset) % 3;

        // compute local sample strings
        std::vector<SampleString> local_samples;
        int size_estimate = ((local_string.size() + 2) / 3) * 2;
        local_samples.reserve(size_estimate);

        for (uint i = 0; i + 2 < local_string.size(); i++) {
            if (i % 3 != offset) {
                int index = chars_before + i;
                std::array<int, 3> letters = {local_string[i],
                                              local_string[i + 1],
                                              local_string[i + 2]};
                local_samples.push_back(SampleString(letters, index));
            }
        }
        // keep a padding sample for last process of in case n mod 3 == 1
        std::array<int, 3> padding = {0, 0, 0};
        const bool is_last_rank = comm.rank() == comm.size() - 1;
        if (is_last_rank && total_chars % 3 != 1 && local_samples.back().letters == padding) {
            local_samples.pop_back();
        }
        memory_monitor.add_memory(local_samples, "samples_create");
        return local_samples;
    }

    // create lexicographic ranks using a prefix sum
    std::vector<RankIndex> compute_lexicographic_ranks(std::vector<SampleString>& local_samples,
                                                       int num_samples) {
        std::vector<RankIndex> local_ranks;

        // exclude sample from process i + 1
        int num_ranks = local_samples.size() - 1;
        local_ranks.reserve(num_ranks);

        // compute local ranks
        int prev_rank = 0;
        for (int i = 0; i < num_ranks; i++) {
            KASSERT(i + 1 < (int)local_samples.size());
            local_ranks.emplace_back(prev_rank, local_samples[i].index);
            int changed = local_samples[i].letters != local_samples[i + 1].letters ? 1 : 0;
            prev_rank += changed;
        }

        // TODO: remove this code later
        mpi_util::check_expected_size(num_samples, local_ranks.size(), comm);

        // shift ranks by 1 + prefix sum
        int ranks_before = mpi_util::ex_prefix_sum(prev_rank, comm);
        std::for_each(local_ranks.begin(), local_ranks.end(), [&](RankIndex& x) {
            x.rank += 1 + ranks_before;
        });
        memory_monitor.add_memory(local_ranks, "ranks_create");
        return local_ranks;
    }

    bool check_chars_distinct(std::vector<RankIndex>& local_ranks, int num_samples) {
        int last_rank = local_ranks.empty() ? 0 : local_ranks.back().rank;
        bool chars_distinct = last_rank >= num_samples;
        comm.bcast_single(send_recv_buf(chars_distinct), root(comm.size() - 1));
        return chars_distinct;
    }

    std::vector<MergeSamples> construct_merge_samples(std::vector<int>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      int chars_before,
                                                      int chars_at_proc) {
        int pos = 0;
        int rank1_offset[3] = {0, 0, 0};
        int rank2_offset[3] = {1, 1, 1};
        std::vector<MergeSamples> merge_samples;
        merge_samples.reserve(chars_at_proc);

        // for each index in local string
        for (int i = 0; i < chars_at_proc; i++) {
            // convert global index to local index
            // for mod 0, find next mod 1 position
            // for mod 1,2 find next mod 1, 2 position
            while (i > local_ranks[pos].index - chars_before) {
                pos++;
                KASSERT(pos < (int)local_ranks.size());
            }

            int local_index = i;
            int global_index = local_index + chars_before;
            int mod = global_index % 3;

            KASSERT(local_index + 1 < (int)local_string.size());
            KASSERT(pos + rank1_offset[mod] < (int)local_ranks.size());
            KASSERT(pos + rank2_offset[mod] < (int)local_ranks.size());

            int char1 = local_string[local_index];
            int char2 = local_string[local_index + 1];
            int rank1 = local_ranks[pos + rank1_offset[mod]].rank;
            int rank2 = local_ranks[pos + rank2_offset[mod]].rank;
            merge_samples.emplace_back(char1, char2, rank1, rank2, global_index);
        }
        memory_monitor.add_memory(merge_samples, "merge_samples_create");
        return merge_samples;
    }

    // sequential SACA and sequential computation ranks computation on root process
    void sequential_sa_and_local_ranks(std::vector<RankIndex>& local_ranks,
                                       int local_sample_size,
                                       int total_chars) {
        std::vector<RankIndex> global_ranks = comm.gatherv(send_buf(local_ranks));
        std::vector<int> SA;
        if (comm.rank() == 0) {
            auto get_rank = [](RankIndex& r) -> int { return r.rank; };
            std::vector<int> ranks = extract_attribute<RankIndex, int>(global_ranks, get_rank);

            // TODO: better sequential SACA
            SA = slow_suffixarray(ranks);
            global_ranks.clear();
            for (uint i = 0; i < SA.size(); i++) {
                int global_index = map_back(SA[i], total_chars);
                global_ranks.emplace_back(i + 1, global_index);
            }
            std::sort(global_ranks.begin(), global_ranks.end(), cmp_by_index);
        }

        memory_monitor.remove_memory(local_ranks, "seq_ranks");
        local_ranks.clear();
        local_ranks = mpi_util::distribute_data_custom(global_ranks, local_sample_size, comm);
        memory_monitor.add_memory(local_ranks, "seq_ranks");
    }

    std::vector<int> pdc3(std::vector<int>& local_string);

    void handle_recursive_call(std::vector<RankIndex>& local_ranks, int total_chars) {
        auto get_rank = [](RankIndex& r) -> int { return r.rank; };
        std::vector<int> recursive_string =
            extract_attribute<RankIndex, int>(local_ranks, get_rank);

        memory_monitor.add_memory(recursive_string, "recursive_string");

        // free memory
        memory_monitor.remove_memory(local_ranks, "ranks");
        local_ranks.clear();
        local_ranks.shrink_to_fit();

        recursion_depth++;
        std::vector<int> SA = call_pdc3(recursive_string);
        recursion_depth--;

        memory_monitor.remove_memory(recursive_string, "recursive_string");
        recursive_string.clear();
        recursive_string.shrink_to_fit();

        size_t local_SA_size = SA.size();
        size_t elements_before = mpi_util::ex_prefix_sum(local_SA_size, comm);

        local_ranks.reserve(SA.size());
        for (uint i = 0; i < SA.size(); i++) {
            int global_index = map_back(SA[i], total_chars);
            int rank = 1 + i + elements_before;
            local_ranks.emplace_back(rank, global_index);
        }
        memory_monitor.add_memory(local_ranks, "ranks");
    }

    constexpr static bool DBG = false;
    constexpr static bool use_recursion = false;


    std::vector<int> call_pdc3(std::vector<int>& local_string) {
        timer.start("pdc3");

        const int process_rank = comm.rank();
        add_padding(local_string);
        memory_monitor.add_memory(local_string, "local_string");

        // figure out lengths of the other strings
        auto chars_at_proc = comm.allgather(send_buf(local_string.size()));
        int64_t total_chars = std::accumulate(chars_at_proc.begin(), chars_at_proc.end(), 0) - 3;
        chars_at_proc.back() -= 3;

        // number of chars before processor i
        std::vector<int> chars_before(comm.size());
        std::exclusive_scan(chars_at_proc.begin(), chars_at_proc.end(), chars_before.begin(), 0);
        std::vector<int64_t> num_mod = {(total_chars + 2) / 3,
                                        (total_chars + 1) / 3,
                                        total_chars / 3};

        // n0 + n2 to account for possible dummy sample of n1
        int64_t num_samples = num_mod[0] + num_mod[2];

        memory_monitor.remove_memory(local_string, "shift_left_string");
        mpi_util::shift_entries_left(local_string, 2, comm);
        local_string.shrink_to_fit();
        memory_monitor.add_memory(local_string, "shift_left_string");


        std::vector<SampleString> local_samples =
            compute_sample_strings(local_string, chars_before[process_rank], total_chars);

        const size_t local_sample_size = local_samples.size();

        // TODO: remove this code later
        mpi_util::check_expected_size(num_samples, local_samples.size(), comm);

        memory_monitor.remove_memory(local_samples, "samples_sort");
        timer.start("sort_local_samples");
        mpi::sort(local_samples, std::less<>{}, comm);
        timer.stop();
        local_samples.shrink_to_fit();
        memory_monitor.add_memory(local_samples, "samples_sort");

        // can happen for small inputs
        KASSERT(local_samples.size() > 0u);

        // adds a dummy sample for last process
        KASSERT(local_string.size() >= 1u);
        memory_monitor.remove_memory(local_samples, "shift_left_samples");
        SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
        local_samples.push_back(recv_sample);
        local_samples.shrink_to_fit();
        memory_monitor.add_memory(local_samples, "shift_left_samples");

        std::vector<RankIndex> local_ranks =
            compute_lexicographic_ranks(local_samples, num_samples);

        // free memory of samples
        memory_monitor.remove_memory(local_samples, "samples_delete");
        local_samples.clear();
        local_samples.shrink_to_fit();

        bool chars_distinct = check_chars_distinct(local_ranks, num_samples);

        if (chars_distinct) {
            memory_monitor.remove_memory(local_ranks, "ranks_sort_base");
            timer.start("sort_local_ranks_index_base");
            mpi::sort(local_ranks, cmp_by_index, comm);
            timer.stop();
            memory_monitor.add_memory(local_ranks, "ranks_sort_base");

            memory_monitor.remove_memory(local_ranks, "ranks_dist_base");
            local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
            local_ranks.shrink_to_fit();
            memory_monitor.add_memory(local_ranks, "ranks_dist_base");

        } else {
            // reorder rank sorted by (i mod 3, i div 3) using sorting or permutation
            memory_monitor.remove_memory(local_ranks, "ranks_mod_div_3");
            timer.start("sort_mod_div_3");
            mpi::sort(local_ranks, cmp_mod_div_3, comm);
            timer.stop();
            memory_monitor.add_memory(local_ranks, "ranks_mod_div_3");
            // can happen for small inputs
            KASSERT(local_ranks.size() >= 2u);

            if (use_recursion) {
                handle_recursive_call(local_ranks, total_chars);

                memory_monitor.remove_memory(local_ranks, "ranks_sort");
                timer.start("sort_ranks_index");
                mpi::sort(local_ranks, cmp_by_index, comm);
                timer.stop();
                memory_monitor.add_memory(local_ranks, "ranks_sort");

                memory_monitor.remove_memory(local_ranks, "ranks_dist");
                local_ranks =
                    mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
                memory_monitor.add_memory(local_ranks, "ranks_dist");
            } else {
                timer.start("sequential_SA");
                sequential_sa_and_local_ranks(local_ranks, local_sample_size, total_chars);
                timer.stop();
            }
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
        mpi::sort(merge_samples, std::less<>{}, comm);
        timer.stop();
        memory_monitor.add_memory(merge_samples, "merge_samples_sort");

        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<int> local_SA = extract_attribute<MergeSamples, int>(merge_samples, get_index);
        memory_monitor.add_memory(local_SA, "SA");

        clean_up(local_string);

        timer.stop(); // pdc3

        return local_SA;
    }

    void report_time() {
        timer.aggregate_and_print(measurements::SimpleJsonPrinter<>{});
        timer.aggregate_and_print(measurements::FlatPrinter{});
        std::cout << "\n";
    }

    void report_memory() {
        comm.barrier();
        // std::string msg = "History \n" + memory_monitor.history_mb_to_string() + "\n";
        // print_result_on_root(msg, comm);

        MemoryKey peak_memory = memory_monitor.get_peak_memory();
        std::string msg2 = "Memory peak: " + peak_memory.to_string_mb();
        print_result(msg2, comm);
    }

    void reset() {
        memory_monitor.reset();
        recursion_depth = 0;
    }

    Communicator<>& comm;
    measurements::Timer<Communicator<>> timer;
    MemoryMonitor memory_monitor;
    int recursion_depth;
};

} // namespace dsss::dc3
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
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kassert/kassert.hpp"
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

// maps the index i from a recursive dc3 call back to the global index
int map_back(int i, int n) {
    int n0 = (n + 2) / 3;
    return i < n0 ? 3 * i + 1 : 3 * (i - n0) + 2;
}

void add_padding(std::vector<int>& local_data, Communicator<>& comm) {
    constexpr int padding = 0;
    if (comm.rank() == comm.size() - 1) {
        local_data.push_back(padding);
        local_data.push_back(padding);
        local_data.push_back(padding);
    }
}
void remove_padding(std::vector<int>& local_data, Communicator<>& comm) {
    if (comm.rank() == comm.size() - 1) {
        local_data.pop_back();
        local_data.pop_back();
        local_data.pop_back();
    }
}

// compute samples strings of length 3 according to difference cover {1, 2}
std::vector<SampleString> compute_sample_strings(std::vector<int>& local_string,
                                                 int chars_before,
                                                 int total_chars,
                                                 Communicator<>& comm) {
    // position of first mod 0 character
    uint offset = chars_before % 3;
    offset = (3 - offset) % 3;

    // compute local sample strings
    std::vector<SampleString> local_samples;
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
    return local_samples;
}

// create lexicographic ranks using a prefix sum
std::vector<RankIndex> compute_lexicographic_ranks(std::vector<SampleString>& local_samples,
                                                   int num_samples,
                                                   Communicator<>& comm) {
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
    return local_ranks;
}

bool check_chars_distinct(std::vector<RankIndex>& local_ranks,
                          int num_samples,
                          Communicator<>& comm) {
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
    return merge_samples;
}

// sequential SACA and sequential computation ranks computation on root process
void sequential_sa_and_local_ranks(std::vector<RankIndex>& local_ranks,
                                   std::vector<size_t>& chars_at_proc,
                                   int total_chars,
                                   Communicator<>& comm) {
    std::vector<RankIndex> global_ranks = comm.gatherv(send_buf(local_ranks));
    std::vector<int> SA;
    if (comm.rank() == 0) {
        std::vector<int> ranks;
        for (auto [rank, index]: global_ranks) {
            ranks.push_back(rank);
        }
        // TODO: better sequential SACA
        SA = slow_suffixarray(ranks);
        global_ranks.clear();
        for (uint i = 0; i < SA.size(); i++) {
            int global_index = map_back(SA[i], total_chars);
            global_ranks.emplace_back(i + 1, global_index);
        }
    }

    std::sort(global_ranks.begin(), global_ranks.end(), cmp_by_index);
    // alternative
    // local_ranks = mpi_util::distribute_data_custom(global_ranks, local_sample_size, comm);

    local_ranks.clear();
    // send to each processor the size of its local_string
    // only relevant to root
    std::vector<int> send_cnts(comm.size());
    for (uint i = 0; i < global_ranks.size(); i++) {
        int index = global_ranks[i].index;
        // check to which process this index belongs
        for (int j = 0; j < (int)comm.size(); j++) {
            if (index < (int)chars_at_proc[j]) {
                send_cnts[j]++;
                break;
            }
            index -= chars_at_proc[j];
        }
    }

    comm.scatterv(send_buf(global_ranks),
                  recv_buf<resize_to_fit>(local_ranks),
                  send_counts(send_cnts));
}

std::vector<int> pdc3(std::vector<int>& local_string, Communicator<>& comm);

void handle_recursive_call(std::vector<RankIndex>& local_ranks,
                           int total_chars,
                           Communicator<>& comm) {
    auto get_rank = [](RankIndex& r) -> int { return r.rank; };
    std::vector<int> recursive_string = extract_attribute<RankIndex, int>(local_ranks, get_rank);

    // free memory
    local_ranks.clear();
    local_ranks.shrink_to_fit();

    std::vector<int> SA = pdc3(recursive_string, comm);

    size_t local_SA_size = SA.size();
    size_t elements_before = mpi_util::ex_prefix_sum(local_SA_size, comm);

    local_ranks.reserve(SA.size());
    for (uint i = 0; i < SA.size(); i++) {
        int global_index = map_back(SA[i], total_chars);
        int rank = 1 + i + elements_before;
        local_ranks.emplace_back(rank, global_index);
    }
}

constexpr static bool DBG = false;

std::vector<int> pdc3(std::vector<int>& local_string, Communicator<>& comm) {
    const int process_rank = comm.rank();
    const int num_processes = comm.size();
    const int root_rank = 0;
    const int last_process = num_processes - 1;

    add_padding(local_string, comm);

    // figure out lengths of the other strings
    auto chars_at_proc = comm.allgather(send_buf(local_string.size()));
    int64_t total_chars = std::accumulate(chars_at_proc.begin(), chars_at_proc.end(), 0) - 3;
    chars_at_proc.back() -= 3;

    // number of chars before processor i
    std::vector<int> chars_before(num_processes);
    std::exclusive_scan(chars_at_proc.begin(), chars_at_proc.end(), chars_before.begin(), 0);
    std::vector<int64_t> num_mod = {(total_chars + 2) / 3, (total_chars + 1) / 3, total_chars / 3};

    // n0 + n2 to account for possible dummy sample of n1
    int64_t num_samples = num_mod[0] + num_mod[2];

    if (DBG)
        print_concatenated(local_string, comm, "local string");

    mpi_util::shift_entries_left(local_string, 2, comm);

    std::vector<SampleString> local_samples =
        compute_sample_strings(local_string, chars_before[process_rank], total_chars, comm);
    const size_t local_sample_size = local_samples.size();

    if (DBG) {
        print_concatenated(local_samples, comm, "local_samples");
    }

    // TODO: remove this code later
    mpi_util::check_expected_size(num_samples, local_samples.size(), comm);

    // sort local samples
    mpi::sort(local_samples, std::less<>{}, comm);

    // can happen for small inputs
    KASSERT(local_samples.size() > 0u);

    if (DBG) {
        print_concatenated(local_samples, comm, "local_samples after sort");
    }

    // processor i + 1 sends first sample to processor i
    // adds a dummy sample for last process
    KASSERT(local_string.size() >= 1u);
    SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
    local_samples.push_back(recv_sample);

    if (DBG)
        print_concatenated(local_samples, comm, "local samples after shift");

    std::vector<RankIndex> local_ranks =
        compute_lexicographic_ranks(local_samples, num_samples, comm);

    // free memory of samples
    local_samples.clear();
    local_samples.shrink_to_fit();

    if (DBG)
        print_concatenated(local_ranks, comm, "local ranks");

    bool chars_distinct = check_chars_distinct(local_ranks, num_samples, comm);

    if (DBG)
        print_on_root("chars distinct: " + std::to_string(chars_distinct), comm);

    if (chars_distinct) {
        mpi::sort(local_ranks, cmp_by_index, comm);
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
    } else {
        // reorder rank sorted by (i mod 3, i div 3) using sorting or permutation
        mpi::sort(local_ranks, cmp_mod_div_3, comm);

        // can happen for small inputs
        KASSERT(local_ranks.size() >= 2u);

        constexpr bool use_recursion = true;
        if (use_recursion) {
            if (DBG)
                print_on_root("--> recursion", comm);

            handle_recursive_call(local_ranks, total_chars, comm);
            mpi::sort(local_ranks, cmp_by_index, comm);
            local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
        } else {
            sequential_sa_and_local_ranks(local_ranks, chars_at_proc, total_chars, comm);
        }
    }

    mpi_util::shift_entries_left(local_ranks, 2, comm);

    // add two paddings with rank 0
    if (process_rank == last_process) {
        local_ranks.push_back(RankIndex(0, total_chars));
        local_ranks.push_back(RankIndex(0, total_chars));
    }

    if (DBG) {
        size_t local_ranks_size = local_ranks.size();
        print_concatenated(local_string, comm, "local string");
        print_concatenated(local_ranks, comm, "local ranks");
        print_concatenated(local_sample_size, comm, "local_sample_size");
        print_concatenated(local_ranks_size, comm, "local_ranks_size");
    }
    std::vector<MergeSamples> merge_samples = construct_merge_samples(local_string,
                                                                      local_ranks,
                                                                      chars_before[process_rank],
                                                                      chars_at_proc[process_rank]);

    // free memory of local_ranks
    local_ranks.clear();
    local_ranks.shrink_to_fit();

    if (DBG)
        print_concatenated(merge_samples, comm, "merge samples before sorting");

    mpi::sort(merge_samples, std::less<>{}, comm);

    if (DBG)
        print_concatenated(merge_samples, comm, "merge samples after sorting");

    auto get_index = [](MergeSamples& m) { return m.index; };
    std::vector<int> local_SA = extract_attribute<MergeSamples, int>(merge_samples, get_index);

    if (DBG) {
        print_concatenated(local_string, comm);
        print_concatenated(local_SA, comm);
    }

    // revert changes made to local string by left shift
    if (process_rank < last_process) {
        local_string.pop_back();
        local_string.pop_back();
    }
    remove_padding(local_string, comm);

    if (DBG) {
        bool sa_ok = check_suffixarray(local_SA, local_string, comm);
        if (!sa_ok) {
            std::vector<int> global_string = comm.gatherv(send_buf(local_string));
            if (process_rank == root_rank) {
                std::vector<int> sa_correct = slow_suffixarray(global_string);
                std::cout << "correct SA: \n";
                print_vector(sa_correct);
            }
        }
        KASSERT(sa_ok);
        print_on_root("--> SA ok", comm);
    }
    return local_SA;
}

} // namespace dsss::dc3
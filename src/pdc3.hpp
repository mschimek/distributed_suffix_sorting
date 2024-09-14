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
#include "sort.hpp"
#include "util.hpp"

namespace dsss::dc3 {

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

constexpr static bool DBG = false;

struct RankIndex {
    int rank;
    int index;
};

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

std::vector<int> pdc3(std::vector<int>& local_string, kamping::Communicator<>& comm) {
    using namespace kamping;

    int process_rank = comm.rank();
    int num_processes = comm.size();
    int root_rank = 0;
    int last_process = num_processes - 1;

    // check if padding exists
    if (process_rank == last_process) {
        int n = local_string.size();
        KASSERT(local_string[n - 1] == 0);
        KASSERT(local_string[n - 2] == 0);
        KASSERT(local_string[n - 3] == 0);
    }

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
    bool n1_padding = total_chars % 3 == 1;

    // process i sends first two characters to process i - 1
    KASSERT(local_string.size() >= 2u);
    std::vector<int> recv_chars = mpi_util::shift_left(local_string, 2, comm);
    if (process_rank < last_process) {
        local_string.insert(local_string.end(), recv_chars.begin(), recv_chars.end());
    }

    // TODO: barrier necessary?
    comm.barrier();

    // position of first mod 0 character
    uint offset = chars_before[process_rank] % 3;
    offset = (3 - offset) % 3;

    // compute local sample strings
    std::vector<SampleString> local_samples;
    for (uint i = 0; i + 2 < local_string.size(); i++) {
        if (i % 3 != offset) {
            int index = chars_before[process_rank] + i;
            std::array<int, 3> letters = {local_string[i],
                                          local_string[i + 1],
                                          local_string[i + 2]};
            local_samples.push_back(SampleString(letters, index));
        }
    }
    // keep a padding sample for last process of in case n mod 3 == 1
    std::array<int, 3> padding = {0, 0, 0};
    if (process_rank == last_process && total_chars % 3 != 1
        && local_samples.back().letters == padding) {
        local_samples.pop_back();
    }

    if (DBG) {
        print_concatenated(local_samples, comm, "local_samples");
    }

    // TODO: remove this code later
    mpi_util::check_expected_size(num_samples, local_samples.size(), comm);

    // sort local samples
    mpi::sort(local_samples, std::less<>{}, comm);

    // processor i + 1 sends first sample to processor i
    KASSERT(local_string.size() >= 1u);
    SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
    // adds dummy sample for last process
    local_samples.push_back(recv_sample);
    comm.barrier();

    if (DBG)
        print_concatenated(local_samples, comm, "local samples");

    // create lexicographic ranks using a prefix sum
    std::vector<RankIndex> local_ranks;
    // exclude sample for next process
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

    if (DBG)
        print_concatenated(local_ranks, comm, "local ranks");

    // last processor checks if all ranks are distinct
    bool chars_distinct = prev_rank >= num_samples;
    comm.bcast_single(send_recv_buf(chars_distinct), root(last_process));

    if (DBG)
        print_on_root("chars distinct: " + std::to_string(chars_distinct), comm);

    if (chars_distinct) {
        auto by_second = [&](RankIndex const& a, RankIndex const& b) { return a.index < b.index; };
        mpi::sort(local_ranks, by_second, comm);
    } else {
        // reorder rank sorted by (i mod 3, i div 3) using sorting or permutation
        auto mod_div = [&](RankIndex const& a, RankIndex const& b) {
            int a_mod3 = a.index % 3;
            int b_mod3 = b.index % 3;
            if (a_mod3 != b_mod3) {
                return a_mod3 < b_mod3;
            }
            return a.index / 3 < b.index / 3;
        };
        mpi::sort(local_ranks, mod_div, comm);

        bool use_recursion = false;
        if (use_recursion) {
            // std::vector<int> recursive_string;
            // recursive_string.reserve(local_ranks.size());
            // for(RankIndex &ri : local_ranks) {
            //     recursive_string.push_back(ri.rank);
            // }
            // std::vector<int> SA = pdc3(recursive_string, comm);

        } else {
            // for now sequential SA on root process
            auto global_ranks = comm.gatherv(send_buf(local_ranks));
            std::vector<int> SA;
            if (process_rank == root_rank) {
                std::vector<int> ranks;
                for (auto [rank, index]: global_ranks) {
                    ranks.push_back(rank);
                }
                SA = slow_suffixarray(ranks);
                global_ranks.clear();
                for (uint i = 0; i < SA.size(); i++) {
                    int global_index = map_back(SA[i], total_chars);
                    global_ranks.emplace_back(i + 1, global_index);
                }
                // redistribute result and continue in distributed fashion
            }
            // print_vector(global_ranks);

            auto by_second = [&](RankIndex const& a, RankIndex const& b) {
                return a.index < b.index;
            };
            std::sort(global_ranks.begin(), global_ranks.end(), by_second);

            local_ranks.clear();
            // send to each processor the size of its local_string
            // only relevant to root
            std::vector<int> send_cnts(num_processes);
            for (uint i = 0; i < global_ranks.size(); i++) {
                size_t index = global_ranks[i].index;
                // check to which process this index belongs
                for (int j = 0; j < num_processes; j++) {
                    if (index < chars_at_proc[j]) {
                        send_cnts[j]++;
                        break;
                    }
                    index -= chars_at_proc[j];
                }
            }

            // removing n1 padding
            if (process_rank == root_rank) {
                KASSERT(std::accumulate(send_cnts.begin(), send_cnts.end(), 0) + n1_padding
                        == (int)global_ranks.size());
            }

            comm.scatterv(send_buf(global_ranks),
                          recv_buf<resize_to_fit>(local_ranks),
                          send_counts(send_cnts));
            // check_expected_size(num_samples, local_ranks, comm);
            // TODO: recursively call SA
            if (DBG)
                print_concatenated(local_ranks, comm, "local_ranks");
        }
    }

    std::vector<MergeSamples> merge_samples;
    merge_samples.reserve(chars_at_proc[process_rank]);

    // processor i sends first two ranks to processor i - 1
    std::vector<RankIndex> recv_rank = mpi_util::shift_left(local_ranks, 2, comm);
    if (process_rank < last_process) {
        local_ranks.insert(local_ranks.end(), recv_rank.begin(), recv_rank.end());
    }

    // add two paddings with rank 0
    if (process_rank == last_process) {
        local_ranks.push_back(RankIndex(0, total_chars));
        local_ranks.push_back(RankIndex(0, total_chars));
    }

    if (DBG)
        print_concatenated(local_ranks, comm, "local ranks");

    int pos = 0;
    int rank1_offset[3] = {0, 0, 0};
    int rank2_offset[3] = {1, 1, 1};
    // for each index in local string
    for (int i = 0; i < (int)chars_at_proc[process_rank]; i++) {
        // convert global index to local index
        // for mod 0, find next mod 1 position
        // for mod 1,2 find next mod 1, 2 position
        while (i > local_ranks[pos].index - chars_before[process_rank]) {
            pos++;
            KASSERT(pos < (int)local_ranks.size());
        }

        int local_index = i;
        int global_index = local_index + chars_before[process_rank];
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

    if (DBG)
        print_concatenated(merge_samples, comm, "merge samples");

    mpi::sort(merge_samples, std::less<>{}, comm);

    if (DBG)
        print_concatenated(merge_samples, comm, "merge samples");

    std::vector<int> local_SA;
    // remove 2 paddings
    local_SA.reserve(merge_samples.size());
    for (uint i = 0; i < merge_samples.size(); i++) {
        local_SA.push_back(merge_samples[i].index);
    }

    if (DBG) {
        print_concatenated(local_string, comm);
        print_concatenated(local_SA, comm);
    }

    // revert changes made to local string by left shift
    if (process_rank < last_process) {
        local_string.pop_back();
        local_string.pop_back();
    }

    return local_SA;
}

} // namespace dsss
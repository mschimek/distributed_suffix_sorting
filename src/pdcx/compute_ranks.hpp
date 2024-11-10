#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "pdcx/sample_string.hpp"


//******* Start Phase 2: Construct Ranks  ********

namespace dsss::dcx {

using namespace kamping;

template <typename char_type, typename index_type, typename DC>
struct DCRankIndex {
    index_type rank;
    index_type index;
    bool unique;

    static bool cmp_by_index(const DCRankIndex& a, const DCRankIndex& b) {
        return a.index < b.index;
    }
    static bool cmp_by_rank(const DCRankIndex& a, const DCRankIndex& b) { return a.rank < b.rank; }
    static bool cmp_mod_div(const DCRankIndex& a, const DCRankIndex& b) {
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

template <typename char_type, typename index_type, typename DC>
struct LexicographicRankPhase {
    using SampleString = DCSampleString<char_type, index_type, DC>;
    using RankIndex = DCRankIndex<char_type, index_type, DC>;

    Communicator<>& comm;

    LexicographicRankPhase(Communicator<>& _comm) : comm(_comm) {}


    // shift one sample left be able to compute rank of last element
    void shift_samples_left(std::vector<SampleString>& local_samples) const {
        // adds a dummy sample for last process
        KASSERT(local_samples.size() >= 1u);
        SampleString recv_sample = mpi_util::shift_left(local_samples.front(), comm);
        local_samples.push_back(recv_sample);
        local_samples.shrink_to_fit();
    }

    std::vector<RankIndex>
    compute_lexicographic_ranks(std::vector<SampleString>& local_samples) const {
        std::vector<RankIndex> local_ranks;

        // exclude sample from process i + 1
        uint64_t num_ranks = local_samples.size() - 1;
        local_ranks.reserve(num_ranks);

        // compute local ranks
        uint64_t prev_rank = 0;
        for (uint64_t i = 0; i < num_ranks; i++) {
            KASSERT(i + 1 < local_samples.size());
            local_ranks.emplace_back(index_type(prev_rank), local_samples[i].index, false);
            uint64_t changed = local_samples[i].letters != local_samples[i + 1].letters ? 1 : 0;
            prev_rank += changed;
        }

        // shift ranks by 1 + prefix sum
        uint64_t ranks_before = mpi_util::ex_prefix_sum(prev_rank, comm);
        std::for_each(local_ranks.begin(), local_ranks.end(), [&](RankIndex& x) {
            x.rank += index_type(1 + ranks_before);
        });
        return local_ranks;
    }

    void flag_unique_ranks(std::vector<RankIndex>& local_ranks) const {
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

    // create lexicographic ranks and flag unique ranks
    // sideeffect: shifts one sample from next PE to local_string
    std::vector<RankIndex>
    create_lexicographic_ranks(std::vector<SampleString>& local_samples) const {
        shift_samples_left(local_samples);
        std::vector<RankIndex> local_ranks = compute_lexicographic_ranks(local_samples);
        flag_unique_ranks(local_ranks);
        return local_ranks;
    }
};

} // namespace dsss::dcx
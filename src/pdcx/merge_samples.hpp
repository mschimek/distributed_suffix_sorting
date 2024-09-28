#pragma once

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/shift.hpp"
#include "pdcx/compute_ranks.hpp"
#include "sorters/sorting_wrapper.hpp"

namespace dsss::dcx {

using namespace kamping;

//******* Start Phase 4: Merge Suffixes  ********

template <typename char_type, typename index_type, typename DC>
struct DCMergeSamples {
    DCMergeSamples() {
        index = 0;
        chars.fill(0);
        ranks.fill(0);
    }
    DCMergeSamples(std::array<char_type, DC::X - 1> _chars,
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


    // materialize all substrings of length X - 1 and corresponding D ranks
    std::vector<MergeSamples> construct_merge_samples(std::vector<char_type>& local_string,
                                                      std::vector<RankIndex>& local_ranks,
                                                      uint64_t chars_before,
                                                      uint64_t chars_at_proc) const {
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

    // sort merge samples using substrings and rank information
    void sort_merge_samples(std::vector<MergeSamples>& merge_samples,
                            mpi::SortingWrapper& atomic_sorter) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_04_sort_merge_samples");
        atomic_sorter.sort(merge_samples, std::less<>{});
        timer.stop();
    }

    // extract SA from merge samples
    std::vector<index_type> extract_SA(std::vector<MergeSamples>& merge_samples) const {
        auto get_index = [](MergeSamples& m) { return m.index; };
        std::vector<index_type> local_SA =
            extract_attribute<MergeSamples, index_type>(merge_samples, get_index);
        return local_SA;
    }
};


} // namespace dsss::dcx
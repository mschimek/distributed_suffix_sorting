#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/shift.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/statistics.hpp"
#include "sorters/sample_sort_strings.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"


namespace dsss::dcx {

using namespace kamping;

//******* Phase 1: Construct Samples  ********

// substring sampled by a difference cover sample
template <typename char_type, typename index_type, typename DC>
struct DCSampleString {
    // for string sorter
    using CharType = char_type;
    const CharType* cbegin_chars() const { return letters.data(); }
    const CharType* cend_chars() const { return letters.data() + DC::X; }
    std::string get_string() { return to_string(); }

    DCSampleString() {
        letters.fill(0);
        index = 0;
    }
    DCSampleString(std::array<char_type, DC::X>&& _letters, index_type _index)
        : letters(_letters),
          index(_index) {}

    bool operator<(const DCSampleString& other) const {
        for (uint i = 0; i < DC::X; i++) {
            if (letters[i] != other.letters[i]) {
                return letters[i] < other.letters[i];
            }
        }
        return index < other.index;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "(" << (uint64_t)letters[0];
        for (uint i = 1; i < DC::X; i++) {
            ss << " " << (uint64_t)letters[i];
        }
        ss << ") " << index;
        return ss.str();
    }

    std::array<char_type, DC::X> letters;
    index_type index;
};


template <typename char_type, typename index_type, typename DC>
struct SampleStringPhase {
    using SampleString = DCSampleString<char_type, index_type, DC>;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;

    SampleStringPhase(Communicator<>& _comm) : comm(_comm) {}

    // shift characters left to compute overlapping samples
    void shift_chars_left(std::vector<char_type>& local_string) const {
        mpi_util::shift_entries_left(local_string, X - 1, comm);
        local_string.shrink_to_fit();
    }

    // sample substrings of length X at difference cover samples
    std::vector<SampleString> compute_sample_strings(std::vector<char_type>& local_string,
                                                     uint64_t chars_before) const {
        std::vector<SampleString> local_samples;
        uint64_t size_estimate = ((local_string.size() + X - 1) / X) * D;
        local_samples.reserve(size_estimate);

        uint64_t offset = chars_before % X;
        for (uint64_t i = 0; i + X - 1 < local_string.size(); i++) {
            uint64_t m = (i + offset) % X;
            if (is_in_dc<DC>(m)) {
                index_type index = index_type(chars_before + i);
                std::array<char_type, X> letters;
                for (uint k = 0; k < X; k++) {
                    letters[k] = local_string[i + k];
                }
                local_samples.push_back(SampleString(std::move(letters), index));
            }
        }
        // last process adds a dummy sample if remainder of some differrence cover element aligns
        // with the string length

        return local_samples;
    }

    void atomic_sort_samples(std::vector<SampleString>& local_samples,
                             mpi::SortingWrapper& atomic_sorter) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_01_sort_local_samples");
        atomic_sorter.sort(local_samples, std::less<>{});
        timer.stop();
        local_samples.shrink_to_fit();
    }

    void string_sort_samples(std::vector<SampleString>& local_samples,
                             dsss::SeqStringSorterWrapper& string_sorter,
                             mpi::SampleSortConfig &config) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_01_sort_local_samples");
        mpi::sample_sort_strings(local_samples, comm, string_sorter, config);
        timer.stop();
        local_samples.shrink_to_fit();
    }
};

} // namespace dsss::dcx
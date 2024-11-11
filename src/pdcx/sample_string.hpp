#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/shift.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/redistribute.hpp"
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
    const CharType* cend_chars() const { return letters.data() + DC::X + 1; }
    std::string get_string() { return to_string(); }

    // X chars and one 0-character
    using SampleStringLetters = std::array<char_type, DC::X + 1>;


    DCSampleString() {
        letters.fill(0);
        index = 0;
    }
    DCSampleString(SampleStringLetters&& _letters, index_type _index)
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

    SampleStringLetters letters;
    index_type index;
};

template <typename char_type, typename index_type, typename DC>
struct SampleStringPhase {
    using SampleString = DCSampleString<char_type, index_type, DC>;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;
    PDCXConfig& config;
    mpi::SortingWrapper& atomic_sorter;
    dsss::SeqStringSorterWrapper& string_sorter;
    uint64_t local_sample_size;

    SampleStringPhase(Communicator<>& _comm,
                      PDCXConfig& _config,
                      mpi::SortingWrapper& _atomic_sorter,
                      dsss::SeqStringSorterWrapper& _string_sorter)
        : comm(_comm),
          config(_config),
          atomic_sorter(_atomic_sorter),
          string_sorter(_string_sorter) {}


    // shift characters left to compute overlapping samples
    void shift_chars_left(std::vector<char_type>& local_string) const {
        mpi_util::shift_entries_left(local_string, X - 1, comm);
        local_string.shrink_to_fit();
    }

    // materialize a difference cover sample
    SampleString::SampleStringLetters materialize_sample(std::vector<char_type>& local_string,
                                                         uint64_t i) const {
        std::array<char_type, X + 1> letters;
        for (uint k = 0; k < X; k++) {
            letters[k] = local_string[i + k];
        }
        letters.back() = 0; // 0-terminated string
        return letters;
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
                std::array<char_type, X + 1> letters = materialize_sample(local_string, i);
                local_samples.push_back(SampleString(std::move(letters), index));
            }
        }
        // last process adds a dummy sample if remainder of some differrence cover element aligns
        // with the string length

        return local_samples;
    }

    // sort samples using an atomic sorter
    void atomic_sort_samples(std::vector<SampleString>& local_samples) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_01_sort_local_samples");
        atomic_sorter.sort(local_samples, std::less<>{});
        timer.stop();
        local_samples.shrink_to_fit();
    }

    // sort samples using a string sorter
    void string_sort_samples(std::vector<SampleString>& local_samples) const {
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_01_sort_local_samples");
        mpi::sample_sort_strings(local_samples, comm, string_sorter, config.sample_sort_config);
        timer.stop();
        local_samples.shrink_to_fit();
    }

    uint64_t get_local_sample_size() const { return local_sample_size; }

    // create and sort difference cover samples
    // sideeffect: shifts characters from next PE to localstring
    std::vector<SampleString> sorted_dc_samples(std::vector<char_type>& local_string,
                                                uint64_t chars_before) {
        shift_chars_left(local_string);
        std::vector<SampleString> local_samples =
            compute_sample_strings(local_string, chars_before);

        // number of dc-samples in local string
        local_sample_size = local_samples.size();
        if (config.use_string_sort) {
            string_sort_samples(local_samples);
        } else {
            atomic_sort_samples(local_samples);
        }

        bool redist_samples = redistribute_if_imbalanced(local_samples, config.min_imbalance, comm);
        get_stats_instance().redistribute_samples.push_back(redist_samples);
        return local_samples;
    }
};

} // namespace dsss::dcx
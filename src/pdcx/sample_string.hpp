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
#include "pdcx/packing.hpp"
#include "pdcx/redistribute.hpp"
#include "pdcx/statistics.hpp"
#include "sorters/sample_sort_strings.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/char_container.hpp"
#include "util/printing.hpp"

namespace dsss::dcx {

using namespace kamping;

//******* Phase 1: Construct Samples  ********

// substring sampled by a difference cover sample
template <typename char_type,
          typename index_type,
          typename DC,
          typename CharContainer = CharArray<char_type, DC::X + 1>>
struct DCSampleString {
    // for string sorter
    using CharType = char_type;
    const CharType* cbegin_chars() const { return chars.cbegin_chars(); }
    const CharType* cend_chars() const { return chars.cend_chars(); }
    std::string get_string() { return to_string(); }

    // to separately send chars, ranks and index
    using CharContainerType = CharContainer;
    struct NonCharData {
        index_type index;
    };

    NonCharData get_non_char_data() const { return {index}; }


    // X chars and one 0-character
    using SampleStringLetters = std::array<char_type, DC::X + 1>;

    DCSampleString() : chars(CharContainer()), index(0) {}

    DCSampleString(CharContainer&& _chars, index_type _index) : chars(_chars), index(_index) {}

    DCSampleString(CharContainer&& _chars, NonCharData&& _non_chars)
        : chars(_chars),
          index(_non_chars.index) {}

    bool operator<(const DCSampleString& other) const { return chars < other.chars; }

    std::string to_string() const {
        std::stringstream ss;
        ss << "(" << chars.to_string();
        ss << ") " << index;
        return ss.str();
    }

    std::array<char_type, DC::X + 1> get_array_letters() const {
        std::array<char_type, DC::X + 1> array;
        for (uint32_t i = 0; i < DC::X; i++) {
            array[i] = chars.at(i);
        }
        array.back() = 0;
        return array;
    }

    CharContainer chars;
    index_type index;
};

template <typename char_type,
          typename index_type,
          typename DC,
          typename CharContainer = CharArray<char_type, DC::X + 1>>
struct SampleStringPhase {
    using SampleString = DCSampleString<char_type, index_type, DC, CharContainer>;

    static constexpr uint32_t X = DC::X;
    static constexpr uint32_t D = DC::D;

    Communicator<>& comm;
    PDCXConfig const& config;
    PDCXLengthInfo& info;
    mpi::SortingWrapper& atomic_sorter;
    dsss::SeqStringSorterWrapper& string_sorter;

    SampleStringPhase(Communicator<>& _comm,
                      PDCXConfig const& _config,
                      PDCXLengthInfo& _info,
                      mpi::SortingWrapper& _atomic_sorter,
                      dsss::SeqStringSorterWrapper& _string_sorter)
        : comm(_comm),
          config(_config),
          info(_info),
          atomic_sorter(_atomic_sorter),
          string_sorter(_string_sorter) {}


    // add padding to local string
    void add_padding(std::vector<char_type>& local_data, uint64_t padding_length) {
        char_type padding = char_type(0);
        if (comm.rank() == comm.size() - 1) {
            std::fill_n(std::back_inserter(local_data), padding_length, padding);
        }
    }


    // shift characters left to compute overlapping samples
    void shift_chars_left(std::vector<char_type>& local_string, double packing_ratio = 1) const {
        // if we can pack 2 chars into one type, we need more 2x padding
        double count = packing_ratio * X - 1;
        mpi_util::shift_entries_left(local_string, count, comm);
        local_string.shrink_to_fit();
    }

    // materialize a difference cover sample
    CharContainer materialize_sample(std::vector<char_type>& local_string,
                                     uint64_t i,
                                     double packing_ratio = 1) const {
        return CharContainer(local_string.begin() + i,
                             local_string.begin() + i + X * packing_ratio);
    }

    std::array<char_type, X + 1> materialize_splitter(std::vector<char_type>& local_string,
                                                      uint64_t i) const {
        std::array<char_type, X + 1> chars;
        std::copy(local_string.begin() + i, local_string.begin() + i + X, chars.begin());
        chars.back() = 0;
        return chars;
    }

    // sample substrings of length X at difference cover samples
    std::vector<SampleString> compute_sample_strings(std::vector<char_type>& local_string,
                                                     auto materialize_sample) const {
        std::vector<SampleString> local_samples;
        local_samples.reserve(info.local_sample_size);

        uint64_t offset = info.chars_before % X;
        for (uint64_t i = 0; i < info.local_chars_with_dummy; i++) {
            uint64_t m = (i + offset) % X;
            if (is_in_dc<DC>(m)) {
                index_type index = index_type(info.chars_before + i);
                CharContainer chars = materialize_sample(local_string, i);
                local_samples.push_back(SampleString(std::move(chars), index));
            }
        }
        KASSERT(local_samples.size() == info.local_sample_size);
        // last process adds a dummy sample if remainder of some differrence cover element aligns
        // with the string length

        return local_samples;
    }

    void tie_break_samples(std::vector<SampleString>& local_samples) const {
        // assuming that chars are not split by sample sorter
        auto cmp_index = [](const SampleString& a, const SampleString& b) {
            return a.index < b.index;
        };

        // sort each segement with the same chars by index
        int64_t start = 0;
        int64_t end = 0;
        for (int64_t i = 0; i < (int64_t)local_samples.size() - 1; i++) {
            bool segment_ended = local_samples[i].chars != local_samples[i + 1].chars;
            if (segment_ended) {
                end = i + 1;
                ips4o::sort(local_samples.begin() + start, local_samples.begin() + end, cmp_index);
                start = end;
            }
        }
        end = local_samples.size();
        if (local_samples.size() > 1) {
            ips4o::sort(local_samples.begin() + start, local_samples.end(), cmp_index);
        }
    }

    // sort samples using an atomic sorter
    void atomic_sort_samples(std::vector<SampleString>& local_samples) const {
        atomic_sorter.sort(local_samples, std::less<>{});
    }

    // sort samples using a string sorter
    void string_sort_samples(std::vector<SampleString>& local_samples) const {
        // dummy tie break
        auto tie_break = [&](std::vector<SampleString>& merge_samples) { return; };
        sample_sort_strings_tie_break(local_samples,
                                      comm,
                                      string_sorter,
                                      tie_break,
                                      config.sample_sort_config);
    }

    void string_sort_tie_break_samples(std::vector<SampleString>& local_samples) const {
        // break ties by comparing index
        auto tie_break = [&](std::vector<SampleString>& local_samples) {
            tie_break_samples(local_samples);
        };
        // use full comparison function to determine splitters
        auto cmp = [&](const SampleString& a, const SampleString& b) {
            if (a.chars != b.chars) {
                return a.chars < b.chars;
            }
            return a.index < b.index;
        };
        sample_sort_strings_tie_break(local_samples,
                                      comm,
                                      string_sorter,
                                      tie_break,
                                      cmp,
                                      config.sample_sort_config);
    }

    void sort_samples(std::vector<SampleString>& local_samples, bool use_packing) const {
        const bool use_string_sort = config.use_string_sort && !use_packing;
        const bool use_tie_break = config.use_string_sort_tie_breaking_phase1;
        auto& timer = measurements::timer();
        timer.synchronize_and_start("phase_01_sort_local_samples");
        if (use_string_sort && !use_tie_break) {
            string_sort_samples(local_samples);
        } else if (use_string_sort && use_tie_break) {
            string_sort_tie_break_samples(local_samples);
        } else {
            atomic_sort_samples(local_samples);
        }
        timer.stop();
        local_samples.shrink_to_fit();
    }

    void make_padding_and_shifts(std::vector<char_type>& local_string,
                                 double char_packing_ratio = 1) {
        // add padding to local string
        const uint64_t padding_length = char_packing_ratio * X;
        add_padding(local_string, padding_length);

        // shift necessary chars from right PE
        shift_chars_left(local_string, char_packing_ratio);
    }

    // create and sort difference cover samples
    // sideeffect: shifts characters from next PE to localstring
    std::vector<SampleString> sorted_dc_samples(std::vector<char_type>& local_string,
                                                bool use_packing = false) {
        // materialize samples
        std::vector<SampleString> local_samples;
        double packing_ratio = use_packing ? config.packing_ratio : 1;
        local_samples = compute_sample_strings(local_string, [&](auto& local_string, auto i) {
            return materialize_sample(local_string, i, packing_ratio);
        });

        // sort samples
        sort_samples(local_samples, use_packing);

        bool redist_samples = redistribute_if_imbalanced(local_samples, config.min_imbalance, comm);
        get_stats_instance().redistribute_samples.push_back(redist_samples);
        get_local_stats_instance().redistribute_samples.push_back(redist_samples);
        return local_samples;
    }
};

} // namespace dsss::dcx

#pragma once

#include <cstdint>
#include <vector>

#include "strings/string_set.hpp"
#include "strings/multikey_quicksort.hpp"
#include "strings/radix_sort.hpp"
#include "tlx/sort/strings/string_ptr.hpp"
#include "tlx/sort/strings/string_set.hpp"

namespace dsss {
enum SeqStringSorter { MultiKeyQSort, RadixSortCI2, RadixSortCI3 };
static std::vector<std::string> string_sorter_names = {
    "multi_key_qsort", "radix_sort_ci2", "radix_sort_ci3"};

struct SeqStringSorterWrapper {
    // we only sort strings with less than 255 characters in DCX
    using LcpType = uint8_t;

    SeqStringSorterWrapper() : sorter(MultiKeyQSort), memory(0) {}

    // string sorter to be used
    void set_sorter(SeqStringSorter new_sorter) { sorter = new_sorter; }

    // memory limit
    void set_memory(uint64_t new_memory) { memory = new_memory; }

    template <typename StringPtr>
    void call_string_sorter(StringPtr& string_ptr) {
        switch (sorter) {
            case MultiKeyQSort:
                tlx::sort_strings_detail::multikey_quicksort(string_ptr, depth, memory);
                break;
            case RadixSortCI2:
                tlx::sort_strings_detail::radixsort_CI2(string_ptr, depth, memory);
                break;
            case RadixSortCI3:
                tlx::sort_strings_detail::radixsort_CI3(string_ptr, depth, memory);
                break;
            default:
                tlx::sort_strings_detail::multikey_quicksort(string_ptr, depth, memory);
                break;
        }
    }
    template <typename DataType>
    void sort(std::vector<DataType>& local_data) {
        using StringSet = GeneralStringSet<DataType>;
        StringSet string_set(local_data.data(), local_data.data() + local_data.size());
        tlx::sort_strings_detail::StringPtr<StringSet> string_ptr(string_set);
        call_string_sorter(string_ptr);
    }

    template <typename DataType>
    void sort_with_lcps(std::vector<DataType>& local_data, std::vector<LcpType>& lcps) {
        using StringSet = GeneralStringSet<DataType>;
        StringSet string_set(local_data.data(), local_data.data() + local_data.size());
        tlx::sort_strings_detail::StringLcpPtr<StringSet, LcpType> string_ptr(string_set,
                                                                              lcps.data());
        call_string_sorter(string_ptr);
    }

    SeqStringSorter sorter;
    uint64_t memory;
    uint64_t depth = 0;
};

} // namespace dsss

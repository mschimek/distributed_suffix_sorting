#pragma once

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace dsss {

template <typename char_type, uint64_t X>
bool cmp_index_substring(std::vector<char_type>& str,
                   uint64_t local_index,
                   std::array<char_type, X>& sub_str) {
    for (uint64_t j = 0; j < X; j++) {
        KASSERT(local_index + j < str.size());
        char_type c = str[local_index + j];
        if (c != sub_str[j]) {
            return c < sub_str[j];
        }
    }
    return false;
};


template <typename char_type>
bool cmp_substrings(std::vector<char_type>& arr, int a, int b) {
    int m = arr.size();
    int j = 0;
    while (a + j < m && b + j < m && arr[a + j] == arr[b + j]) {
        j++;
    }
    if (a + j == m)
        return true; // substring "a" ended first
    if (b + j == m)
        return false; // substring "b" ended first
    return arr[a + j] < arr[b + j];
}

template <typename char_type, typename index_type>
std::vector<index_type> slow_suffixarray(std::vector<char_type>& arr) {
    std::vector<index_type> sa(arr.size());
    std::iota(sa.begin(), sa.end(), 0);
    std::sort(sa.begin(), sa.end(), [&arr](int a, int b) { return cmp_substrings(arr, a, b); });
    return sa;
}

template <typename char_type>
void print_substrings(std::vector<char_type>& arr) {
    for (uint i = 0; i < arr.size(); i++) {
        std::cout << i << ": ";
        for (uint j = i; j < arr.size(); j++) {
            std::cout << arr[j] << "";
        }
        std::cout << "\n";
    }
}

template <typename Combined, typename Extracted>
std::vector<Extracted> extract_attribute(std::vector<Combined>& combined,
                                         std::function<Extracted(Combined&)> get_attribute) {
    std::vector<Extracted> extracted;
    extracted.reserve(combined.size());
    for (Combined& c: combined) {
        extracted.push_back(get_attribute(c));
    }
    return extracted;
}

} // namespace dsss
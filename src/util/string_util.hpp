#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "strings/lcp_type.hpp"

namespace dsss {

// compare strings by scanning
template <typename String>
static inline int64_t string_cmp(String& s1, String& s2) {
    KASSERT(std::distance(s1.cbegin_chars(), s1.cend_chars())
            == std::distance(s2.cbegin_chars(), s2.cend_chars()));
    auto it1 = s1.cbegin_chars();
    auto it2 = s2.cbegin_chars();
    auto end = s1.cend_chars();
    end--;

    while (it1 < end && *it1 == *it2) {
        it1++;
        it2++;
    }
    return static_cast<int64_t>(*it1) - static_cast<int64_t>(*it2);
}

// compare strings by scanning. Start at given lcp, which also returns the final lcp.
template <typename String, typename lcp_type>
static inline int64_t string_cmp(String& s1, String& s2, lcp_type& lcp) {
    KASSERT(std::distance(s1.cbegin_chars(), s1.cend_chars())
            == std::distance(s2.cbegin_chars(), s2.cend_chars()));
    auto it1 = s1.cbegin_chars() + lcp;
    auto it2 = s2.cbegin_chars() + lcp;
    auto end = s1.cend_chars();
    end--;

    while (it1 < end && *it1 == *it2) {
        it1++;
        it2++;
        lcp++;
    }
    return static_cast<int64_t>(*it1) - static_cast<int64_t>(*it2);
}


template <typename String>
static inline LcpType compute_lcp(String& s1, String& s2) {
    LcpType lcp = 0;
    string_cmp(s1, s2, lcp);
    return lcp;
}

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

template <typename char_type, uint64_t X>
bool cmp_index_substring(std::vector<char_type>& str,
                         uint64_t local_index,
                         std::array<char_type, X>& sub_str,
                         uint64_t len) {
    for (uint64_t j = 0; j < len; j++) {
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
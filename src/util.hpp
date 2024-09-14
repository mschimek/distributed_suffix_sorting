#pragma once

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace dsss {

bool cmp_substrings(std::vector<int>& arr, int a, int b) {
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

std::vector<int> slow_suffixarray(std::vector<int>& arr) {
    std::vector<int> sa(arr.size());
    std::iota(sa.begin(), sa.end(), 0);
    std::sort(sa.begin(), sa.end(), [&arr](int a, int b) { return cmp_substrings(arr, a, b); });
    return sa;
}

void print_substrings(std::vector<int>& arr) {
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
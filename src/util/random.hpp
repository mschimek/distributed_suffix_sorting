#pragma once

#include <random>
#include <vector>

namespace dsss::random {
template <typename T>
std::vector<T> generate_random_data(uint64_t n, uint64_t alphabet_size, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(1, alphabet_size);
    std::vector<T> v(n);
    for (uint64_t i = 0; i < n; i++) {
        v[i] = dist(rng);
    }
    return v;
}


} // namespace dsss::random
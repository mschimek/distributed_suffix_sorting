#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dsss::dcx {

template <typename char_type, size_t N>
struct CharPacking {
    CharPacking(uint64_t largest_char) {
        char_bits = 8 * sizeof(char_type);
        packed_char_bits = std::ceil(std::log2(largest_char + 1)); // + 1 for padding
        char_packing_ratio = char_bits / packed_char_bits;
    }

    std::array<char_type, N> materialize_packed_sample(std::vector<char_type>& local_string,
                                                       uint64_t i) const {
        std::array<char_type, N> letters;
        letters.fill(0);
        uint64_t char_pos = i;
        for (uint64_t k = 0; k < N - 1; k++) {
            for (uint64_t j = 0; j < char_packing_ratio; j++) {
                KASSERT(char_pos < local_string.size());
                letters[k] = (letters[k] << packed_char_bits) | (local_string[char_pos]);
                char_pos++;
            }
        }
        letters.back() = 0; // 0-terminated string
        return letters;
    }

    uint64_t char_bits;
    uint64_t packed_char_bits;
    uint64_t char_packing_ratio;
};
}; // namespace dsss::dcx
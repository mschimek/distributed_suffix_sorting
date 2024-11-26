#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>

// An array container for a 0-terminated char array.
template <typename char_type, size_t N>
struct CharArray {
    CharArray() { chars.fill(0); }

    template <typename CharIterator>
    CharArray(CharIterator begin, CharIterator end) {
        std::copy(begin, end, chars.begin());
        chars.back() = 0;
    }

    char_type at(uint64_t i) const { return chars[i]; }

    bool operator<(const CharArray& other) const { return chars < other.chars; }
    bool operator==(const CharArray& other) const { return chars == other.chars; }
    bool operator!=(const CharArray& other) const { return chars != other.chars; }

    const char_type* cbegin_chars() const { return chars.data(); }
    const char_type* cend_chars() const { return chars.data() + chars.size(); }

    std::string to_string() const {
        std::stringstream ss;
        ss << chars[0];
        for (uint i = 1; i < N; i++) {
            ss << " " << chars[i];
        }
        return ss.str();
    }

    std::array<char_type, N> chars;
};
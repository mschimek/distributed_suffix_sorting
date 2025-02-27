#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "kassert/kassert.hpp"
#include "tlx/define/likely.hpp"
#include "util/printing.hpp"

// An array container for a 0-terminated char array.
template <typename char_type, size_t N>
struct CharArray {
    CharArray() { chars.fill(0); }

    template <typename CharIterator>
    CharArray(CharIterator begin, CharIterator end) {
        chars.fill(0);
        std::copy(begin, end, chars.begin());
        chars.back() = 0;
    }

    char_type at(uint64_t i) const { return chars[i]; }

    bool operator<(const CharArray& other) const { return chars < other.chars; }
    bool operator==(const CharArray& other) const { return chars == other.chars; }
    bool operator!=(const CharArray& other) const { return chars != other.chars; }

    const char_type* cbegin_chars() const { return chars.data(); }
    const char_type* cend_chars() const { return chars.data() + chars.size(); }
    char_type* begin() { return chars.data(); }
    char_type* end() { return chars.data() + chars.size(); }
    size_t size() const { return chars.size(); }

    std::string to_string() const {
        std::stringstream ss;
        ss << (uint64_t)chars[0];
        for (uint i = 1; i < N; i++) {
            ss << " " << (uint64_t)chars[i];
        }
        return ss.str();
    }

    static constexpr bool IS_PACKED = false;
    std::array<char_type, N> chars;
};

template <typename char_type, size_t BitsChar, typename IntType = uint64_t>
struct PackedInteger {
    PackedInteger() : chars(0) {}

    static constexpr IntType char_mask = (IntType(1) << BitsChar) - 1;
    static constexpr IntType last_bit = 8 * sizeof(IntType) - BitsChar;

    template <typename CharIterator>
    PackedInteger(CharIterator begin, CharIterator end) {
        chars = 0;
        auto it = begin;
        IntType cnt = 0;
        // pack integer from left to right to be able to make lexicographic comparisons with integer
        while (it != end) {
            IntType bits = *it & char_mask;
            KASSERT(last_bit >= cnt);
            IntType shifted_bits = bits << (last_bit - cnt);
            chars |= shifted_bits;
            it++;
            cnt += BitsChar;
        }
    }

    char_type at(uint64_t i) const {
        KASSERT(8 * sizeof(IntType) >= (i + 1) * BitsChar);
        int shift_len = 8 * sizeof(IntType) - (i + 1) * BitsChar;
        return (chars >> shift_len) & char_mask;
    }

    bool operator<(const PackedInteger& other) const { return chars < other.chars; }
    bool operator==(const PackedInteger& other) const { return chars == other.chars; }
    bool operator!=(const PackedInteger& other) const { return chars != other.chars; }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&chars;
    }
    const char_type* cend_chars() const { return (const char_type*)&chars; }

    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&chars;
    }
    char_type* end() const { return (char_type*)&chars; }
    size_t size() const { return 0; }


    std::string to_string() const { return std::to_string(chars); }

    static constexpr bool IS_PACKED = true;
    IntType chars;
};

// does not encode padding character 0
template <typename char_type, size_t BitsChar, typename IntType = uint64_t>
struct PackedIntegerPadding {
    PackedIntegerPadding() : chars(0) {}

    static constexpr IntType char_mask = (IntType(1) << BitsChar) - 1;
    static constexpr IntType max_chars = 8 * sizeof(IntType) / BitsChar;
    static constexpr IntType last_bit = 8 * sizeof(IntType) - BitsChar;


    template <typename CharIterator>
    PackedIntegerPadding(CharIterator begin, CharIterator end) {
        chars = 0;
        padding_len = 0;
        auto it = begin;
        IntType cnt = 0;

        // pack integer from left to right to be able to make lexicographic comparisons with integer
        while (it != end) {
            // read chars until first 0-character
            if (*it == 0) {
                padding_len = max_chars - (cnt / BitsChar);
                break;
            }
            // let non padding chars start at 0
            IntType bits = (*it - 1) & char_mask;
            KASSERT(last_bit >= cnt);
            IntType shifted_bits = bits << (last_bit - cnt);
            chars |= shifted_bits;
            it++;
            cnt += BitsChar;
        }
    }

    char_type at(uint64_t i) const {
        if (i >= max_chars - padding_len) {
            return 0;
        } else {
            KASSERT(8 * sizeof(IntType) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType) - (i + 1) * BitsChar;
            uint64_t c = (chars >> shift_len) & char_mask;
            // chars normally start at 1
            c++;
            return c;
        }
    }

    bool operator<(const PackedIntegerPadding& other) const {
        // TODO test branch hint
        // if (TLX_LIKELY(padding_len == other.padding_len)) {
        if (padding_len == other.padding_len) {
            return chars < other.chars;
        } else {
            // compare until padding char
            uint8_t max_len = std::max(padding_len, other.padding_len);
            uint64_t clear_mask = (IntType(1) << (max_len * BitsChar)) - 1;
            uint64_t a = chars & (~clear_mask);
            uint64_t b = other.chars & (~clear_mask);
            if (a != b) {
                return a < b;
            } else {
                return padding_len > other.padding_len;
            }
        }
    }
    bool operator==(const PackedIntegerPadding& other) const {
        return chars == other.chars && padding_len == other.padding_len;
    }
    bool operator!=(const PackedIntegerPadding& other) const {
        return chars != other.chars || padding_len != other.padding_len;
    }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&chars;
    }
    const char_type* cend_chars() const { return (const char_type*)&chars; }
    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&chars;
    }
    char_type* end() const { return (char_type*)&chars; }
    size_t size() const { return 0; }

    std::string to_string() const {
        return std::to_string(chars) + "-" + std::to_string(padding_len);
    }

    static constexpr bool IS_PACKED = true;
    IntType chars;
    uint8_t padding_len;
};

template <typename char_type,
          size_t BitsChar,
          typename IntType1 = uint64_t,
          typename IntType2 = uint32_t>
struct DoublePackedInteger {
    DoublePackedInteger() : chars1(0), chars2(0) {}

    static constexpr uint64_t char_mask1 = (IntType1(1) << BitsChar) - 1;
    static constexpr uint64_t char_mask2 = (IntType2(1) << BitsChar) - 1;

    static constexpr uint64_t last_bit1 = 8 * sizeof(IntType1) - BitsChar;
    static constexpr uint64_t last_bit2 = 8 * sizeof(IntType2) - BitsChar;

    static constexpr uint64_t max_chars1 = 8 * sizeof(IntType1) / BitsChar;
    static constexpr uint64_t max_chars2 = 8 * sizeof(IntType2) / BitsChar;


    template <typename CharIterator>
    DoublePackedInteger(CharIterator begin, CharIterator end) {
        chars1 = 0;
        chars2 = 0;
        auto it = begin;
        uint64_t cnt = 0;

        // pack integer from left to right to be able to make lexicographic comparisons with integer

        // pack first integer
        while (it != end && cnt < max_chars1 * BitsChar) {
            IntType1 bits = *it & char_mask1;
            KASSERT(last_bit1 >= cnt);
            IntType1 shifted_bits = bits << (last_bit1 - cnt);
            chars1 |= shifted_bits;
            it++;
            cnt += BitsChar;
        }

        // pack second integer
        cnt = 0;
        while (it != end) {
            IntType2 bits = *it & char_mask2;
            KASSERT(last_bit2 >= cnt);
            IntType2 shifted_bits = bits << (last_bit2 - cnt);
            chars2 |= shifted_bits;
            it++;
            cnt += BitsChar;
        }
    }

    char_type at(uint64_t i) const {
        if (i < max_chars1) {
            KASSERT(8 * sizeof(IntType1) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType1) - (i + 1) * BitsChar;
            uint64_t c = (chars1 >> shift_len) & char_mask1;
            return c;
        } else {
            i -= max_chars1;
            KASSERT(8 * sizeof(IntType2) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType2) - (i + 1) * BitsChar;
            uint64_t c = (chars2 >> shift_len) & char_mask2;
            return c;
        }
    }

    bool operator<(const DoublePackedInteger& other) const {
        if (chars1 != other.chars1) {
            return chars1 < other.chars1;
        } else {
            return chars2 < other.chars2;
        }
    }
    bool operator==(const DoublePackedInteger& other) const {
        return chars1 == other.chars1 && chars2 == other.chars2;
    }
    bool operator!=(const DoublePackedInteger& other) const {
        return chars1 != other.chars1 || chars2 != other.chars2;
    }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&chars1;
    }
    const char_type* cend_chars() const { return (const char_type*)&chars1; }

    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&chars1;
    }
    char_type* end() const { return (char_type*)&chars1; }
    size_t size() const { return 0; }

    std::string to_string() const { return std::to_string(chars1) + "-" + std::to_string(chars2); }

    static constexpr bool IS_PACKED = true;
    IntType1 chars1;
    IntType2 chars2;
};

template <typename char_type,
          size_t BitsChar,
          typename IntType1 = uint64_t,
          typename IntType2 = uint64_t,
          typename IntType3 = uint64_t>
struct TriplePackedInteger {
    TriplePackedInteger() : chars1(0), chars2(0), chars3(0) {}

    static constexpr uint64_t char_mask1 = (IntType1(1) << BitsChar) - 1;
    static constexpr uint64_t char_mask2 = (IntType2(1) << BitsChar) - 1;
    static constexpr uint64_t char_mask3 = (IntType3(1) << BitsChar) - 1;

    static constexpr uint64_t last_bit1 = 8 * sizeof(IntType1) - BitsChar;
    static constexpr uint64_t last_bit2 = 8 * sizeof(IntType2) - BitsChar;
    static constexpr uint64_t last_bit3 = 8 * sizeof(IntType3) - BitsChar;

    static constexpr uint64_t max_chars1 = 8 * sizeof(IntType1) / BitsChar;
    static constexpr uint64_t max_chars2 = 8 * sizeof(IntType2) / BitsChar;
    static constexpr uint64_t max_chars3 = 8 * sizeof(IntType3) / BitsChar;


    template <typename CharIterator>
    TriplePackedInteger(CharIterator begin, CharIterator end) {
        chars1 = 0;
        chars2 = 0;
        chars3 = 0;
        auto it = begin;
        uint64_t cnt = 0;

        // pack integer from left to right to be able to make lexicographic comparisons with integer

        // pack first integer
        while (it != end && cnt < max_chars1 * BitsChar) {
            IntType1 bits = *it & char_mask1;
            KASSERT(last_bit1 >= cnt);
            IntType1 shifted_bits = bits << (last_bit1 - cnt);
            chars1 |= shifted_bits;
            it++;
            cnt += BitsChar;
        }

        // pack second integer
        cnt = 0;
        while (it != end && cnt < max_chars2 * BitsChar) {
            IntType2 bits = *it & char_mask2;
            KASSERT(last_bit2 >= cnt);
            IntType2 shifted_bits = bits << (last_bit2 - cnt);
            chars2 |= shifted_bits;
            it++;
            cnt += BitsChar;
        }

        // pack third integer
        cnt = 0;
        while (it != end) {
            IntType3 bits = *it & char_mask3;
            KASSERT(last_bit3 >= cnt);
            IntType3 shifted_bits = bits << (last_bit3 - cnt);
            chars3 |= shifted_bits;
            it++;
            cnt += BitsChar;
        }
    }

    char_type at(uint64_t i) const {
        if (i < max_chars1) {
            KASSERT(8 * sizeof(IntType1) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType1) - (i + 1) * BitsChar;
            uint64_t c = (chars1 >> shift_len) & char_mask1;
            return c;
        } else if (i < max_chars1 + max_chars2) {
            i -= max_chars1;
            KASSERT(8 * sizeof(IntType2) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType2) - (i + 1) * BitsChar;
            uint64_t c = (chars2 >> shift_len) & char_mask2;
            return c;
        } else {
            i -= max_chars1;
            i -= max_chars2;
            KASSERT(8 * sizeof(IntType3) >= (i + 1) * BitsChar);
            int shift_len = 8 * sizeof(IntType3) - (i + 1) * BitsChar;
            uint64_t c = (chars3 >> shift_len) & char_mask3;
            return c;
        }
    }

    bool operator<(const TriplePackedInteger& other) const {
        if (chars1 != other.chars1) {
            return chars1 < other.chars1;
        } else if (chars2 != other.chars2) {
            return chars2 < other.chars2;
        } else {
            return chars3 < other.chars3;
        }
    }
    bool operator==(const TriplePackedInteger& other) const {
        return chars1 == other.chars1 && chars2 == other.chars2 && chars3 == other.chars3;
    }
    bool operator!=(const TriplePackedInteger& other) const {
        return chars1 != other.chars1 || chars2 != other.chars2 || chars3 != other.chars3;
    }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&chars1;
    }
    const char_type* cend_chars() const { return (const char_type*)&chars1; }

    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&chars1;
    }
    char_type* end() const { return (char_type*)&chars1; }
    size_t size() const { return 0; }

    std::string to_string() const { return std::to_string(chars1) + "-" + std::to_string(chars2); }

    static constexpr bool IS_PACKED = true;
    IntType1 chars1;
    IntType2 chars2;
    IntType3 chars3;
};

// only 8 bit
template <typename char_type>
struct QuadruplePackedInteger {
    QuadruplePackedInteger() : chars1(), chars2() {}

    template <typename CharIterator>
    QuadruplePackedInteger(CharIterator begin, CharIterator end)
        : chars1(begin, begin + 16),
          chars2(begin + 16, end) {}

    char_type at(uint64_t i) const {
        if (i < 16) {
            return chars1.at(i);
        } else {
            return chars2.at(i - 16);
        }
    }

    bool operator<(const QuadruplePackedInteger& other) const {
        if (chars1 != other.chars1) {
            return chars1 < other.chars1;
        } else {
            return chars2 < other.chars2;
        }
    }
    bool operator==(const QuadruplePackedInteger& other) const {
        return chars1 == other.chars1 && chars2 == other.chars2;
    }
    bool operator!=(const QuadruplePackedInteger& other) const {
        return chars1 != other.chars1 || chars2 != other.chars2;
    }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&chars1;
    }
    const char_type* cend_chars() const { return (const char_type*)&chars1; }

    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&chars1;
    }
    char_type* end() const { return (char_type*)&chars1; }
    size_t size() const { return 0; }

    std::string to_string() const { return chars1.to_string() + "-" + chars2.to_string(); }

    static constexpr bool IS_PACKED = true;
    DoublePackedInteger<char_type, 8, uint64_t, uint64_t> chars1;
    DoublePackedInteger<char_type, 8, uint64_t, uint64_t> chars2;
};

template <size_t K, typename char_type, size_t BitsChar, typename IntType = uint64_t>
struct KPackedInteger {
    KPackedInteger() : packed_integers() {}

    static constexpr size_t BITS_INT_TYPE = 8 * sizeof(IntType);
    static constexpr size_t CHARS_PER_WORD = BITS_INT_TYPE / BitsChar;
    using PackedIntegerType = PackedInteger<char_type, BitsChar, IntType>;

    template <typename CharIterator>
    KPackedInteger(CharIterator begin, CharIterator end) {
        size_t num_chars = end - begin;
        for (size_t k = 0; k < K; k++) {
            size_t limit = std::min((k + 1) * CHARS_PER_WORD, num_chars);
            auto start = begin + k * CHARS_PER_WORD;
            auto end = begin + limit;
            packed_integers[k] = PackedIntegerType(start, end);
        }
    }

    char_type at(uint64_t i) const {
        size_t k = i / CHARS_PER_WORD;
        size_t j = i - k * CHARS_PER_WORD;
        return packed_integers[k].at(j);
    }

    bool operator<(const KPackedInteger& other) const {
        return packed_integers < other.packed_integers;
    }
    bool operator==(const KPackedInteger& other) const {
        return packed_integers == other.packed_integers;
    }
    bool operator!=(const KPackedInteger& other) const {
        return packed_integers != other.packed_integers;
    }

    // dummy method, do not use!
    const char_type* cbegin_chars() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (const char_type*)&packed_integers;
    }
    const char_type* cend_chars() const { return (const char_type*)&packed_integers; }

    char_type* begin() const {
        std::cout << "iterator not supported for PackedInteger\n";
        exit(1);
        return (char_type*)&packed_integers;
    }
    char_type* end() const { return (char_type*)&packed_integers; }
    size_t size() const { return 0; }


    std::string to_string() const {
        std::string out = "";
        for (size_t k = 0; k < K; k++) {
            out += packed_integers[k].to_string();
        }
        return out;
    }

    static constexpr bool IS_PACKED = true;
    std::array<PackedIntegerType, K> packed_integers;
};
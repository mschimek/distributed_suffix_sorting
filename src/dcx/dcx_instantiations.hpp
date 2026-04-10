#pragma once

#include <cstdint>
#include <vector>

#include <kamping/communicator.hpp>

#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "util/uint_types.hpp"

/// Wrapper for a specific PDCX instantiation with character packing.
/// Each specialization is compiled in its own TU for parallel compilation.
/// @tparam char_t     Input character type (e.g. uint8_t, UIntPair<uint8_t>)
/// @tparam DCXParam   Difference cover parameter struct (e.g. DC39Param)
/// @tparam BITS_CHAR  Bits per character for packing (3, 5, 8), or 0 for unpacked
template <typename char_t, typename DCXParam, uint64_t BITS_CHAR>
struct DCXAlgorithm {
    using index_t = dsss::UIntPair<uint8_t>;
    using WordType = uint64_t;
    static constexpr uint64_t X = DCXParam::X;
    static constexpr uint64_t BITS_PER_CHAR = BITS_CHAR;
    static constexpr uint64_t BITS_WORD = 8 * sizeof(WordType);
    static constexpr uint64_t CHARS_PER_WORD = BITS_WORD / BITS_CHAR;
    static constexpr uint64_t NUM_WORDS = (X + CHARS_PER_WORD - 1) / CHARS_PER_WORD;
    static constexpr uint64_t PACKED_CHARS = NUM_WORDS * CHARS_PER_WORD;

    DCXAlgorithm(dsss::dcx::PDCXConfig config) : pdcx_config(config) {
        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
    }

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

/// Specialization for unpacked variant (BITS_CHAR=0): no packing constants needed.
template <typename char_t, typename DCXParam>
struct DCXAlgorithm<char_t, DCXParam, 0> {
    using index_t = dsss::UIntPair<uint8_t>;

    DCXAlgorithm(dsss::dcx::PDCXConfig config) : pdcx_config(config) {}

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

// Convenience aliases for the instantiated variants
using DC39_u8_3bit  = DCXAlgorithm<uint8_t, dsss::dcx::DC39Param, 3>;
using DC39_u8_5bit  = DCXAlgorithm<uint8_t, dsss::dcx::DC39Param, 5>;
using DC39_u8_8bit  = DCXAlgorithm<uint8_t, dsss::dcx::DC39Param, 8>;
using DC39_u8       = DCXAlgorithm<uint8_t, dsss::dcx::DC39Param, 0>;
using DC39_u40      = DCXAlgorithm<dsss::UIntPair<uint8_t>, dsss::dcx::DC39Param, 0>;

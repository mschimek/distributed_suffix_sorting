#pragma once

#include <cstdint>
#include <vector>

#include <kamping/communicator.hpp>

#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "util/uint_types.hpp"

struct DC39Algorithm_uint8_3bit_packing {
    using char_t = std::uint8_t;
    using index_t = dsss::UIntPair<uint8_t>;
    static constexpr uint64_t X = dsss::dcx::DC39Param::X;
    static constexpr uint64_t BITS_CHAR = 3;
    static constexpr uint64_t PACKED_CHARS = 42; // 2 words * 21 chars/word

    DC39Algorithm_uint8_3bit_packing(dsss::dcx::PDCXConfig config) : pdcx_config(config) {
        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
    }

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

struct DC39Algorithm_uint8_5bit_packing {
    using char_t = std::uint8_t;
    using index_t = dsss::UIntPair<uint8_t>;
    static constexpr uint64_t X = dsss::dcx::DC39Param::X;
    static constexpr uint64_t BITS_CHAR = 5;
    static constexpr uint64_t PACKED_CHARS = 48; // 4 words * 12 chars/word

    DC39Algorithm_uint8_5bit_packing(dsss::dcx::PDCXConfig config) : pdcx_config(config) {
        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
    }

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

struct DC39Algorithm_uint8_8bit_packing {
    using char_t = std::uint8_t;
    using index_t = dsss::UIntPair<uint8_t>;
    static constexpr uint64_t X = dsss::dcx::DC39Param::X;
    static constexpr uint64_t BITS_CHAR = 8;
    static constexpr uint64_t PACKED_CHARS = 40; // 5 words * 8 chars/word

    DC39Algorithm_uint8_8bit_packing(dsss::dcx::PDCXConfig config) : pdcx_config(config) {
        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
    }

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

struct DC39Algorithm_uint8_unpacked {
    using char_t = std::uint8_t;
    using index_t = dsss::UIntPair<uint8_t>;

    DC39Algorithm_uint8_unpacked(dsss::dcx::PDCXConfig config) : pdcx_config(config) {}

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

struct DC39Algorithm_uint40_unpacked {
    using char_t = dsss::UIntPair<uint8_t>;
    using index_t = dsss::UIntPair<uint8_t>;

    DC39Algorithm_uint40_unpacked(dsss::dcx::PDCXConfig config) : pdcx_config(config) {}

    std::vector<index_t> compute_suffix_array(std::vector<char_t>& local_string,
                                              kamping::Communicator<>& comm);
    dsss::dcx::PDCXConfig pdcx_config;
};

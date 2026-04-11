#pragma once

#include <cstdint>
#include <vector>

#include <kamping/communicator.hpp>

#include "util/uint_types.hpp"

namespace dsss::dcx {

/// Compute the distributed suffix array of a uint8_t input text.
/// Returns the local portion of the SA using 40-bit indices (UIntPair<uint8_t>).
/// @param input   Local portion of the input text on this PE (will be modified by alphabet compression).
/// @param comm    KaMPIng communicator.
/// @param argc    Number of CLI arguments (for algorithm configuration flags like --dcx, --buckets-*, etc.).
/// @param argv    CLI argument strings.
std::vector<dsss::UIntPair<uint8_t>> compute_suffix_array(
    std::vector<uint8_t>& input,
    kamping::Communicator<>& comm,
    int32_t argc,
    char const* argv[]);

/// Compute the distributed suffix array of a uint40 input text (no character packing).
/// Returns the local portion of the SA using 40-bit indices (UIntPair<uint8_t>).
/// @param input   Local portion of the input text on this PE.
/// @param comm    KaMPIng communicator.
/// @param argc    Number of CLI arguments (for algorithm configuration flags like --dcx, --buckets-*, etc.).
/// @param argv    CLI argument strings.
std::vector<dsss::UIntPair<uint8_t>> compute_suffix_array(
    std::vector<dsss::UIntPair<uint8_t>>& input,
    kamping::Communicator<>& comm,
    int32_t argc,
    char const* argv[]);

} // namespace dsss::dcx

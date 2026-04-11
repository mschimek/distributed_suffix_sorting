#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/named_parameters.hpp>

#include "dcx/dcx.hpp"

/// Build the naive suffix array on rank 0 by sorting suffix indices lexicographically.
std::vector<uint64_t> naive_sa(std::string const& text) {
    size_t const n = text.size();
    std::vector<uint64_t> sa(n);
    std::iota(sa.begin(), sa.end(), 0u);
    std::sort(sa.begin(), sa.end(), [&](uint64_t a, uint64_t b) {
        return text.compare(a, std::string::npos, text, b, std::string::npos) < 0;
    });
    return sa;
}

int main(int argc, char const* argv[]) {
    kamping::Environment env;
    kamping::Communicator comm;

    // Build the input string on rank 0 and distribute it evenly across all ranks.
    std::string const text =
        "mississippi_is_a_long_river_and_has_certainly_more_characters_than_banana";

    std::vector<uint8_t> local_input;
    if (comm.rank() == 0) {
        std::vector<uint8_t> full_input(text.begin(), text.end());
        size_t const n = full_input.size();
        size_t const p = static_cast<size_t>(comm.size());

        // Compute per-rank counts for an even split.
        std::vector<int> send_counts(p);
        for (size_t i = 0; i < p; ++i) {
            send_counts[i] = static_cast<int>(n / p + (i < n % p ? 1 : 0));
        }

        local_input =
            comm.scatterv(kamping::send_buf(full_input), kamping::send_counts(send_counts));
    } else {
        local_input = comm.scatterv<uint8_t>();
    }

    // Minimal argv to configure the algorithm.
    std::vector<char const*> lib_argv = {
        "example",
        "--use-char-packing-samples",
        "--use-char-packing-merging",
    };
    int32_t lib_argc = static_cast<int32_t>(lib_argv.size());

    auto local_sa = dsss::dcx::compute_suffix_array(local_input, comm, lib_argc, lib_argv.data());

    // Gather the full SA on rank 0, verify against naive SA, and print.
    auto sa = comm.gatherv(kamping::send_buf(local_sa));
    if (comm.rank() == 0) {
        std::cout << "Input: \"" << text << "\"  (length " << text.size() << ")" << std::endl;

        // Convert distributed SA to uint64_t for comparison.
        std::vector<uint64_t> dcx_sa(sa.size());
        for (size_t i = 0; i < sa.size(); i++) {
            dcx_sa[i] = static_cast<uint64_t>(sa[i]);
        }

        // Compute reference SA.
        auto ref_sa = naive_sa(text);

        // Compare.
        if (dcx_sa.size() != ref_sa.size()) {
            std::cerr << "FAILED: SA size mismatch (dcx=" << dcx_sa.size()
                      << " vs ref=" << ref_sa.size() << ")" << std::endl;
            return 1;
        }

        bool correct = true;
        for (size_t i = 0; i < dcx_sa.size(); i++) {
            if (dcx_sa[i] != ref_sa[i]) {
                std::cerr << "FAILED: SA[" << i << "] = " << dcx_sa[i] << " (expected " << ref_sa[i]
                          << ")" << std::endl;
                correct = false;
            }
        }

        if (correct) {
            std::cout << "PASSED: DCX suffix array matches naive reference." << std::endl;
        }

        std::cout << "SA:";
        for (auto const& idx: dcx_sa) {
            std::cout << " " << idx;
        }
        std::cout << std::endl;
    }

    return 0;
}

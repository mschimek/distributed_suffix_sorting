// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <vector>

#include <tlx/die.hpp>

#include "strings/bingmann-lcp_losertree.hpp"
#include "strings/lcp_type.hpp"
#include "strings/string_set.hpp"
#include "util/printing.hpp"

// adapted from:
// https://github.com/pmehnert/distributed-string-sorting/blob/master/src/sorter/distributed/merging.hpp

namespace dsss {
inline uint64_t pow2roundup(uint64_t x) {
    if (x == 0u)
        return 1u;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

template <uint64_t K, typename StringType>
void multiway_merge(std::vector<StringType>& input_strings,
                    std::vector<LcpType>& input_lcps,
                    std::vector<int64_t>& interval_sizes) {
    using StringSet = GeneralStringSet<StringType>;
    using String = StringSet::String;
    using MergeAdapter = dsss::StringLcpPtrMergeAdapter<StringSet>;
    using LoserTree = dsss::LcpStringLoserTree_<K, StringSet>;

    if (input_strings.size() == 0) {
        return;
    }

    KASSERT(K == pow2roundup(interval_sizes.size()));
    KASSERT(input_strings.size() == input_lcps.size());

    // check that input lcps are correct
    // int64_t i = 0;
    // for (int64_t s: interval_sizes) {
    //     i++;
    //     for (int j = 1; j < s; j++) {
    //         KASSERT((uint64_t)input_lcps[i] == (uint64_t)compute_lcp(input_strings[i - 1], input_strings[i]));
    //         i++;
    //     }
    // }

    const uint64_t num_elements = input_strings.size();

    // fill up empty intervals with zeros
    interval_sizes.resize(K, 0);

    // allocate output buffer
    std::vector<String> merged_strings(num_elements);
    std::vector<LcpType> lcps(num_elements);

    // create output stream
    StringSet merged_string_set{merged_strings.data(),
                                merged_strings.data() + merged_strings.size()};
    MergeAdapter out_ptr{merged_string_set, lcps.data()};

    // create input stream
    StringSet input_string_set{input_strings.data(), input_strings.data() + input_strings.size()};
    MergeAdapter in_ptr{input_string_set, input_lcps.data()};

    // init loser tree
    LoserTree loser_tree{in_ptr, interval_sizes};
    loser_tree.writeElementsToStream(out_ptr, num_elements);

    // update input vector
    std::swap(input_strings, merged_strings);
}

template <typename StringType>
void multiway_merge(std::vector<StringType>& input_strings,
                    std::vector<LcpType>& input_lcps,
                    std::vector<int64_t>& interval_sizes) {
    switch (pow2roundup(interval_sizes.size())) {
        // clang-format off
        case 1:     multiway_merge<1>(input_strings, input_lcps, interval_sizes); break;
        case 2:     multiway_merge<2>(input_strings, input_lcps, interval_sizes); break;
        case 4:     multiway_merge<4>(input_strings, input_lcps, interval_sizes); break;
        case 8:     multiway_merge<8>(input_strings, input_lcps, interval_sizes); break;
        case 16:    multiway_merge<16>(input_strings, input_lcps, interval_sizes); break;
        case 32:    multiway_merge<32>(input_strings, input_lcps, interval_sizes); break;
        case 64:    multiway_merge<64>(input_strings, input_lcps, interval_sizes); break;
        case 128:   multiway_merge<128>(input_strings, input_lcps, interval_sizes); break;
        case 256:   multiway_merge<256>(input_strings, input_lcps, interval_sizes); break;
        case 512:   multiway_merge<512>(input_strings, input_lcps, interval_sizes); break;
        case 1024:  multiway_merge<1024>(input_strings, input_lcps, interval_sizes); break;
        case 2048:  multiway_merge<2048>(input_strings, input_lcps, interval_sizes); break;
        case 4096:  multiway_merge<4096>(input_strings, input_lcps, interval_sizes); break;
        case 8192:  multiway_merge<8192>(input_strings, input_lcps, interval_sizes); break;
        case 16384: multiway_merge<16384>(input_strings, input_lcps, interval_sizes); break;
        default:
            // todo consider increasing this to 2^15
            tlx_die("Error in merge: K is not 2^i for i in {0,...,14} ");
    }
}
}

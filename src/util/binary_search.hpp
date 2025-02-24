#pragma once

#include <cstdint>

namespace dsss::util {

// finds first index in [l, r] that compares true using binary search

// f f ... f t t ... t t t t
//           x
int64_t binary_search(int64_t l, int64_t r, auto cmp) {
    // (l, r] always contains the searched element
    l -= 1;
    while (r - l > 1) {
        uint64_t m = (l + r) / 2;
        if (cmp(m)) {
            r = m;
        } else {
            l = m;
        }
    }
    return r;
}

// finds first index in [l, r] that compares true using a left to right scan

// f f ... f t t ... t t t t
//           x
int64_t linear_scan(int64_t l, int64_t r, auto cmp) {
    for (int64_t i = l; i < r; i++) {
        if (cmp(i)) {
            return i;
        }
    }
    return r;
}
} // namespace dsss::util
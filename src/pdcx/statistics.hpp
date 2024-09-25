#pragma once

#include <cstdint>
#include <vector>
namespace dsss::dcx {

struct Statistics {
    Statistics() : max_depth(0) {}
    void reset() {
        max_depth = 0;
        string_sizes.clear();
        local_string_sizes.clear();
        highest_ranks.clear();
        char_type_used.clear();
        discarding_reduction.clear();
    }

    int max_depth;
    std::vector<uint64_t> local_string_sizes;
    std::vector<uint64_t> string_sizes;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
    std::vector<double> discarding_reduction;
};

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

} // namespace dsss::dcx
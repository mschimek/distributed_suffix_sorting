#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"
#include "util/printing.hpp"

namespace dsss::dcx {

struct Statistics {
    Statistics() : max_depth(0) {}
    void reset() {
        algo = "";
        num_processors = 0;
        max_depth = 0;
        string_sizes.clear();
        local_string_sizes.clear();
        highest_ranks.clear();
        char_type_used.clear();
        discarding_reduction.clear();
        use_discarding.clear();
        space_efficient_sort.clear();
        string_imbalance.clear();
        sample_imbalance.clear();
        sa_imbalance.clear();
        bucket_imbalance.clear();
        redistribute_chars.clear();
        redistribute_samples.clear();
        avg_lcp_len_samples.clear();
        avg_lcp_len_merging.clear();
        avg_segment.clear();
        max_segment.clear();
    }

    void print() const {
        using namespace kamping;
        std::cout << "\nStatistics:\n";
        std::cout << V(algo) << std::endl;
        std::cout << V(num_processors) << std::endl;
        std::cout << "max_depth=" << max_depth << std::endl;
        std::cout << "string_sizes=";
        print_vector(string_sizes, ",");
        std::cout << "highest_ranks=";
        print_vector(highest_ranks, ",");
        std::cout << "char_type_bits=";
        print_vector(char_type_used, ",");
        std::cout << "discarding_reduction=";
        print_vector(discarding_reduction, ",");
        std::cout << "use_discarding=";
        print_vector(use_discarding, ",");
        std::cout << "space_efficient_sort=";
        print_vector(space_efficient_sort, ",");
        std::cout << "string_imbalance=";
        print_vector(string_imbalance, ",");
        std::cout << "sample_imbalance=";
        print_vector(sample_imbalance, ",");
        std::cout << "sa_imbalance=";
        print_vector(sa_imbalance, ",");
        std::cout << "bucket_imbalance=";
        print_vector(bucket_imbalance, ",");
        std::cout << "redistribute_chars=";
        print_vector(redistribute_chars, ",");
        std::cout << "redistribute_samples=";
        print_vector(redistribute_samples, ",");
        std::cout << "avg_lcp_len_samples=";
        print_vector(avg_lcp_len_samples, ",");
        std::cout << "avg_lcp_len_merging=";
        print_vector(avg_lcp_len_merging, ",");
        std::cout << "avg_segment=";
        print_vector(avg_segment, ",");
        std::cout << "max_segment=";
        print_vector(max_segment, ",");
        std::cout << std::endl;
    }

    std::string algo;
    int num_processors;
    int max_depth;
    std::vector<uint64_t> local_string_sizes;
    std::vector<uint64_t> string_sizes;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
    std::vector<double> discarding_reduction;
    std::vector<bool> use_discarding;
    std::vector<bool> space_efficient_sort;
    std::vector<double> string_imbalance;
    std::vector<double> sample_imbalance;
    std::vector<double> sa_imbalance;
    std::vector<double> bucket_imbalance;
    std::vector<bool> redistribute_chars;
    std::vector<bool> redistribute_samples;
    std::vector<double> avg_lcp_len_samples;
    std::vector<double> avg_lcp_len_merging;
    std::vector<double> avg_segment;
    std::vector<uint64_t> max_segment;
};

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

} // namespace dsss::dcx
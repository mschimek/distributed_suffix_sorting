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
        space_efficient_sort.clear();
        string_imbalance.clear();
        sample_imbalance.clear();
        sa_imbalance.clear();
        bucket_imbalance.clear();
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
        std::cout << "\n";
        // TODO round doubles
    }

    std::string algo;
    int num_processors;
    int max_depth;
    std::vector<uint64_t> local_string_sizes;
    std::vector<uint64_t> string_sizes;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
    std::vector<double> discarding_reduction;
    std::vector<bool> space_efficient_sort;
    std::vector<double> string_imbalance;
    std::vector<double> sample_imbalance;
    std::vector<double> sa_imbalance;
    std::vector<double> bucket_imbalance;
};

double compute_imbalance(uint64_t local_size, kamping::Communicator<> &comm) {
    using namespace kamping;
    uint64_t total_size = mpi_util::all_reduce(local_size, ops::plus<>(), comm);
    uint64_t largest_size = mpi_util::all_reduce(local_size, ops::max<>(), comm);
    double avg_size = (double)total_size / comm.size();
    double imbalance = ((double)largest_size / avg_size) - 1.0;
    return imbalance;
}

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

} // namespace dsss::dcx
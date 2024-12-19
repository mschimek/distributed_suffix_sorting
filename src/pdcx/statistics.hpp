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
        string_imbalance.clear();
        sample_imbalance.clear();
        sa_imbalance.clear();
        bucket_imbalance_samples.clear();
        bucket_imbalance_samples_received.clear();
        bucket_imbalance_merging.clear();
        bucket_imbalance_merging_received.clear();
        redistribute_chars.clear();
        redistribute_samples.clear();
        avg_segment.clear();
        max_segment.clear();
        bucket_sizes.clear();
        packed_chars_samples.clear();
        packed_chars_merging.clear();
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
        std::cout << "string_imbalance=";
        print_vector(string_imbalance, ",");
        std::cout << "sample_imbalance=";
        print_vector(sample_imbalance, ",");
        std::cout << "sa_imbalance=";
        print_vector(sa_imbalance, ",");
        std::cout << "bucket_imbalance_samples=";
        print_vector(bucket_imbalance_samples_received, ",");
        std::cout << "bucket_imbalance_samples_received=";
        print_vector(bucket_imbalance_samples, ",");
        std::cout << "bucket_imbalance_merging=";
        print_vector(bucket_imbalance_merging, ",");
        std::cout << "bucket_imbalance_merging_received=";
        print_vector(bucket_imbalance_merging_received, ",");
        std::cout << "redistribute_chars=";
        print_vector(redistribute_chars, ",");
        std::cout << "redistribute_samples=";
        print_vector(redistribute_samples, ",");
        std::cout << "avg_segment=";
        print_vector(avg_segment, ",");
        std::cout << "max_segment=";
        print_vector(max_segment, ",");
        std::cout << "bucket_sizes=";
        print_vector(bucket_sizes, ",");
        std::cout << "packed_chars_samples=";
        print_vector(packed_chars_samples, ",");
        std::cout << "packed_chars_merging=";
        print_vector(packed_chars_merging, ",");
        std::cout << "max_mem_pe_phase_01=";
        print_vector(max_mem_pe_phase_01, ",");
        std::cout << "max_mem_pe_phase_02=";
        print_vector(max_mem_pe_phase_02, ",");
        std::cout << "max_mem_pe_phase_03=";
        print_vector(max_mem_pe_phase_03, ",");
        std::cout << "max_mem_pe_phase_04=";
        print_vector(max_mem_pe_phase_04, ",");
        std::cout << "max_mem_pe_chunking_before_sort=";
        print_vector(max_mem_pe_chunking_before_sort, ",");
        std::cout << "max_mem_pe_chunking_after_sort=";
        print_vector(max_mem_pe_chunking_after_sort, ",");
        std::cout << "max_mem_pe_chunking_after_concat=";
        print_vector(max_mem_pe_chunking_after_concat, ",");
        std::cout << "max_mem_pe_chunking_after_alltoal=";
        print_vector(max_mem_pe_chunking_after_alltoal, ",");
        std::cout << "phase_04_sa_size=";
        print_vector(phase_04_sa_size, ",");
        std::cout << "phase_04_sa_capacity=";
        print_vector(phase_04_sa_capacity, ",");
        std::cout << "phase_04_before_alltoall_chunks=";
        print_vector(phase_04_before_alltoall_chunks, ",");
        std::cout << "phase_04_after_alltoall_chunks=";
        print_vector(phase_04_after_alltoall_chunks, ",");
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
    std::vector<double> string_imbalance;
    std::vector<double> sample_imbalance;
    std::vector<double> sa_imbalance;
    std::vector<double> bucket_imbalance_samples;
    std::vector<double> bucket_imbalance_samples_received;
    std::vector<double> bucket_imbalance_merging;
    std::vector<double> bucket_imbalance_merging_received;
    std::vector<bool> redistribute_chars;
    std::vector<bool> redistribute_samples;
    std::vector<double> avg_segment;
    std::vector<uint64_t> max_segment;
    std::vector<uint64_t> bucket_sizes;
    std::vector<double> packed_chars_samples;
    std::vector<double> packed_chars_merging;
    std::vector<uint64_t> max_mem_pe_phase_01;
    std::vector<uint64_t> max_mem_pe_phase_02;
    std::vector<uint64_t> max_mem_pe_phase_03;
    std::vector<uint64_t> max_mem_pe_phase_04;
    std::vector<uint64_t> max_mem_pe_chunking_before_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_concat;
    std::vector<uint64_t> max_mem_pe_chunking_after_alltoal;
    std::vector<uint64_t> phase_04_sa_size;
    std::vector<uint64_t> phase_04_sa_capacity;
    std::vector<uint64_t> phase_04_before_alltoall_chunks;
    std::vector<uint64_t> phase_04_after_alltoall_chunks;
    
};

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

} // namespace dsss::dcx
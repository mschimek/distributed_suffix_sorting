#pragma once

#include <cstdint>
#include <iostream>
#include <limits>

#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXConfig {
    mpi::SampleSortConfig sample_sort_config = mpi::SampleSortConfig();
    mpi::AtomicSorters atomic_sorter = mpi::AtomicSorters::SampleSort;
    dsss::SeqStringSorter string_sorter = dsss::SeqStringSorter::MultiKeyQSort;
    std::vector<uint32_t> buckets_samples;
    std::vector<uint32_t> buckets_merging;
    uint64_t ams_levels = 1;
    uint64_t num_samples_splitters = 100;
    double discarding_threshold = 0.7;
    double min_imbalance = 0.25;
    bool use_string_sort = false;
    bool use_lcps_tie_breaking = false;
    bool use_random_sampling_splitters = false;
    bool balance_blocks_space_efficient_sort = false;
    bool print_phases = true;


    uint32_t buckets_samples_at_level(uint32_t level) const {
        return level < buckets_samples.size() ? buckets_samples[level] : 1;
    }

    uint32_t buckets_merging_at_level(uint32_t level) const {
        return level < buckets_merging.size() ? buckets_merging[level] : 1;
    }

    void print_config() const {
        std::cout << "PDCXConfig:\n";
        std::cout << V(discarding_threshold) << "\n";
        std::cout << "atomic_sorter=" << mpi::atomic_sorter_names[atomic_sorter] << "\n";
        std::cout << "string_sorter=" << dsss::string_sorter_names[string_sorter] << "\n";
        std::cout << "buckets_samples=";
        kamping::print_vector(buckets_samples, ",");
        std::cout << "buckets_merging=";
        kamping::print_vector(buckets_merging, ",");
        std::cout << V(use_string_sort) << "\n";
        std::cout << V(use_lcps_tie_breaking) << "\n";
        std::cout << V(ams_levels) << "\n";
        std::cout << V(num_samples_splitters) << "\n";
        std::cout << V(use_random_sampling_splitters) << "\n";
        std::cout << V(balance_blocks_space_efficient_sort) << "\n";
        std::cout << std::endl;

        sample_sort_config.print_config();
        std::cout << std::endl;
    }
};

} // namespace dsss::dcx
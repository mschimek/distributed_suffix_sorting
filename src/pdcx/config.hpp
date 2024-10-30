#pragma once

#include <cstdint>
#include <iostream>
#include <limits>

#include "sorters/sample_sort_config.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXConfig {
    mpi::SampleSortConfig sample_sort_config = mpi::SampleSortConfig();
    mpi::AtomicSorters atomic_sorter = mpi::AtomicSorters::SampleSort;
    dsss::SeqStringSorter string_sorter = dsss::SeqStringSorter::MultiKeyQSort;
    uint64_t blocks_space_efficient_sort = 1;
    uint64_t threshold_space_efficient_sort = std::numeric_limits<uint64_t>::max();
    uint64_t ams_levels = 1;
    double discarding_threshold = 0.7;
    bool use_string_sort = false;
    bool use_loser_tree = false;
    bool print_phases = true;

    void print_config() const {
        std::cout << "PDCXConfig:\n";
        std::cout << V(discarding_threshold) << "\n";
        std::cout << "atomic_sorter=" << mpi::atomic_sorter_names[atomic_sorter] << "\n";
        std::cout << "string_sorter=" << dsss::string_sorter_names[string_sorter] << "\n";
        std::cout << V(use_string_sort) << "\n";
        std::cout << V(use_loser_tree) << "\n";
        std::cout << V(blocks_space_efficient_sort) << "\n";
        std::cout << V(threshold_space_efficient_sort) << "\n";
        std::cout << V(ams_levels) << "\n";
        std::cout << std::endl;
        
        sample_sort_config.print_config();
        std::cout << std::endl;
    }
};

} // namespace dsss::dcx
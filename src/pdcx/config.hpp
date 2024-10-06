#pragma once

#include <cstdint>
#include <iostream>
#include <limits>

#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXConfig {
    double discarding_threshold = 0.5;
    bool use_old_discarding = false;
    mpi::AtomicSorters atomic_sorter = mpi::AtomicSorters::SampleSort;
    uint64_t blocks_space_efficient_sort = 2;
    uint64_t threshold_space_efficient_sort = std::numeric_limits<uint64_t>::max();
    bool print_phases = true;

    void print_config() const {
        std::cout << "PDCXConfig:\n";
        std::cout << V(discarding_threshold) << "\n";
        std::cout << V(use_old_discarding) << "\n";
        std::cout << "atomic_sorter=" << mpi::atomic_sorter_names[atomic_sorter] << "\n";
        std::cout << V(blocks_space_efficient_sort) << "\n";
        std::cout << V(threshold_space_efficient_sort) << "\n";
        std::cout << std::endl;
    }
};

} // namespace dsss::dcx
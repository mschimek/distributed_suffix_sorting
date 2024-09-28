#pragma once

#include <iostream>

#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXConfig {
    double discarding_threshold = 0.5;
    bool use_old_discarding = false;
    mpi::AtomicSorters atomic_sorter = mpi::AtomicSorters::SampleSort;


    void print_config() const {
        std::cout << "PDCXConfig\n";
        std::cout << V(discarding_threshold) << "\n";
        std::cout << V(use_old_discarding) << "\n";
        std::cout << "atomic_sorter=" << mpi::atomic_sorter_names[atomic_sorter] << "\n";
        std::cout << "\n";
    }
};

} // namespace dsss::dcx
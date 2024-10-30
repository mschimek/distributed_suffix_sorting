#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "util/printing.hpp"

namespace dsss::mpi {

enum SplitterSampling { Uniform, Random };
enum SplitterSorting { Central, Distributed };
static std::vector<std::string> splitter_sampling_names = {"uniform", "random"};
static std::vector<std::string> splitter_sorting_names = {"central", "distributed"};

struct SampleSortConfig {
    bool use_loser_tree = false;
    bool use_rquick_for_splitters = false;
    SplitterSampling splitter_sampling = SplitterSampling::Uniform;
    SplitterSorting splitter_sorting = SplitterSorting::Central;

    void print_config() const {
        std::cout << "SampleSortConfig:\n";
        std::cout << V(use_loser_tree) << "\n";
        std::cout << V(use_rquick_for_splitters) << "\n";
        std::cout << "splitter_sampling=" << splitter_sampling_names[splitter_sampling] << "\n";
        std::cout << "splitter_sorting=" << splitter_sorting_names[splitter_sorting] << "\n";
        std::cout << std::endl;
    }
};
} // namespace dsss::mpi
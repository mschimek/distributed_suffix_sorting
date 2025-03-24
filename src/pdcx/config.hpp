#pragma once

#include <cstdint>
#include <iostream>
#include <limits>

#include <magic_enum/magic_enum.hpp>

#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXLengthInfo {
    uint64_t local_sample_size = 0;
    uint64_t total_sample_size = 0;
    uint64_t local_chars = 0;
    uint64_t local_chars_with_dummy = 0;
    uint64_t total_chars = 0;
    uint64_t largest_char = 0;
    uint64_t chars_before = 0;
    uint64_t samples_before = 0;
    uint64_t recursion_depth = 0;
};

struct PDCXConfig {
    mpi::SampleSortConfig sample_sort_config = mpi::SampleSortConfig();
    mpi::AtomicSorters atomic_sorter = mpi::AtomicSorters::SampleSort;
    dsss::SeqStringSorter string_sorter = dsss::SeqStringSorter::MultiKeyQSort;
    std::vector<uint32_t> buckets_samples;
    std::vector<uint32_t> buckets_merging;
    uint32_t buckets_phase3 = 1;
    uint32_t num_samples_phase3 = 10000;
    uint64_t ams_levels = 1;
    uint64_t memory_seq_string_sorter = 0;
    uint64_t num_samples_splitters = 1000;
    uint64_t avg_chunks_pe = 1000;
    uint64_t seed = 0;
    uint64_t pack_extra_words = 0;
    mutable double packing_ratio = 1;
    double discarding_threshold = 0.7;
    double min_imbalance = 0.25;
    bool use_string_sort = false;
    bool use_string_sort_tie_breaking_phase1 = false;
    bool use_string_sort_tie_breaking_phase4 = false;
    bool use_random_sampling_splitters = false;
    bool balance_blocks_space_efficient_sort = false;
    bool use_randomized_chunks = false;
    bool use_char_packing_samples = false;
    bool use_char_packing_merging = false;
    bool print_phases = true;
    bool rearrange_buckets_balanced = false;
    bool use_robust_tie_break = false;
    bool use_compressed_buckets = false;


    uint32_t buckets_samples_at_level(uint32_t level) const {
        return level < buckets_samples.size() ? buckets_samples[level] : 1;
    }

    uint32_t buckets_merging_at_level(uint32_t level) const {
        return level < buckets_merging.size() ? buckets_merging[level] : 1;
    }

    std::vector<std::pair<std::string, std::string>> config() const {

      auto join_vector = [](const std::vector<std::uint32_t>& vec) {
        std::ostringstream oss;
        for (std::size_t i = 0; i < vec.size(); ++i) {
          oss << vec[i];
          if (i + 1 != vec.size()) {
            oss << ",";
          }
        }
        return oss.str();
      };

      std::vector<std::pair<std::string, std::string>> config_vector = sample_sort_config.config();
      config_vector.emplace_back("atomic_sorter", magic_enum::enum_name(atomic_sorter));
      config_vector.emplace_back("string_sorter", magic_enum::enum_name(string_sorter));
      config_vector.emplace_back("buckets_samples", join_vector(buckets_samples));
      config_vector.emplace_back("buckets_merging", join_vector(buckets_merging));
      config_vector.emplace_back("buckets_phase3", std::to_string(buckets_phase3));
      config_vector.emplace_back("num_samples_phase3", std::to_string(num_samples_phase3));
      config_vector.emplace_back("ams_levels", std::to_string(ams_levels));
      config_vector.emplace_back("memory_seq_string_sorter", std::to_string(memory_seq_string_sorter));
      config_vector.emplace_back("num_samples_splitters", std::to_string(num_samples_splitters));
      config_vector.emplace_back("avg_chunks_pe", std::to_string(avg_chunks_pe));
      config_vector.emplace_back("seed", std::to_string(seed));
      config_vector.emplace_back("pack_extra_words", std::to_string(pack_extra_words));
      config_vector.emplace_back("packing", std::to_string(packing_ratio));
      config_vector.emplace_back("discarding_threshold", std::to_string(discarding_threshold));
      config_vector.emplace_back("min_imbalance", std::to_string(min_imbalance));
      config_vector.emplace_back("use_string_sort", std::to_string(use_string_sort));
      config_vector.emplace_back("use_string_sort_tie_breaking_phase1", std::to_string(use_string_sort_tie_breaking_phase1));
      config_vector.emplace_back("use_string_sort_tie_breaking_phase4", std::to_string(use_string_sort_tie_breaking_phase4));
      config_vector.emplace_back("use_random_sampling_splitters", std::to_string(use_random_sampling_splitters));
      config_vector.emplace_back("balance_blocks_space_efficient_sort", std::to_string(balance_blocks_space_efficient_sort));
      config_vector.emplace_back("use_randomized_chunks", std::to_string(use_randomized_chunks));
      config_vector.emplace_back("use_char_packing_samples", std::to_string(use_char_packing_samples));
      config_vector.emplace_back("use_char_packing_merging", std::to_string(use_char_packing_merging));
      config_vector.emplace_back("print_phases", std::to_string(print_phases));
      config_vector.emplace_back("rearrange_buckets_balanced", std::to_string(rearrange_buckets_balanced));
      config_vector.emplace_back("use_robust_tie_break", std::to_string(use_robust_tie_break));
      config_vector.emplace_back("use_compressed_buckets", std::to_string(use_compressed_buckets));
      return config_vector;
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
        std::cout << V(buckets_phase3) << "\n";
        std::cout << V(num_samples_phase3) << "\n";
        std::cout << V(use_string_sort) << "\n";
        std::cout << V(use_string_sort_tie_breaking_phase1) << "\n";
        std::cout << V(use_string_sort_tie_breaking_phase4) << "\n";
        std::cout << V(use_char_packing_samples) << "\n";
        std::cout << V(use_char_packing_merging) << "\n";
        std::cout << V(ams_levels) << "\n";
        std::cout << V(memory_seq_string_sorter) << "\n";
        std::cout << V(num_samples_splitters) << "\n";
        std::cout << V(use_random_sampling_splitters) << "\n";
        std::cout << V(balance_blocks_space_efficient_sort) << "\n";
        std::cout << V(use_randomized_chunks) << "\n";
        std::cout << V(avg_chunks_pe) << "\n";
        std::cout << V(packing_ratio) << "\n";
        std::cout << V(rearrange_buckets_balanced) << "\n";
        std::cout << V(use_robust_tie_break) << "\n";
        std::cout << V(use_compressed_buckets) << "\n";
        std::cout << V(pack_extra_words) << "\n";
        std::cout << std::endl;

        sample_sort_config.print_config();
        std::cout << std::endl;
    }
};

} // namespace dsss::dcx

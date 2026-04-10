#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/named_parameters.hpp>
#include <magic_enum/magic_enum.hpp>

#include "CLI_mpi.hpp"
#include "mpi/reduce.hpp"
#include "pdcx/config.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/uint_types.hpp"

#include "dcx/dcx_instantiations.hpp"

namespace dsss::dcx::common {

using namespace dsss;

template <typename Enum>
auto string_map() -> std::map<std::string, Enum> {
    std::map<std::string, Enum> result;
    auto entries = magic_enum::enum_entries<Enum>();
    for (auto const& entry: entries) {
        result.emplace(entry.second, entry.first);
    }
    return result;
}

/// Adds all PDCXConfig-related CLI options to an existing CLI::App.
/// Call this to share option definitions between the standalone CLI and the library.
inline void add_pdcx_options(CLI::App& app, PDCXConfig& config) {
    app.add_option("--seed", config.seed,
                   "Seed to be used for random. PE i uses seed: seed + i");
    app.add_option("--discarding-threshold", config.discarding_threshold,
                   "Value between [0, 1], threshold when to use discarding optimization.");
    app.add_option("--num-samples-splitters", config.num_samples_splitters,
                   "Total number of random samples to use to determine bucket splitters in space "
                   "efficient sort.");
    app.add_flag("--use-random-sampling-splitters", config.use_random_sampling_splitters,
                 "Use random sampling to determine block splitters in space efficient sort.");
    app.add_flag("--balance-blocks-space-efficient-sort",
                 config.balance_blocks_space_efficient_sort,
                 "Balance blocks after materialization in space efficient sorting.");

    uint64_t const num_buckets_limit = std::numeric_limits<uint16_t>::max();
    app.add_option("--buckets-sample-phase", config.buckets_samples,
                   "Number of buckets to use for space efficient sorting of samples on each "
                   "recursion level. Missing values default to 1. Example: 16,8,4")
        ->delimiter(',')
        ->check(CLI::Range(num_buckets_limit));
    app.add_option("--buckets-merging-phase", config.buckets_merging,
                   "Number of buckets to use for space efficient sorting in merging phase on each "
                   "recursion level. Missing values default to 1. Example: 16,8,4. If you "
                   "use large bucket sizes you should also set num_samples_splitters (-m) high "
                   "enough. 16 b log b for b buckets should be enough.")
        ->delimiter(',')
        ->check(CLI::Range(num_buckets_limit));
    app.add_option("--buckets-phase3", config.buckets_phase3,
                   "Number of buckets to use for space efficient sorting in Phase 3 for rri.");
    app.add_option("--samples-buckets-phase3", config.num_samples_phase3);
    app.add_flag("--use-randomized-chunks", config.use_randomized_chunks,
                 "Use randomized chunks in bucket sorting to distribute work.");
    app.add_option("--avg-chunks-pe", config.avg_chunks_pe,
                   "Average number of chunks on a PE.");
    app.add_flag("--use-char-packing-samples", config.use_char_packing_samples,
                 "Pack multiple characters in the same datatype for phase 1 (samples) on the first level.");
    app.add_flag("--use-char-packing-merging", config.use_char_packing_merging,
                 "Pack multiple characters in the same datatype for phase 4 (merging) on the first level.");
    app.add_flag("--rearrange-buckets-balanced", config.rearrange_buckets_balanced,
                 "Balances the buckets in a balanced way, which needs an additional output buffer "
                 "and some bookkeeping information.");
    app.add_flag("--use-robust-tie-break", config.use_robust_tie_break,
                 "Use ranks as a tie break in space efficient sorting in Phase 4. Is slower but "
                 "splits equal strings amoung buckets.");
    app.add_flag("--use-compressed-buckets", config.use_compressed_buckets,
                 "Store the bucket mapping compressed in the same memory as the SA.");
    app.add_option("--pack-extra-words", config.pack_extra_words,
                   "Use specificed number of extra words when packing characters into words. "
                   "Currently supports 0 and 1.");

    // sorter configuration
    app.add_option("--atomic-sorter", config.atomic_sorter,
                   "Atomic sorter to be used. [sample_sort, rquick, ams, bitonic, rfis]")
        ->transform(CLI::CheckedTransformer(string_map<mpi::AtomicSorters>(), CLI::ignore_case));
    app.add_option("--ams-levels", config.ams_levels,
                   "Number of levels to be used in ams.");

    app.add_option("--splitter-sampling", config.sample_sort_config.splitter_sampling,
                   "Splitter sampling method in sample sort. [uniform, random]")
        ->transform(CLI::CheckedTransformer(string_map<mpi::SplitterSampling>(), CLI::ignore_case));
    app.add_option("--splitter-sorting", config.sample_sort_config.splitter_sorting,
                   "Splitter sorting method in sample sort [central, distributed]")
        ->transform(CLI::CheckedTransformer(string_map<mpi::SplitterSorting>(), CLI::ignore_case));
    app.add_option("--string-sorter", config.string_sorter,
                   "String sorter to be used. [multi_key_qsort, radix_sort_ci2, radix_sort_ci3]")
        ->transform(CLI::CheckedTransformer(string_map<dsss::SeqStringSorter>(), CLI::ignore_case));
    app.add_option("--memory_seq_string_sorter", config.memory_seq_string_sorter,
                   "Memory hint for sequential string sorter.");

    app.add_flag("--use-string-sort", config.use_string_sort,
                 "Use string sorting instead of atomic sorting.");
    app.add_flag("--use-string-sort-tie-breaking-phase1",
                 config.use_string_sort_tie_breaking_phase1,
                 "Use string sorting with index-tie-breaking in Phase 1.");
    app.add_flag("--use-string-sort-tie-breaking-phase4",
                 config.use_string_sort_tie_breaking_phase4,
                 "Use string sorting with rank-tie-breaking in Phase 4.");
    app.add_flag("--use-loser-tree", config.sample_sort_config.use_loser_tree,
                 "Use loser tree in merging step of sample sort.");
    app.add_flag("--use-rquick-for-splitters",
                 config.sample_sort_config.use_rquick_for_splitters,
                 "Use Rquick to sort splitter.");
    app.add_flag("--use-binary-search-for-splitters",
                 config.sample_sort_config.use_binary_search_for_splitters,
                 "Use binary search instead of linear scan to find intervals in sample sort.");
    app.add_flag("--use-lcp-compression", config.sample_sort_config.use_lcp_compression,
                 "Use lcp-compression in string sample sort to reduce communication volume.");
    app.add_option("--lcp-compression-threshold",
                   config.sample_sort_config.lcp_compression_threshold,
                   "Value between [0, 1], threshold on compression ratio when to start using "
                   "LCP-compression.");
    app.add_flag("--use-prefix-doubling", config.sample_sort_config.use_prefix_doubling,
                 "Use prefix-doubling in string sample sort to reduce communication volume.");
    app.add_option("--inital-prefix-length",
                   config.sample_sort_config.inital_prefix_length,
                   "Inital prefix-length to use for prefix doubling.");
}

/// Convenience wrapper: creates a CLI::App, adds PDCXConfig options, parses, and returns the config.
inline PDCXConfig parse_pdcx_config(int32_t argc, char const* argv[], bool allow_extras = false) {
    PDCXConfig config;
    CLI::App app{"DCX Configuration"};
    if (allow_extras) {
        app.allow_extras();
    }
    add_pdcx_options(app, config);
    CLI11_PARSE_MPI(app, argc, argv);
    return config;
}

template <typename char_t>
uint64_t compress_alphabet(std::vector<char_t>& input, kamping::Communicator<>& comm) {
    uint64_t max_alphabet_size = 256;

    // should not happen, because we read characters as bytes
    uint64_t max_char = mpi_util::all_reduce_max(input, comm);
    if (max_char > max_alphabet_size) {
        kamping::report_on_root(
            "Can only process alphabets with not more than 255 distinct "
            "characters. 0 is reserved for special characters. Change char_t.",
            comm);
        exit(1);
    }

    // determine character frequencies
    std::vector<uint64_t> local_counts(max_alphabet_size, 0);
    for (auto c: input) {
        local_counts[c]++;
    }
    std::vector<uint64_t> global_counts =
        comm.allreduce(kamping::send_buf(local_counts), kamping::op(kamping::ops::plus<>{}));
    uint64_t alphabet_size =
        local_counts.size() - std::count(global_counts.begin(), global_counts.end(), 0);

    if (alphabet_size == local_counts.size()) {
        kamping::report_on_root(
            "Can only process alphabets with not more than 255 distinct "
            "characters. 0 is reserved for special characters. Change char_t.",
            comm);
        exit(1);
    }

    // reserve character 0 for padding
    uint64_t next_char = 1;
    std::vector<uint64_t> map_char(max_alphabet_size);
    for (uint64_t i = 0; i < max_alphabet_size; i++) {
        if (global_counts[i] > 0) {
            map_char[i] = next_char++;
        }
    }

    // map input alphabet to compressed alphabet
    for (uint64_t i = 0; i < input.size(); i++) {
        input[i] = map_char[input[i]];
    }
    kamping::report_on_root("input_alphabet_size=" + std::to_string(alphabet_size), comm);
    return alphabet_size;
}

template <typename PDCXType, typename char_t, typename index_t>
void run_pdcx(kamping::Communicator<>& comm,
              const PDCXConfig& pdcx_config,
              std::vector<char_t>& local_string,
              std::vector<index_t>& local_sa) {
    auto algo = PDCXType(pdcx_config, comm);
    local_sa = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
}

inline void run_packed_dcx_variant(kamping::Communicator<>& comm,
                            PDCXConfig const& pdcx_config,
                            uint64_t input_alphabet_size,
                            std::vector<uint8_t>& local_string,
                            std::vector<dsss::UIntPair<uint8_t>>& local_sa) {
    uint64_t packed_chars;
    uint64_t bits_per_char;
    double packing_ratio;

    if (input_alphabet_size <= (1 << 3) - 1) {
        DC39_u8_3bit algo(pdcx_config);
        local_sa = algo.compute_suffix_array(local_string, comm);
        packed_chars = algo.PACKED_CHARS;
        bits_per_char = algo.BITS_PER_CHAR;
        packing_ratio = algo.pdcx_config.packing_ratio;
    } else if (input_alphabet_size <= (1 << 5) - 1) {
        DC39_u8_5bit algo(pdcx_config);
        local_sa = algo.compute_suffix_array(local_string, comm);
        packed_chars = algo.PACKED_CHARS;
        bits_per_char = algo.BITS_PER_CHAR;
        packing_ratio = algo.pdcx_config.packing_ratio;
    } else {
        DC39_u8_8bit algo(pdcx_config);
        local_sa = algo.compute_suffix_array(local_string, comm);
        packed_chars = algo.PACKED_CHARS;
        bits_per_char = algo.BITS_PER_CHAR;
        packing_ratio = algo.pdcx_config.packing_ratio;
    }

    // logging
    kamping::report_on_root("packed_chars=" + std::to_string(packed_chars), comm);
    kamping::report_on_root("_packing_ratio=" + std::to_string(packing_ratio), comm);
    kamping::report_on_root("bits_per_char=" + std::to_string(bits_per_char), comm);
}

inline void compute_sa(kamping::Communicator<>& comm,
                PDCXConfig const& pdcx_config,
                std::vector<uint8_t>& local_string,
                std::vector<dsss::UIntPair<uint8_t>>& local_sa) {
    using namespace kamping;

    measurements::Timer<Communicator<>> algo_timer;
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("total_time");
    algo_timer.synchronize_and_start("total_time");

    timer.synchronize_and_start("compress_alphabet");
    uint64_t input_alphabet_size = compress_alphabet(local_string, comm);
    timer.stop();

    if (pdcx_config.use_char_packing_merging || pdcx_config.use_char_packing_samples) {
        if (pdcx_config.pack_extra_words == 0) {
            run_packed_dcx_variant(comm, pdcx_config, input_alphabet_size,
                                  local_string, local_sa);
        } else {
            throw std::runtime_error("currently not instantiated");
        }
    } else {
        DC39_u8 algo(pdcx_config);
        local_sa = algo.compute_suffix_array(local_string, comm);
    }

    algo_timer.stop();
    timer.stop();
    algo_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
    kamping::report_on_root("\n", comm);
}

} // namespace dsss::dcx::common

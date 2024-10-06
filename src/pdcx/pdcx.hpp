#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kassert/kassert.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "mpi/zip.hpp"
#include "pdcx/compute_ranks.hpp"
#include "pdcx/config.hpp"
#include "pdcx/merge_samples.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/sequential_sa.hpp"
#include "pdcx/statistics.hpp"
#include "sa_check.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"
#include "util/uint_types.hpp"

namespace dsss::dcx {

using namespace kamping;


template <typename char_type, typename index_type, typename DC>
class PDCX {
    using SampleString = DCSampleString<char_type, index_type, DC>;
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using MergeSamples = DCMergeSamples<char_type, index_type, DC>;

public:
    PDCX(PDCXConfig& _config, Communicator<>& _comm)
        : config(_config),
          atomic_sorter(_comm),
          comm(_comm),
          timer(measurements::timer()),
          stats(get_stats_instance()),
          recursion_depth(0) {
        atomic_sorter.set_sorter(config.atomic_sorter);
    }

    // maps the index i from a recursive dcx call back to the global index
    index_type map_back(index_type idx) {
        // find interval into which index belongs
        for (uint i = 0; i < DC::D; i++) {
            if (idx < samples_before[i + 1]) {
                index_type d = DC::DC[i];
                index_type k = idx - samples_before[i];
                return DC::X * k + d;
            }
        }
        KASSERT(false);
        return 0;
    }

    template <typename T>
    void free_memory(std::vector<T>& vec) {
        vec.clear();
        vec.shrink_to_fit();
    }

    void add_padding(std::vector<char_type>& local_data) {
        char_type padding = char_type(0);
        if (comm.rank() == comm.size() - 1) {
            std::fill_n(std::back_inserter(local_data), DC::X, padding);
        }
    }

    void remove_padding(std::vector<char_type>& local_data) {
        if (comm.rank() == comm.size() - 1) {
            KASSERT(local_data.size() >= DC::X);
            local_data.resize(local_data.size() - DC::X);
        }
    }

    // revert changes made to local string by left shift
    void clean_up(std::vector<char_type>& local_string) {
        if (comm.rank() < comm.size() - 1) {
            local_string.resize(local_string.size() - DC::X + 1);
        }
        remove_padding(local_string);
    }

    // computes how many chars are at position with a remainder
    std::array<uint64_t, DC::X> compute_num_pos_mod() const {
        std::array<uint64_t, X> num_pos_mod;
        num_pos_mod.fill(0);
        for (uint64_t i = 0; i < X; i++) {
            num_pos_mod[i] = (total_chars + X - 1 - i) / X;
        }
        return num_pos_mod;
    }

    void dispatch_recursive_call(std::vector<RankIndex>& local_ranks, uint64_t last_rank) {
        auto map_back_func = [&](index_type sa_i) { return map_back(sa_i); };
        if (total_chars <= 80u) {
            // continue with sequential algorithm
            report_on_root("Sequential SA on local ranks",
                           comm,
                           recursion_depth,
                           config.print_phases);
            sequential_sa_on_local_ranks<char_type, index_type, DC>(local_ranks,
                                                                    local_sample_size,
                                                                    map_back_func,
                                                                    comm);
        } else {
// pick smallest data type that will fit
#ifdef OPTIMIZE_DATA_TYPES
            if (last_rank <= std::numeric_limits<uint8_t>::max()) {
                handle_recursive_call<uint8_t>(local_ranks, map_back_func);
            } else if (last_rank <= std::numeric_limits<uint16_t>::max()) {
                handle_recursive_call<uint16_t>(local_ranks, map_back_func);
            } else if (last_rank <= std::numeric_limits<uint32_t>::max()) {
                handle_recursive_call<uint32_t>(local_ranks, map_back_func);
            } else if (last_rank <= std::numeric_limits<dsss::uint40>::max()) {
                handle_recursive_call<uint40>(local_ranks, map_back_func);
            } else {
                print_on_root("Max Rank input size that can be handled is 2^40", comm);
            }
#else
            handle_recursive_call<uint40>(local_ranks, map_back_func);
#endif
        }
    }

    template <typename new_char_type>
    void handle_recursive_call(std::vector<RankIndex>& local_ranks, auto map_back_func) {
        // sort by (mod X, div X)
        timer.synchronize_and_start("phase_03_sort_mod_div");
        atomic_sorter.sort(local_ranks, RankIndex::cmp_mod_div);
        timer.stop();

        redistribute_if_imbalanced(local_ranks, 0.25);

        uint64_t after_discarding = num_ranks_after_discarding(local_ranks);
        uint64_t total_after_discarding = mpi_util::all_reduce_sum(after_discarding, comm);
        double reduction = ((double)total_after_discarding / total_sample_size);
        stats.discarding_reduction.push_back(reduction);

        bool use_discarding = reduction <= config.discarding_threshold;
        stats.use_discarding.push_back(use_discarding);
        if (use_discarding) {
            report_on_root("using discarding, reduction: " + std::to_string(reduction),
                           comm,
                           recursion_depth,
                           config.print_phases);
            recursive_call_with_discarding<new_char_type>(local_ranks, after_discarding);
        } else {
            recursive_call_direct<new_char_type>(local_ranks, map_back_func);
        }

        // sort samples by original index and distribute back to PEs
        timer.synchronize_and_start("phase_03_sort_ranks_index");
        atomic_sorter.sort(local_ranks, RankIndex::cmp_by_index);
        timer.stop();
        local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
    }

    template <typename new_char_type>
    void recursive_call_direct(std::vector<RankIndex>& local_ranks, auto map_back_func) {
        auto get_rank = [](RankIndex& r) -> new_char_type { return r.rank; };
        std::vector<new_char_type> recursive_string =
            extract_attribute<RankIndex, new_char_type>(local_ranks, get_rank);

        // TODO: is distirbute worth it?
        // recursive_string =
        //     mpi_util::distribute_data_custom(recursive_string, local_sample_size, comm);

        free_memory(local_ranks);

        // TODO: flexible selection of DC
        // create new instance of PDC3 with templates of new char type size
        PDCX<new_char_type, index_type, DC> rec_pdcx(config, comm);

        // memory of SA is counted in recursive call
        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;
        free_memory(recursive_string);

        auto index_function = [&](index_type index, index_type sa_at_i) {
            index_type global_index = map_back_func(sa_at_i);
            index_type rank = 1 + index;
            bool unique = true; // does not matter here
            return RankIndex(rank, global_index, unique);
        };

        local_ranks = mpi_util::zip_with_index<index_type, RankIndex>(SA, index_function, comm);
        free_memory(SA);
    }


    uint64_t num_ranks_after_discarding(std::vector<RankIndex>& local_ranks) {
        // all ranks can be dropped that are unique and are not needed to determine a not unique
        // rank

        // for simplicity we always keep the first element
        // and don't do shifts for the first and last elements
        uint64_t count_discarded = 0;
        for (uint64_t i = 1; i < local_ranks.size(); i++) {
            bool is_unique = local_ranks[i].unique;
            bool prev_is_unique = local_ranks[i - 1].unique;
            count_discarded += is_unique && prev_is_unique;
        }
        return local_ranks.size() - count_discarded;
    }

    template <typename new_char_type>
    void recursive_call_with_discarding(std::vector<RankIndex>& local_ranks,
                                        uint64_t after_discarding) {
        KASSERT(local_ranks.size() >= 0u);

        // build recursive string and discard ranks
        std::vector<new_char_type> recursive_string;
        std::vector<bool> red_pos_unique;
        recursive_string.reserve(after_discarding);
        red_pos_unique.reserve(after_discarding);

        // always keep first element
        recursive_string.push_back(local_ranks[0].rank);
        red_pos_unique.push_back(local_ranks[0].unique);
        for (uint64_t i = 1; i < local_ranks.size(); i++) {
            bool is_unique = local_ranks[i].unique;
            bool prev_is_unique = local_ranks[i - 1].unique;
            bool can_drop = is_unique && prev_is_unique;
            if (!can_drop) {
                recursive_string.push_back(local_ranks[i].rank);
                red_pos_unique.push_back(is_unique);
            }
        }

        // recursive call
        PDCX<new_char_type, index_type, DC> rec_pdcx(config, comm);
        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> reduced_SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;
        free_memory(recursive_string);

        // zip SA with 1, ..., n
        struct IndexRank {
            index_type index, rank;
            std::string to_string() const {
                return "(" + std::to_string(index) + ", " + std::to_string(rank) + ")";
            }
        };
        auto index_function = [&](uint64_t idx, index_type sa_index) {
            return IndexRank{sa_index, index_type(1 + idx)};
        };

        std::vector<IndexRank> ranks_sa =
            mpi_util::zip_with_index<index_type, IndexRank>(reduced_SA, index_function, comm);

        free_memory(reduced_SA);

        // invert reduced SA to get ranks
        auto cmp_index_sa = [](const IndexRank& l, const IndexRank& r) {
            return l.index < r.index;
        };

        timer.synchronize_and_start("phase_03_sort_index_sa");
        atomic_sorter.sort(ranks_sa, cmp_index_sa);
        timer.stop();

        // get ranks of recursive string that was generated locally on this PE
        ranks_sa = mpi_util::distribute_data_custom(ranks_sa, after_discarding, comm);

        // sort ranks and use second rank as a tie breaker
        struct RankRankIndex {
            index_type rank1, rank2, index;

            std::string to_string() const {
                return "(" + std::to_string(rank1) + ", " + std::to_string(rank2) + ", "
                       + std::to_string(index) + ")";
            }
        };
        auto cmp_rri = [](const RankRankIndex& l, const RankRankIndex& r) {
            if (l.rank1 != r.rank1) {
                return l.rank1 < r.rank1;
            }
            return l.rank2 < r.rank2;
        };

        uint64_t index_reduced = 0;
        auto get_next_rank = [&]() {
            while (red_pos_unique[index_reduced]) {
                KASSERT(index_reduced + 1 < red_pos_unique.size());
                index_reduced++;
            }
            return ranks_sa[index_reduced++].rank;
        };

        std::vector<RankRankIndex> rri;
        rri.reserve(local_ranks.size());
        for (uint64_t i = 0; i < local_ranks.size(); i++) {
            index_type rank1 = local_ranks[i].rank;
            index_type rank2 = local_ranks[i].unique ? index_type(0) : get_next_rank();
            index_type index = local_ranks[i].index;
            rri.emplace_back(rank1, rank2, index);
        }

        timer.synchronize_and_start("phase_03_sort_rri");
        atomic_sorter.sort(rri, cmp_rri);
        timer.stop();

        // extract local ranks
        auto index_local_ranks = [&](uint64_t idx, RankRankIndex& rr) {
            return RankIndex{index_type(1 + idx), rr.index, true};
        };

        local_ranks =
            mpi_util::zip_with_index<RankRankIndex, RankIndex>(rri, index_local_ranks, comm);
    }

    // compute even distributed splitters from sorted local samples
    std::vector<std::array<char_type, DC::X>>
    get_sample_splitters(std::vector<SampleString>& local_samples, uint64_t blocks) {
        using Substring = std::array<char_type, DC::X>;
        int64_t num_samples = local_samples.size();
        int64_t samples_before = mpi_util::ex_prefix_sum(num_samples, comm);
        std::vector<Substring> local_splitters;
        for (uint64_t i = 0; i < blocks - 1; i++) {
            int64_t global_index = (i + 1) * total_sample_size / blocks;
            int64_t x = global_index - samples_before;
            if (x >= 0 && x < num_samples) {
                local_splitters.push_back(local_samples[x].letters);
            }
        }
        std::vector<Substring> global_splitters = comm.allgatherv(send_buf(local_splitters));
        return global_splitters;
    }

    template <typename T>
    bool redistribute_if_imbalanced(std::vector<T>& data, double min_imbalance) {
        double imbalance = compute_min_imbalance(data.size(), comm);
        if (imbalance <= min_imbalance) {
            data = mpi_util::distribute_data(data, comm);
            return true;
        }
        return false;
    }

    std::vector<index_type> compute_sa(std::vector<char_type>& local_string) {
        timer.synchronize_and_start("pdcx");

        const int process_rank = comm.rank();
        double min_imbalance = 0.25;

        //******* Start Phase 0: Preparation  ********
        timer.synchronize_and_start("phase_00_preparation");

        bool redist_chars = redistribute_if_imbalanced(local_string, min_imbalance);
        stats.redistribute_chars.push_back(redist_chars);

        // figure out lengths of the other strings
        auto chars_at_proc = comm.allgather(send_buf(local_string.size()));
        total_chars = std::accumulate(chars_at_proc.begin(), chars_at_proc.end(), 0);
        local_chars = chars_at_proc[process_rank];

        std::string msg_level = "Recursion Level: " + std::to_string(recursion_depth)
                                + ", Total Chars: " + std::to_string(total_chars);
        report_on_root(msg_level, comm, recursion_depth, config.print_phases);
        report_on_root("Phase 0: Preparation", comm, recursion_depth, config.print_phases);

        // number of chars before processor i
        std::vector<uint64_t> chars_before(comm.size());
        std::exclusive_scan(chars_at_proc.begin(), chars_at_proc.end(), chars_before.begin(), 0);

        // number of positions with mod X = d
        std::array<uint64_t, X> num_at_mod = compute_num_pos_mod();
        const uint64_t rem = total_chars % X;
        bool added_dummy = is_in_dc<DC>(rem);
        num_at_mod[rem] += added_dummy;

        // inclusive prefix sum to compute map back
        samples_before[0] = 0;
        for (uint i = 1; i < D + 1; i++) {
            uint d = DC::DC[i - 1];
            samples_before[i] = samples_before[i - 1] + num_at_mod[d];
        }
        total_sample_size = samples_before.back();
        add_padding(local_string);

        // logging
        stats.algo = "DC" + std::to_string(X);
        stats.num_processors = comm.size();
        stats.max_depth = std::max(stats.max_depth, recursion_depth);
        stats.string_sizes.push_back(total_chars);
        stats.local_string_sizes.push_back(local_string.size());
        stats.char_type_used.push_back(8 * sizeof(char_type));
        timer.stop();

        // solve sequentially on root to avoid corner cases with empty PEs
        if (total_chars <= comm.size() * 2 * X) {
            report_on_root("Solve SA sequentially on root",
                           comm,
                           recursion_depth,
                           config.print_phases);
            remove_padding(local_string);
            std::vector<index_type> local_SA =
                compute_sa_on_root<char_type, index_type>(local_string, comm);
            timer.stop(); // pdcx
            return local_SA;
        }
        //******* End Phase 0: Preparation  ********

        //******* Start Phase 1: Construct Samples  ********
        report_on_root("Phase 1: Sort Samples", comm, recursion_depth, config.print_phases);
        timer.synchronize_and_start("phase_01_samples");

        SampleStringPhase<char_type, index_type, DC> phase1(comm);
        phase1.shift_chars_left(local_string);
        std::vector<SampleString> local_samples =
            phase1.compute_sample_strings(local_string, chars_before[process_rank]);
        local_sample_size = local_samples.size();
        phase1.sort_samples(local_samples, atomic_sorter);
        bool redist_samples = redistribute_if_imbalanced(local_samples, min_imbalance);
        stats.redistribute_samples.push_back(redist_samples);
        timer.stop();

        // get splitters from sorted sample sequence
        bool use_space_efficient_sort = config.blocks_space_efficient_sort > 1
                                        && total_chars >= config.threshold_space_efficient_sort;
        uint64_t blocks = config.blocks_space_efficient_sort;
        stats.space_efficient_sort.push_back(use_space_efficient_sort);
        std::vector<std::array<char_type, DC::X>> global_samples_splitters;
        if (use_space_efficient_sort) {
            global_samples_splitters = get_sample_splitters(local_samples, blocks);
        }

        //******* End Phase 1: Construct Samples  ********

        //******* Start Phase 2: Construct Ranks  ********
        report_on_root("Phase 2: Construct Ranks", comm, recursion_depth, config.print_phases);
        timer.synchronize_and_start("phase_02_ranks");

        LexicographicRankPhase<char_type, index_type, DC> phase2(comm);
        phase2.shift_samples_left(local_samples);
        std::vector<RankIndex> local_ranks = phase2.compute_lexicographic_ranks(local_samples);
        phase2.flag_unique_ranks(local_ranks);
        free_memory(local_samples);

        timer.stop();
        //******* End Phase 2: Construct Ranks  ********

        //******* Start Phase 3: Recursive Call  ********
        report_on_root("Phase 3: Recursion", comm, recursion_depth, config.print_phases);

        timer.synchronize_and_start("phase_03_recursion");

        index_type last_rank = local_ranks.empty() ? index_type(0) : local_ranks.back().rank;
        comm.bcast_single(send_recv_buf(last_rank), root(comm.size() - 1));
        stats.highest_ranks.push_back(last_rank);
        bool chars_distinct = last_rank >= index_type(total_sample_size);

        if (chars_distinct) {
            timer.synchronize_and_start("phase_03_sort_index_base");
            atomic_sorter.sort(local_ranks, RankIndex::cmp_by_index);
            timer.stop();

            local_ranks = mpi_util::distribute_data_custom(local_ranks, local_sample_size, comm);
            local_ranks.shrink_to_fit();

        } else {
            dispatch_recursive_call(local_ranks, last_rank);
        }
        timer.stop();
        //******* End Phase 3: Recursive Call  ********

        //******* Start Phase 4: Merge Suffixes  ********
        report_on_root("Phase 4: Merge Suffixes", comm, recursion_depth, config.print_phases);
        timer.synchronize_and_start("phase_04_merge");

        MergeSamplePhase<char_type, index_type, DC> phase4(comm);
        phase4.shift_ranks_left(local_ranks);
        phase4.push_padding(local_ranks, total_chars);

        std::vector<index_type> local_SA;
        if (use_space_efficient_sort) {
            std::vector<uint64_t> bucket_sizes =
                phase4.compute_bucket_sizes(local_string, local_chars, global_samples_splitters);

            local_SA = phase4.space_effient_sort_SA(local_string,
                                                    local_ranks,
                                                    bucket_sizes,
                                                    chars_before[comm.rank()],
                                                    local_chars,
                                                    global_samples_splitters,
                                                    atomic_sorter);
            uint64_t largest_bucket = *std::max_element(bucket_sizes.begin(), bucket_sizes.end());
            double avg_buckets = (double)total_chars / (blocks * comm.size());
            double bucket_imbalance = ((double)largest_bucket / avg_buckets) - 1.0;
            double max_bucket_imbalance =
                mpi_util::all_reduce(bucket_imbalance, ops::max<>{}, comm);
            stats.bucket_imbalance.push_back(max_bucket_imbalance);
            // print_concatenated(bucket_sizes, comm, "bucket_sizes");
            // print_concatenated_string(std::to_string(local_SA.size()), comm, "local_SA.size()");
        } else {
            std::vector<MergeSamples> merge_samples =
                phase4.construct_merge_samples(local_string,
                                               local_ranks,
                                               chars_before[process_rank],
                                               chars_at_proc[process_rank]);

            free_memory(local_ranks);
            phase4.sort_merge_samples(merge_samples, atomic_sorter);
            local_SA = phase4.extract_SA(merge_samples);
        }

        timer.stop();
        //******* End Phase 4: Merge Suffixes  ********

        // logging
        stats.string_imbalance.push_back(compute_max_imbalance(local_chars, comm));
        stats.sample_imbalance.push_back(compute_max_imbalance(local_sample_size, comm));
        stats.sa_imbalance.push_back(compute_max_imbalance(local_SA.size(), comm));
        if (recursion_depth == 0) {
            std::reverse(stats.string_imbalance.begin(), stats.string_imbalance.end());
            std::reverse(stats.sample_imbalance.begin(), stats.sample_imbalance.end());
            std::reverse(stats.sa_imbalance.begin(), stats.sa_imbalance.end());
            std::reverse(stats.bucket_imbalance.begin(), stats.bucket_imbalance.end());
        }

        clean_up(local_string);

        timer.stop(); // pdcx
        return local_SA;
    }

    void report_time() {
        comm.barrier();
        // timer.aggregate_and_print(measurements::SimpleJsonPrinter<>{});
        timer.aggregate_and_print(measurements::FlatPrinter{});
        comm.barrier();
    }

    void report_stats() {
        comm.barrier();
        if (comm.rank() == comm.size() - 1) {
            stats.print();
        }
        comm.barrier();
    }

    void reset() {
        stats.reset();
        recursion_depth = 0;
        timer.clear();
    }

    constexpr static uint32_t X = DC::X;
    constexpr static uint32_t D = DC::D;
    constexpr static bool DBG = false;
    constexpr static bool use_recursion = true;

    uint64_t local_sample_size;
    uint64_t total_sample_size;
    uint64_t local_chars;
    uint64_t total_chars;
    std::array<index_type, DC::D + 1> samples_before;

    PDCXConfig& config;
    mpi::SortingWrapper atomic_sorter;

    Communicator<>& comm;
    measurements::Timer<Communicator<>>& timer;
    Statistics& stats;
    int recursion_depth;
};

} // namespace dsss::dcx
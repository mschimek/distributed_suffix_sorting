#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
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
#include "mpi/shift.hpp"
#include "mpi/stats.hpp"
#include "mpi/zip.hpp"
#include "pdcx/compute_ranks.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/merge_samples.hpp"
#include "pdcx/redistribute.hpp"
#include "pdcx/sample_string.hpp"
#include "pdcx/sequential_sa.hpp"
#include "pdcx/space_efficient_sort.hpp"
#include "pdcx/statistics.hpp"
#include "sa_check.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/char_container.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"
#include "util/uint_types.hpp"


namespace dsss::dcx {

using namespace kamping;


template <typename char_type,
          typename index_type,
          typename DC,
          typename CharContainerSamples = CharArray<char_type, DC::X + 1>,
          typename CharContainerMerging = CharArray<char_type, DC::X>>
class PDCX {
    using SampleString = DCSampleString<char_type, index_type, DC, CharContainerSamples>;
    using RankIndex = DCRankIndex<char_type, index_type, DC>;
    using MergeSamples = DCMergeSamples<char_type, index_type, DC, CharContainerMerging>;

    using SamplePhase = SampleStringPhase<char_type, index_type, DC, CharContainerSamples>;
    using RankPhase = LexicographicRankPhase<char_type, index_type, DC, SampleString>;
    using MergePhase = MergeSamplePhase<char_type, index_type, DC, CharContainerMerging>;

public:
    PDCX(PDCXConfig const& _config, Communicator<>& _comm)
        : config(_config),
          atomic_sorter(_comm),
          comm(_comm),
          timer(measurements::timer()),
          stats(get_stats_instance()),
          recursion_depth(0) {
        atomic_sorter.set_sorter(config.atomic_sorter);
        atomic_sorter.set_num_levels(config.ams_levels);
        atomic_sorter.set_sample_sort_config(config.sample_sort_config);
        atomic_sorter.finalize_setting();
        string_sorter_samples.set_memory(config.memory_seq_string_sorter);
        string_sorter_merging.set_memory(config.memory_seq_string_sorter);
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

    void remove_padding(std::vector<char_type>& local_data) {
        if (comm.rank() == comm.size() - 1) {
            KASSERT(local_data.size() >= DC::X);
            local_data.resize(info.local_chars);
        }
    }

    // revert changes made to local string by left shift
    void clean_up(std::vector<char_type>& local_string) {
        if (comm.rank() < comm.size() - 1) {
            local_string.resize(info.local_chars);
        }
        remove_padding(local_string);
    }


    void dispatch_recursive_call(std::vector<RankIndex>& local_ranks, uint64_t last_rank) {
        auto map_back_func = [&](index_type sa_i) { return map_back(sa_i); };
        // if (total_chars <= 80u) {
        if (info.total_chars <= 80u) {
            // continue with sequential algorithm
            report_on_root("Sequential SA on local ranks",
                           comm,
                           recursion_depth,
                           config.print_phases);
            sequential_sa_on_local_ranks<char_type, index_type, DC>(local_ranks,
                                                                    info.local_sample_size,
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

        redistribute_if_imbalanced(local_ranks, config.min_imbalance, comm);

        uint64_t const after_discarding = num_ranks_after_discarding(local_ranks);
        uint64_t const total_after_discarding = mpi_util::all_reduce_sum(after_discarding, comm);
        double const reduction = ((double)total_after_discarding / info.total_sample_size);
        stats.discarding_reduction.push_back(reduction);
        get_local_stats_instance().discarding_reduction.push_back(reduction);

        bool const use_discarding = reduction <= config.discarding_threshold;
        stats.use_discarding.push_back(use_discarding);
        get_local_stats_instance().use_discarding.push_back(use_discarding);
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
        local_ranks = mpi_util::distribute_data_custom(local_ranks, info.local_sample_size, comm);
    }

    template <typename new_char_type>
    void recursive_call_direct(std::vector<RankIndex>& local_ranks, auto map_back_func) {
        auto get_rank = [](RankIndex& r) -> new_char_type { return r.rank; };
        std::vector<new_char_type> recursive_string =
            extract_attribute<RankIndex, new_char_type>(local_ranks, get_rank);

        free_memory(std::move(local_ranks));

        // TODO: flexible selection of DC

        // create new instance of PDC3 with templates of new char type size
        PDCX<new_char_type, index_type, DC> rec_pdcx(config, comm);

        // memory of SA is counted in recursive call

        recursion_depth++;
        rec_pdcx.recursion_depth = recursion_depth;
        std::vector<index_type> SA = rec_pdcx.compute_sa(recursive_string);
        recursion_depth--;
        free_memory(std::move(recursive_string));

        auto index_function = [&](index_type index, index_type sa_at_i) {
            index_type global_index = map_back_func(sa_at_i);
            index_type rank = 1 + index;
            bool unique = true; // does not matter here
            return RankIndex(rank, global_index, unique);
        };

        local_ranks = mpi_util::zip_with_index<index_type, RankIndex>(SA, index_function, comm);
        free_memory(std::move(SA));
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
        KASSERT(local_ranks.size() > 0u);

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
        free_memory(std::move(recursive_string));


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

        free_memory(std::move(reduced_SA));

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

        // extract local ranks
        auto index_local_ranks = [&](uint64_t idx, RankRankIndex& rr) {
            return RankIndex{index_type(1 + idx), rr.index, true};
        };

        uint64_t buckets = config.buckets_phase3;
        uint64_t total_random_samples = config.num_samples_phase3;

        if (buckets == 1 || recursion_depth > 0) {
            std::vector<RankRankIndex> rri;
            rri.reserve(local_ranks.size());
            for (uint64_t i = 0; i < local_ranks.size(); i++) {
                index_type rank1 = local_ranks[i].rank;
                index_type rank2 = local_ranks[i].unique ? index_type(0) : get_next_rank();
                index_type index = local_ranks[i].index;
                rri.emplace_back(rank1, rank2, index);
            }

            free_memory(std::move(local_ranks));
            free_memory(std::move(ranks_sa));

            timer.synchronize_and_start("phase_03_sort_rri");
            atomic_sorter.sort(rri, cmp_rri);
            timer.stop();

            local_ranks =
                mpi_util::zip_with_index<RankRankIndex, RankIndex>(rri, index_local_ranks, comm);
        } else {
            report_on_root("Space efficient sort in Phase 3 with " + std::to_string(buckets)
                               + " buckets",
                           comm);

            // space efficient variant
            SpaceEfficientSort<char_type, index_type, DC> helper(comm, config);
            using SA = std::vector<index_type>;

            // 1. collect reduced index with dummy elements for easier random access
            std::vector<index_type> ranks_sa_with_dummy;
            ranks_sa_with_dummy.reserve(local_ranks.size());
            for (uint64_t i = 0; i < local_ranks.size(); i++) {
                index_type rank = local_ranks[i].unique ? index_type(0) : get_next_rank();
                ranks_sa_with_dummy.emplace_back(rank);
            }
            free_memory(std::move(ranks_sa));
            free_memory(std::move(red_pos_unique));


            // 2. determine splitters
            auto get_element_at = [&](uint64_t i) {
                return RankRankIndex{local_ranks[i].rank,
                                     ranks_sa_with_dummy[i],
                                     local_ranks[i].index};
            };
            uint64_t local_size = local_ranks.size();
            uint64_t total_size = mpi_util::all_reduce_sum(local_size, comm);

            bool use_rquick = config.sample_sort_config.use_rquick_for_splitters;
            std::vector<RankRankIndex> splitters =
                helper.template general_random_sample_splitters<RankRankIndex>(get_element_at,
                                                                               cmp_rri,
                                                                               local_size,
                                                                               buckets,
                                                                               total_random_samples,
                                                                               use_rquick);


            // 3. determine block mapping
            auto [bucket_sizes, sample_to_bucket] =
                helper.compute_sample_to_block_mapping(get_element_at,
                                                       cmp_rri,
                                                       local_size,
                                                       splitters);


            // 4. report bucket imbalance
            double bucket_imbalance = get_imbalance_bucket(bucket_sizes, total_size, comm);
            report_on_root("--> Bucket Imbalance " + std::to_string(bucket_imbalance),
                           comm,
                           info.recursion_depth);
            get_local_stats_instance().input_bucket_imbalance_rank_computation.push_back(
                get_max_local_imbalance(bucket_sizes, total_size, comm.size()));

            // 5. allocate datastructures to store results
            std::vector<index_type> concat_sa_buckets;
            std::vector<uint64_t> sa_bucket_size;
            sa_bucket_size.reserve(buckets);
            std::vector<RankRankIndex> block_rri;

            uint64_t estimated_size = (total_size / comm.size()) * 1.03;
            concat_sa_buckets.reserve(estimated_size);

            // 6. materialize data in k rounds
            for (uint64_t k = 0; k < buckets; k++) {
                timer.synchronize_and_start("phase_03_space_efficient_sort_collect_bucket");

                // collect samples falling into kth block
                block_rri.reserve(bucket_sizes[k]);
                for (uint64_t idx = 0; idx < local_size; idx++) {
                    if (sample_to_bucket[idx] == k) {
                        RankRankIndex element = get_element_at(idx);
                        block_rri.emplace_back(element);
                    }
                }
                timer.stop();
                KASSERT(bucket_sizes[k] == block_rri.size());

                timer.synchronize_and_start("phase_03_space_efficient_sort_rri");
                atomic_sorter.sort(block_rri, cmp_rri);
                timer.stop();

                // extract SA of block
                for (auto& r: block_rri) {
                    concat_sa_buckets.push_back(r.index);
                }
                sa_bucket_size.push_back(block_rri.size());
                block_rri.clear();
            }
            // log imbalance of received suffixes
            double bucket_imbalance_received =
                get_imbalance_bucket(sa_bucket_size, total_size, comm);
            report_on_root("--> Bucket Imbalance Received "
                               + std::to_string(bucket_imbalance_received),
                           comm,
                           info.recursion_depth);
            get_local_stats_instance().output_bucket_imbalance_rank_computation.push_back(
                get_max_local_imbalance(sa_bucket_size, total_size, comm.size()));

            timer.synchronize_and_start("phase_03_space_efficient_sort_alltoall");
            SA local_SA = mpi_util::transpose_blocks_wrapper(concat_sa_buckets,
                                                             sa_bucket_size,
                                                             comm,
                                                             config.rearrange_buckets_balanced);
            timer.stop();

            // zip SA with 1, ..., n
            auto get_index_local_ranks = [&](uint64_t idx, index_type& index) {
                return RankIndex{index_type(1 + idx), index, true};
            };
            local_ranks = mpi_util::zip_with_index<index_type, RankIndex>(local_SA,
                                                                          get_index_local_ranks,
                                                                          comm);
        }
    }

    // computes how many chars are at position with a remainder
    std::array<uint64_t, DC::X> compute_num_pos_mod(uint64_t total_chars) const {
        std::array<uint64_t, X> num_pos_mod;
        num_pos_mod.fill(0);
        for (uint64_t i = 0; i < X; i++) {
            num_pos_mod[i] = (total_chars + X - 1 - i) / X;
        }
        return num_pos_mod;
    }
    PDCXLengthInfo compute_length_info(std::vector<char_type>& local_string) {
        // compute length information
        PDCXLengthInfo info;
        info.local_chars = local_string.size();
        info.total_chars = mpi_util::all_reduce_sum(local_string.size(), comm);
        info.largest_char = mpi_util::all_reduce_max(local_string, comm);
        info.chars_before = mpi_util::ex_prefix_sum(local_string.size(), comm);
        info.recursion_depth = recursion_depth;

        const uint64_t rem = info.total_chars % X;
        bool added_dummy = is_in_dc<DC>(rem);
        bool added_dummy_to_pe = added_dummy && (comm.rank() == comm.size() - 1);

        uint64_t local_sample_size = 0;
        uint64_t offset = info.chars_before % X;
        for (uint64_t i = 0; i < local_string.size(); i++) {
            uint64_t m = (i + offset) % X;
            local_sample_size += is_in_dc<DC>(m);
        }
        info.local_chars_with_dummy = info.local_chars + added_dummy_to_pe;

        local_sample_size += added_dummy_to_pe;
        info.local_sample_size = local_sample_size;
        info.total_sample_size = mpi_util::all_reduce_sum(local_sample_size, comm);
        info.samples_before = mpi_util::ex_prefix_sum(local_sample_size, comm);

        // number of positions with mod X = d
        std::array<uint64_t, X> num_at_mod = compute_num_pos_mod(info.total_chars);
        num_at_mod[rem] += added_dummy;

        // inclusive prefix sum to compute map back
        samples_before[0] = 0;
        for (uint i = 1; i < D + 1; i++) {
            uint d = DC::DC[i - 1];
            samples_before[i] = samples_before[i - 1] + num_at_mod[d];
        }
        return info;
    }


    void report_max_mem(std::string name) {
        // temporary
        // return;
        std::replace(name.begin(), name.end(), ' ', '_');
        name = "DEBUG_PE_MEM_" + name;
        if (recursion_depth == 0) {
            uint64_t max_mem = dsss::get_max_mem_bytes();
            auto all_mem = comm.allgather(kamping::send_buf(max_mem));
            if (comm.rank() == 0) {
                std::cout << name << "=";
                kamping::print_vector(all_mem, ",");
                std::cout << std::endl;
            }
        }
    }

    std::vector<index_type> compute_sa(std::vector<char_type>& local_string) {
        uint64_t max_mem_pdcx_start = dsss::get_max_mem_bytes();
        auto all_max_mem_pdcx_start = comm.allgather(kamping::send_buf(max_mem_pdcx_start));
        if (recursion_depth == 0 && comm.rank() == 0) {
            std::cout << "max_mem_pdcx_start=";
            kamping::print_vector(all_max_mem_pdcx_start, ",");
        }

        timer.synchronize_and_start("pdcx");

        if constexpr (DEBUG_SIZE)
            print_concatenated_size(local_string, comm, "local_string");


        //******* Start Phase 0: Preparation  ********
        timer.synchronize_and_start("phase_00_preparation");

        bool redist_chars = redistribute_if_imbalanced(local_string, config.min_imbalance, comm);
        stats.redistribute_chars.push_back(redist_chars);
        get_local_stats_instance().redistribute_chars.push_back(redist_chars);


        info = compute_length_info(local_string);

        std::string msg_level = "Recursion Level: " + std::to_string(recursion_depth)
                                + ", Total Chars: " + std::to_string(info.total_chars);
        report_on_root(msg_level, comm, recursion_depth, config.print_phases);
        report_on_root("Phase 0: Preparation", comm, recursion_depth, config.print_phases);

        // configure packing
        const bool use_packed_samples = config.use_char_packing_samples && recursion_depth == 0;
        const bool use_packed_merging = config.use_char_packing_merging && recursion_depth == 0;

        // configure string sorting algorithm, can use radix sort only for small chars
        dsss::SeqStringSorter string_algo =
            info.largest_char < 256 ? config.string_sorter : dsss::SeqStringSorter::MultiKeyQSort;
        string_sorter_samples.set_sorter(string_algo);
        string_sorter_merging.set_sorter(string_algo);

        // when we use packing, character values get to large for radix sort
        if (use_packed_samples && sizeof(char_type) > 1) {
            string_sorter_samples.set_sorter(dsss::SeqStringSorter::MultiKeyQSort);
        }
        if (use_packed_merging && sizeof(char_type) > 1) {
            string_sorter_merging.set_sorter(dsss::SeqStringSorter::MultiKeyQSort);
        }

        // configure space efficient sort
        SpaceEfficientSort<char_type, index_type, DC> space_efficient_sort(comm, config);
        uint64_t buckets_samples = config.buckets_samples_at_level(recursion_depth);
        uint64_t buckets_merging = config.buckets_merging_at_level(recursion_depth);
        const bool use_bucket_sorting_samples = buckets_samples > 1;
        const bool use_bucket_sorting_merging = buckets_merging > 1;
        const bool too_many_buckets_samples =
            buckets_samples > comm.size() && !config.rearrange_buckets_balanced;
        const bool too_many_buckets_merging =
            buckets_merging > comm.size() && !config.rearrange_buckets_balanced;
        if (use_bucket_sorting_samples && too_many_buckets_samples) {
            buckets_samples = comm.size();
            report_on_root(
                "Warning: #buckets_samples > #PEs, setting blocks to #PEs. Set "
                "--rearrange_buckets_balanced (-E) flag to support more buckets than PEs.",
                comm,
                recursion_depth,
                config.print_phases);
        }
        if (use_bucket_sorting_merging && too_many_buckets_merging) {
            buckets_merging = comm.size();
            report_on_root(
                "Warning: #buckets_merging > #PEs, setting blocks to #PEs. Set "
                "--rearrange_buckets_balanced (-E) flag to support more buckets than PEs.",
                comm,
                recursion_depth,
                config.print_phases);
        }

        // logging
        stats.algo = "DC" + std::to_string(X);
        stats.num_processors = comm.size();
        stats.max_depth = std::max(stats.max_depth, recursion_depth);
        stats.local_string_sizes.push_back(info.local_chars);
        stats.string_sizes.push_back(info.total_chars);
        stats.char_type_used.push_back(8 * sizeof(char_type));
        get_local_stats_instance().algo = "DC" + std::to_string(X);
        get_local_stats_instance().num_processors = comm.size();
        get_local_stats_instance().max_depth = std::max(stats.max_depth, recursion_depth);
        get_local_stats_instance().local_text_size.push_back(info.local_chars);
        get_local_stats_instance().text_size.push_back(info.total_chars);
        get_local_stats_instance().char_type_used.push_back(8 * sizeof(char_type));

        timer.stop();
        //******* End Phase 0: Preparation  ********

        // solve sequentially on root to avoid corner cases with empty PEs
        if (info.total_chars <= std::max(static_cast<std::uint64_t>(comm.size() * 2u * X),
                                         static_cast<std::uint64_t>(10'000u))) {
            report_on_root("Solve SA sequentially on root",
                           comm,
                           recursion_depth,
                           config.print_phases);
            std::vector<index_type> local_SA =
                compute_sa_on_root<char_type, index_type>(local_string, comm);
            timer.stop(); // pdcx
            return local_SA;
        }

        std::vector<RankIndex> local_ranks;
        std::vector<SampleString> global_samples_splitters;
        SamplePhase phase1(comm, config, info, atomic_sorter, string_sorter_samples);

        // add a padding of zeros to local string taking into account char packing
        const double char_packing_ratio_samples = use_packed_samples ? config.packing_ratio : 1;
        const double char_packing_ratio_merging = use_packed_merging ? config.packing_ratio : 1;
        const double char_packing_ratio =
            std::max(char_packing_ratio_samples, char_packing_ratio_merging);
        if (info.recursion_depth == 0) {
            get_stats_instance().packed_chars_samples.push_back(char_packing_ratio_samples);
            get_stats_instance().packed_chars_merging.push_back(char_packing_ratio_merging);
            get_local_stats_instance().packed_chars_samples.push_back(char_packing_ratio_samples);
            get_local_stats_instance().packed_chars_merging.push_back(char_packing_ratio_merging);
        }
        phase1.make_padding_and_shifts(local_string, char_packing_ratio);

        if (use_bucket_sorting_samples) {
            //******* Start Phase 1 + 2: Construct Samples +   Construct Ranks********
            report_on_root("Phase 1 + 2: Sort Samples + Compute Ranks with "
                               + std::to_string(buckets_samples) + " buckets.",
                           comm,
                           recursion_depth,
                           config.print_phases);
            timer.synchronize_and_start("phase_01_02_samples_ranks");
            RankPhase phase2(comm, config, info);
            if (config.use_randomized_chunks) {
                report_on_root("using randomized chunks for Phase 1 + 2", comm);
                local_ranks = phase2.create_ranks_space_efficient_chunking(phase1,
                                                                           local_string,
                                                                           buckets_samples,
                                                                           use_packed_samples);
            } else {
                local_ranks = phase2.create_ranks_space_efficient(phase1,
                                                                  local_string,
                                                                  buckets_samples,
                                                                  use_packed_samples);
            }
            timer.stop();

            if (recursion_depth == 0) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                auto all_mem = comm.allgather(kamping::send_buf(max_mem));
                get_stats_instance().max_mem_pe_phase_01 = all_mem;
                get_stats_instance().max_mem_pe_phase_02 = all_mem;
            }
            if (recursion_depth == 1) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                auto all_mem = comm.allgather(kamping::send_buf(max_mem));
                get_stats_instance().max_mem_pe_phase_01_1 = all_mem;
                get_stats_instance().max_mem_pe_phase_02_1 = all_mem;
            }

            //******* End Phase 1 + 2: Construct Samples +   Construct Ranks********
        } else {
            //******* Start Phase 1: Construct Samples  ********
            report_on_root("Phase 1: Sort Samples", comm, recursion_depth, config.print_phases);
            timer.synchronize_and_start("phase_01_samples");
            std::vector<SampleString> local_samples =
                phase1.sorted_dc_samples(local_string, use_packed_samples);
            timer.stop();

            if constexpr (DEBUG_SIZE)
                print_concatenated_size(local_samples, comm, "local_samples");

            // get splitters from sorted sample sequence
            if (use_bucket_sorting_merging && !config.use_random_sampling_splitters) {
                global_samples_splitters =
                    space_efficient_sort.get_uniform_splitters(local_samples, buckets_merging);
            }
            if (recursion_depth == 0) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                get_stats_instance().max_mem_pe_phase_01 =
                    comm.allgather(kamping::send_buf(max_mem));
            }
            if (recursion_depth == 1) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                get_stats_instance().max_mem_pe_phase_01_1 =
                    comm.allgather(kamping::send_buf(max_mem));
            }
            //******* End Phase 1: Construct Samples  ********

            //******* Start Phase 2: Construct Ranks  ********
            report_on_root("Phase 2: Construct Ranks", comm, recursion_depth, config.print_phases);
            timer.synchronize_and_start("phase_02_ranks");
            RankPhase phase2(comm, config, info);
            local_ranks = phase2.create_lexicographic_ranks(local_samples);
            free_memory(std::move(local_samples));
            timer.stop();
            if (recursion_depth == 0) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                get_stats_instance().max_mem_pe_phase_02 =
                    comm.allgather(kamping::send_buf(max_mem));
            }
            if (recursion_depth == 1) {
                uint64_t max_mem = dsss::get_max_mem_bytes();
                get_stats_instance().max_mem_pe_phase_02_1 =
                    comm.allgather(kamping::send_buf(max_mem));
            }

            //******* End Phase 2: Construct Ranks  ********
        }

        //******* Start Phase 3: Recursive Call  ********
        report_on_root("Phase 3: Recursion", comm, recursion_depth, config.print_phases);

        timer.synchronize_and_start("phase_03_recursion");

        index_type last_rank = local_ranks.empty() ? index_type(0) : local_ranks.back().rank;
        comm.bcast_single(send_recv_buf(last_rank), root(comm.size() - 1));
        stats.highest_ranks.push_back(last_rank);
        get_local_stats_instance().highest_ranks.push_back(last_rank);
        bool chars_distinct = last_rank >= index_type(info.total_sample_size);

        if (chars_distinct) {
            timer.synchronize_and_start("phase_03_sort_index_base");
            atomic_sorter.sort(local_ranks, RankIndex::cmp_by_index);
            timer.stop();

            local_ranks =
                mpi_util::distribute_data_custom(local_ranks, info.local_sample_size, comm);
            local_ranks.shrink_to_fit();

        } else {
            dispatch_recursive_call(local_ranks, last_rank);
        }
        timer.stop();
        if (recursion_depth == 0) {
            uint64_t max_mem = dsss::get_max_mem_bytes();
            get_stats_instance().max_mem_pe_phase_03 = comm.allgather(kamping::send_buf(max_mem));
        }
        if (recursion_depth == 1) {
            uint64_t max_mem = dsss::get_max_mem_bytes();
            get_stats_instance().max_mem_pe_phase_03_1 = comm.allgather(kamping::send_buf(max_mem));
        }
        if constexpr (DEBUG_SIZE)
            print_concatenated_size(local_ranks, comm, "local_ranks");


        //******* End Phase 3: Recursive Call  ********

        //******* Start Phase 4: Merge Suffixes  ********
        report_on_root("Phase 4: Merge Suffixes", comm, recursion_depth, config.print_phases);
        timer.synchronize_and_start("phase_04_merge");

        auto get_rank = [](RankIndex& r) { return r.rank; };
        std::vector<index_type> sample_ranks =
            extract_attribute<RankIndex, index_type>(local_ranks, get_rank);
        free_memory(std::move(local_ranks));

        MergePhase phase4(comm, config, info, atomic_sorter, string_sorter_merging);
        phase4.shift_ranks_left(sample_ranks);
        phase4.push_padding(sample_ranks);

        using MergePhaseSplitter = typename MergePhase::SplitterType;

        std::vector<index_type> local_SA;
        std::vector<MergePhaseSplitter> bucket_splitter;
        if (use_bucket_sorting_merging) {
            // determine splitters for buckets
            double char_packing_ratio = use_packed_merging ? config.packing_ratio : 1;
            auto materialize_chars = [&](std::vector<char_type>& local_string, uint64_t char_pos) {
                return phase4.materialize_characters(local_string, char_pos, char_packing_ratio);
            };
            auto get_merge_splitter_at = [&](uint64_t local_index) {
                return phase4.materialize_merge_sample_at(local_string,
                                                          sample_ranks,
                                                          local_index,
                                                          materialize_chars);
            };
            auto cmp_splitters = phase4.get_splitter_comparator();

            if (config.use_random_sampling_splitters || global_samples_splitters.empty()) {
                report_on_root("--> using random samples for merging",
                               comm,
                               recursion_depth,
                               config.print_phases);
                SpaceEfficientSort<char_type, index_type, DC> space_efficient_sort(comm, config);
                timer.synchronize_and_start("phase_04_random_sample_splitters");
                uint64_t num_samples = config.num_samples_splitters;
                bool use_rquick = config.sample_sort_config.use_rquick_for_splitters;
                bucket_splitter = space_efficient_sort
                                      .template general_random_sample_splitters<MergePhaseSplitter>(
                                          get_merge_splitter_at,
                                          cmp_splitters,
                                          info.local_chars,
                                          buckets_merging,
                                          num_samples,
                                          use_rquick);
                timer.stop();
            } else {
                // convert uniform splitters
                report_on_root("--> using uniform samples for merging",
                               comm,
                               recursion_depth,
                               config.print_phases);

                std::vector<MergePhaseSplitter> local_splitters;
                for (auto& s: global_samples_splitters) {
                    int64_t local_index = int64_t(s.index) - int64_t(info.chars_before);
                    // convert splitters that can be converted with local data
                    if (local_index >= 0 && local_index < int64_t(info.local_chars)) {
                        MergePhaseSplitter splitter = get_merge_splitter_at(uint64_t(local_index));
                        local_splitters.push_back(splitter);
                    }
                }
                // splitters might not be in sorted order after redistributing the samples
                bucket_splitter = comm.allgatherv(send_buf(local_splitters));
                ips4o::sort(bucket_splitter.begin(), bucket_splitter.end(), cmp_splitters);
                free_memory(std::move(global_samples_splitters));
            }
            KASSERT(bucket_splitter.size() == buckets_merging - 1);
            KASSERT(std::is_sorted(bucket_splitter.begin(), bucket_splitter.end(), cmp_splitters));

            report_on_root("Using " + std::to_string(buckets_merging) + " buckets for merging.",
                           comm,
                           recursion_depth,
                           config.print_phases);

            if (config.use_randomized_chunks) {
                // probabilty is higher to crash on small inputs
                if (config.use_compressed_buckets && recursion_depth == 0) {
                    local_SA = phase4.space_effient_sort_chunking_SA_compressed(local_string,
                                                                                sample_ranks,
                                                                                bucket_splitter,
                                                                                use_packed_merging);
                } else {
                    local_SA = phase4.space_effient_sort_chunking_SA(local_string,
                                                                     sample_ranks,
                                                                     bucket_splitter,
                                                                     use_packed_merging);
                }

            } else {
                local_SA = phase4.space_effient_sort_SA(local_string,
                                                        sample_ranks,
                                                        bucket_splitter,
                                                        use_packed_merging);
            }

        } else {
            std::vector<MergeSamples> merge_samples =
                phase4.construct_merge_samples(local_string, sample_ranks, use_packed_merging);
            free_memory(std::move(sample_ranks));
            phase4.sort_merge_samples(merge_samples, use_packed_merging);
            local_SA = phase4.extract_SA(merge_samples);
        }

        redistribute_if_imbalanced(local_SA, config.min_imbalance, comm);
        timer.stop();
        if (recursion_depth == 0) {
            uint64_t max_mem = dsss::get_max_mem_bytes();
            get_stats_instance().max_mem_pe_phase_04 = comm.allgather(kamping::send_buf(max_mem));
        }
        if (recursion_depth == 1) {
            uint64_t max_mem = dsss::get_max_mem_bytes();
            get_stats_instance().max_mem_pe_phase_04_1 = comm.allgather(kamping::send_buf(max_mem));
        }
        if constexpr (DEBUG_SIZE)
            print_concatenated_size(local_SA, comm, "local_SA");

        //******* End Phase 4: Merge Suffixes  ********

        // logging
        stats.string_imbalance.push_back(mpi_util::compute_max_imbalance(info.local_chars, comm));
        stats.sample_imbalance.push_back(
            mpi_util::compute_max_imbalance(info.local_sample_size, comm));
        stats.sa_imbalance.push_back(mpi_util::compute_max_imbalance(local_SA.size(), comm));
        get_local_stats_instance().local_text_size_exit.push_back(info.local_chars);
        get_local_stats_instance().local_sample_size_exit.push_back(info.local_sample_size);
        get_local_stats_instance().local_sa_size.push_back(local_SA.size());

        clean_up(local_string);

        if (comm.rank() == 0) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << stats.string_imbalance.back() << ", "
               << stats.sample_imbalance.back() << ", " << stats.sa_imbalance.back();
            std::string imbalance_msg = "Imbalance: string, samples, sa: " + ss.str();
            std::string end_msg = "Finished recursion Level: " + std::to_string(recursion_depth);
            report_on_root(end_msg, comm, recursion_depth, config.print_phases);
            report_on_root(imbalance_msg, comm, recursion_depth, config.print_phases);
        }

        timer.stop(); // pdcx
        if (recursion_depth == 0) {
            std::reverse(stats.string_imbalance.begin(), stats.string_imbalance.end());
            std::reverse(stats.sample_imbalance.begin(), stats.sample_imbalance.end());
            std::reverse(stats.sa_imbalance.begin(), stats.sa_imbalance.end());
            std::reverse(stats.bucket_imbalance_merging.begin(),
                         stats.bucket_imbalance_merging.end());
            std::reverse(stats.bucket_imbalance_merging_received.begin(),
                         stats.bucket_imbalance_merging_received.end());
            std::reverse(stats.avg_segment.begin(), stats.avg_segment.end());
            std::reverse(stats.max_segment.begin(), stats.max_segment.end());
            std::reverse(stats.bucket_sizes.begin(), stats.bucket_sizes.end());
            std::reverse(stats.chunk_sizes_phase4.begin(), stats.chunk_sizes_phase4.end());
        }

        KASSERT(local_string.size() == info.local_chars);
        KASSERT(mpi_util::all_reduce_sum(local_SA.size(), comm) == info.total_chars);
        DBG("checking SA on recursion level " + std::to_string(recursion_depth));
        KASSERT(check_suffixarray(local_SA, local_string, comm),
                "Suffix array is not sorted on level " + std::to_string(recursion_depth));
        DBG("returing from recursion level " + std::to_string(recursion_depth));
        return local_SA;
    }

    void report_time() {
        comm.barrier();
        timer.aggregate_and_print(measurements::FlatPrinter{});
        comm.barrier();
    }

    void report_stats() {
        comm.barrier();
        if (comm.rank() == 0) {
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

    PDCXLengthInfo info;
    PDCXConfig const& config;

    std::array<index_type, DC::D + 1> samples_before;

    mpi::SortingWrapper atomic_sorter;
    dsss::SeqStringSorterWrapper string_sorter_samples;
    dsss::SeqStringSorterWrapper string_sorter_merging;

    Communicator<>& comm;
    measurements::Timer<Communicator<>>& timer;
    Statistics& stats;
    int recursion_depth;
}; // namespace dsss::dcx

} // namespace dsss::dcx

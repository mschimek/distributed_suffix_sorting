#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/measurements/counter.hpp"
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
        max_mem_pe_phase_01.clear();
        max_mem_pe_phase_02.clear();
        max_mem_pe_phase_03.clear();
        max_mem_pe_phase_04.clear();
        max_mem_pe_phase_01_1.clear();
        max_mem_pe_phase_02_1.clear();
        max_mem_pe_phase_03_1.clear();
        max_mem_pe_phase_04_1.clear();
        sample_sort_lcp_compression.clear();
        sample_sort_avg_lcp.clear();
        sample_sort_lcp_total_reduction.clear();
        sample_sort_avg_prefix_length.clear();
        chunk_sizes_phase1.clear();
        chunk_sizes_phase4.clear();
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
        std::cout << "max_mem_pe_phase_01_1=";
        print_vector(max_mem_pe_phase_01_1, ",");
        std::cout << "max_mem_pe_phase_02_1=";
        print_vector(max_mem_pe_phase_02_1, ",");
        std::cout << "max_mem_pe_phase_03_1=";
        print_vector(max_mem_pe_phase_03_1, ",");
        std::cout << "max_mem_pe_phase_04_1=";
        print_vector(max_mem_pe_phase_04_1, ",");
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
        std::cout << "sample_sort_lcp_compression=";
        print_vector(sample_sort_lcp_compression, ",");
        std::cout << "sample_sort_avg_lcp=";
        print_vector(sample_sort_avg_lcp, ",");
        std::cout << "sample_sort_lcp_total_reduction=";
        print_vector(sample_sort_lcp_total_reduction, ",");
        std::cout << "sample_sort_avg_prefix_length=";
        print_vector(sample_sort_avg_prefix_length, ",");
        std::cout << "chunk_sizes_phase1=";
        print_vector(chunk_sizes_phase1, ",");
        std::cout << "chunk_sizes_phase4=";
        print_vector(chunk_sizes_phase4, ",");
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
    std::vector<uint64_t> max_mem_pe_phase_01_1;
    std::vector<uint64_t> max_mem_pe_phase_02_1;
    std::vector<uint64_t> max_mem_pe_phase_03_1;
    std::vector<uint64_t> max_mem_pe_phase_04_1;
    std::vector<uint64_t> max_mem_pe_chunking_before_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_concat;
    std::vector<uint64_t> max_mem_pe_chunking_after_alltoal;
    std::vector<uint64_t> phase_04_sa_size;
    std::vector<uint64_t> phase_04_sa_capacity;
    std::vector<uint64_t> phase_04_before_alltoall_chunks;
    std::vector<uint64_t> phase_04_after_alltoall_chunks;
    std::vector<double> sample_sort_lcp_compression;
    std::vector<double> sample_sort_avg_lcp;
    std::vector<double> sample_sort_lcp_total_reduction;
    std::vector<double> sample_sort_avg_prefix_length;
    std::vector<uint64_t> chunk_sizes_phase1;
    std::vector<uint64_t> chunk_sizes_phase4;
};

// singleton instance
inline Statistics& get_stats_instance() {
    static Statistics stats;
    return stats;
}

struct LocalStats {
    LocalStats() : max_depth(0) {}
    void reset() {
        algo = "";
        num_processors = 0;
        max_depth = 0;
        text_size.clear();
        local_text_size.clear();
        highest_ranks.clear();
        char_type_used.clear();
        discarding_reduction.clear();
        use_discarding.clear();
        local_text_size_exit.clear();
        local_sample_size_exit.clear();
        local_sa_size.clear();
        input_bucket_imbalance_dcx_samples.clear();
        output_bucket_imbalance_dcx_samples.clear();
        input_bucket_imbalance_rank_computation.clear();
        output_bucket_imbalance_rank_computation.clear();
        input_max_bucket_merging_all.clear();
        input_all_bucket_sizes_merging_all.clear();
        input_bucket_imbalance_merging_all.clear();
        output_bucket_imbalance_merging_all.clear();
        redistribute_chars.clear();
        redistribute_samples.clear();
        avg_segment.clear();
        max_segment.clear();
        bucket_sizes.clear();
        packed_chars_samples.clear();
        packed_chars_merging.clear();
        max_mem_pe_phase_01.clear();
        max_mem_pe_phase_02.clear();
        max_mem_pe_phase_03.clear();
        max_mem_pe_phase_04.clear();
        max_mem_pe_phase_01_1.clear();
        max_mem_pe_phase_02_1.clear();
        max_mem_pe_phase_03_1.clear();
        max_mem_pe_phase_04_1.clear();
        // sample_sort_lcp_compression.clear();
        // sample_sort_avg_lcp.clear();
        // sample_sort_lcp_total_reduction.clear();
        // sample_sort_avg_prefix_length.clear();
        chunk_sizes_phase1.clear();
        chunk_sizes_phase4.clear();
    }

    void commit() {
        std::reverse(local_text_size_exit.begin(), local_text_size_exit.end());
        std::reverse(local_sample_size_exit.begin(), local_sample_size_exit.end());
        std::reverse(local_sa_size.begin(), local_sa_size.end());
        std::reverse(input_max_bucket_merging_all.begin(), input_max_bucket_merging_all.end());
        std::reverse(input_bucket_imbalance_merging_all.begin(),
                     input_bucket_imbalance_merging_all.end());
        std::reverse(output_bucket_imbalance_merging_all.begin(),
                     output_bucket_imbalance_merging_all.end());
        std::reverse(bucket_sizes.begin(), bucket_sizes.end());
        std::reverse(chunk_sizes_phase4.begin(), chunk_sizes_phase4.end());
        using namespace kamping::measurements;
        auto append = [&](const std::string& key,
                          const auto& values,
                          std::vector<GlobalAggregationMode> agg = {GlobalAggregationMode::max}) {
            for (const auto& value: values) {
                counter().append(key, static_cast<std::int64_t>(value), agg);
            }
        };
        auto scale = [](auto const& data, std::size_t scalefactor) {
            auto scaled_data = data;
            for (auto& entry: scaled_data) {
                entry *= scalefactor;
            }
            return scaled_data;
        };
        const std::vector<GlobalAggregationMode> aggregate_all{GlobalAggregationMode::max,
                                                               GlobalAggregationMode::min,
                                                               GlobalAggregationMode::sum};
        const std::size_t scale_factor = 1'000'000;
        counter().append("max_depth", max_depth, {GlobalAggregationMode::max});
        append("textsize", text_size);
        append("local_textsize", local_text_size, aggregate_all);
        append("local_textsize_exit", local_text_size_exit, aggregate_all);
        append("local_sample_size_exit", local_sample_size_exit, aggregate_all);
        append("local_sa_size", local_sa_size, aggregate_all);
        append("char_type_used", char_type_used);
        append("discarding_reduction", scale(discarding_reduction, scale_factor));
        append("use_discarding", use_discarding);
        append("highest_rank", highest_ranks);
        // bucket imbalances
        append("input_bucket_imbalance_dcx_samples",
               scale(input_bucket_imbalance_merging_all, scale_factor));
        append("output_bucket_imbalance_dcx_samples",
               scale(output_bucket_imbalance_merging_all, scale_factor));
        append("output_bucket_imbalance_rank_computation",
               scale(output_bucket_imbalance_rank_computation, scale_factor));
        append("input_bucket_imbalance_rank_computation",
               scale(input_bucket_imbalance_rank_computation, scale_factor));
        append("input_bucket_imbalance_merging_all",
               scale(input_bucket_imbalance_merging_all, scale_factor));
        append("output_bucket_imbalance_rank_computation",
               scale(output_bucket_imbalance_rank_computation, scale_factor));
        append("output_bucket_imbalance_merging_all",
               scale(output_bucket_imbalance_merging_all, scale_factor));
        append("input_max_bucket_merging_all", input_max_bucket_merging_all);
        append("input_all_bucket_sizes_merging_all",
               input_all_bucket_sizes_merging_all,
               aggregate_all);
        // remaining stats
        append("packed_chars_samples", scale(packed_chars_samples, scale_factor));
        append("packed_chars_merging", scale(packed_chars_merging, scale_factor));
        append("redistribute_chars", redistribute_chars);
        append("redistribute_samples", redistribute_samples);

        // string_imbalance=0.000,0.002,0.044,0.039,0.039,0.040,0.039
        // sample_imbalance=0.000,0.002,0.044,0.039,0.039,0.041,0.050
        // sa_imbalance=0.000,0.000,0.000,0.039,0.041,0.040,0.039
        // bucket_imbalance_samples=0.044,0.036,0.030
        // bucket_imbalance_samples_received=0.080,0.071,0.077
        // bucket_imbalance_merging=0.233,0.254,0.176
        // bucket_imbalance_merging_received=0.225,0.239,0.154
        // redistribute_chars=0,0,0,0,0,0,0,1
        // redistribute_samples=0,0,0,0
        // avg_segment=
        // max_segment=
        // bucket_sizes=
        // packed_chars_samples=1.077
        // packed_chars_merging=1.077
    }

    std::string algo;
    int num_processors;
    int max_depth;
    std::vector<uint64_t> local_text_size;
    std::vector<uint64_t> text_size;
    std::vector<uint64_t> highest_ranks;
    std::vector<uint64_t> char_type_used;
    std::vector<double> discarding_reduction;
    std::vector<bool> use_discarding;
    std::vector<std::uint64_t> local_text_size_exit;
    std::vector<std::uint64_t> local_sample_size_exit;
    std::vector<std::uint64_t> local_sa_size;
    std::vector<double> input_bucket_imbalance_dcx_samples;
    std::vector<double> output_bucket_imbalance_dcx_samples;
    std::vector<double> input_bucket_imbalance_rank_computation;
    std::vector<double> output_bucket_imbalance_rank_computation;
    std::vector<double> input_max_bucket_merging_all;
    std::vector<double> input_all_bucket_sizes_merging_all;
    std::vector<double> input_bucket_imbalance_merging_all;
    std::vector<double> output_bucket_imbalance_merging_all;
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
    std::vector<uint64_t> max_mem_pe_phase_01_1;
    std::vector<uint64_t> max_mem_pe_phase_02_1;
    std::vector<uint64_t> max_mem_pe_phase_03_1;
    std::vector<uint64_t> max_mem_pe_phase_04_1;
    std::vector<uint64_t> max_mem_pe_chunking_before_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_sort;
    std::vector<uint64_t> max_mem_pe_chunking_after_concat;
    std::vector<uint64_t> max_mem_pe_chunking_after_alltoal;
    std::vector<uint64_t> phase_04_sa_size;
    std::vector<uint64_t> phase_04_sa_capacity;
    std::vector<uint64_t> phase_04_before_alltoall_chunks;
    std::vector<uint64_t> phase_04_after_alltoall_chunks;
    std::vector<uint64_t> chunk_sizes_phase1;
    std::vector<uint64_t> chunk_sizes_phase4;
};

// singleton instance
inline LocalStats& get_local_stats_instance() {
    static LocalStats stats;
    return stats;
}

} // namespace dsss::dcx

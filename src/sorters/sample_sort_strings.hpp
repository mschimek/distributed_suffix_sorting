#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "mpi/stats.hpp"
#include "pdcx/statistics.hpp"
#include "sorters/prefix_doubling.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "strings/merging.hpp"
#include "strings/string_ptr.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"
#include "util/uint_types.hpp"

#ifdef INCLUDE_ALL_SORTERS
#include "RQuick/RQuick.hpp"
#endif

namespace dsss::mpi {

using LcpType = SeqStringSorterWrapper::LcpType;

template <typename DataType>
inline std::vector<LcpType> sample_sort_strings(std::vector<DataType>& local_data,
                                                kamping::Communicator<>& comm,
                                                SeqStringSorterWrapper sorting_wrapper,
                                                SampleSortConfig config = SampleSortConfig(),
                                                bool output_lcps = false) {
    auto& timer = kamping::measurements::timer();

    std::vector<LcpType> lcps;
    auto local_sorter = [&](std::vector<DataType>& local_data) {
        sorting_wrapper.sort(local_data);
    };
    auto local_sorter_with_lcp = [&](std::vector<DataType>& local_data) {
        if (output_lcps) {
            lcps.resize(local_data.size());
            std::fill(lcps.begin(), lcps.end(), LcpType(0));
            sorting_wrapper.sort_with_lcps(local_data, lcps);
        } else {
            sorting_wrapper.sort(local_data);
        }
    };
    auto comp = [&](const DataType& s1, const DataType& s2) { return string_cmp(s1, s2) < 0; };

    auto distributed_sorter = [&](std::vector<DataType>& local_splitters) {
        SampleSortConfig config2 = config;
        config2.splitter_sorting = SplitterSorting::Central;
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
        std::mt19937_64 gen;
        int tag = 42;
        MPI_Comm mpi_comm(comm.mpi_communicator());
        if (config.use_rquick_for_splitters) {
            RQuick::sort(my_mpi_type, local_splitters, tag, gen, mpi_comm, comp);
        } else {
            sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
        }
#else
        sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
#endif
    };

    // code breaks for very small inputs --> switch to sequential sorting
    if (input_is_small(local_data, comm)) {
        report_on_root("sorting on root", comm);
        sort_on_root(local_data, comm, local_sorter_with_lcp);
        if (output_lcps) {
            lcps = mpi_util::distribute_data_custom(lcps, local_data.size(), comm);
        }
        return lcps;
    }

    // print_concatenated_string("redistributed data", comm);
    // handle cases with empty PEs
    timer.synchronize_and_start("string_sample_sort_distribute_data");
    redistribute_imbalanced_data(local_data, comm);
    timer.stop();

    // print_concatenated_size(local_data, comm, "local_data");

    // Sort data locally
    // print_concatenated_string("local sorting 1", comm);

    timer.synchronize_and_start("string_sample_sort_local_sorting_01");
    if (config.use_loser_tree) {
        lcps.resize(local_data.size());
        std::fill(lcps.begin(), lcps.end(), LcpType(0));
        sorting_wrapper.sort_with_lcps(local_data, lcps);
    } else {
        local_sorter(local_data);
    }
    timer.stop();

    // print_concatenated_string("local get global splitters", comm);

    // compute global splitters
    timer.synchronize_and_start("string_sample_sort_global_splitters");
    std::vector<DataType> global_splitters =
        get_global_splitters(local_data, local_sorter, distributed_sorter, comm, config);
    timer.stop();

    // Use the final set of splitters to find the intervals
    timer.synchronize_and_start("string_sample_sort_interval_sizes");
    std::vector<int64_t> interval_sizes =
        compute_interval_sizes(local_data, global_splitters, comm, comp, config);
    timer.stop();

    // exchange data in intervals
    timer.synchronize_and_start("string_sample_sort_alltoall");
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);
    timer.stop();

    if (config.use_loser_tree) {
        timer.synchronize_and_start("string_sample_sort_lcp_alltoall");
        lcps = mpi_util::alltoallv_combined(lcps, interval_sizes, comm);
        timer.stop();
    }

    // merge buckets
    if (config.use_loser_tree) {
        timer.synchronize_and_start("string_sample_sort_loser_tree");
        std::vector<int64_t> receiving_sizes = comm.alltoall(kamping::send_buf(interval_sizes));
        for (uint64_t i = interval_sizes.size(); i < comm.size(); ++i) {
            interval_sizes.emplace_back(0);
        }
        multiway_merge(local_data, lcps, receiving_sizes);
        timer.stop();
    } else {
        timer.synchronize_and_start("string_sample_sort_local_sorting_02");
        local_sorter_with_lcp(local_data);
        timer.stop();
    }
    return lcps;
}

bool need_lcps(SampleSortConfig& config) {
    return config.use_loser_tree || config.use_lcp_compression || config.use_prefix_doubling;
}

void exchange_lcps(std::vector<LcpType>& lcps,
                   std::vector<int64_t> interval_sizes,
                   Communicator<>& comm) {
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("string_sample_sort_lcp_alltoall");
    lcps = mpi_util::alltoallv_combined(lcps, interval_sizes, comm);
    timer.stop();
}

template <typename DataType>
void exchange_reduced_data_volume(std::vector<DataType>& local_data,
                                  std::vector<LcpType>& lcps,
                                  std::vector<int64_t> interval_sizes,
                                  SampleSortConfig& config,
                                  Communicator<>& comm) {
    auto& timer = kamping::measurements::timer();
    using char_type = DataType::CharType;
    using CharContainer = DataType::CharContainerType;
    using NonCharData = DataType::NonCharData;
    using LengthType = dsss::distinguishing_prefix::LengthType;
    const char_type string_separator = std::numeric_limits<char_type>::max();
    KASSERT(!CharContainer::IS_PACKED);
    KASSERT(lcps.size() == local_data.size());

    uint64_t static_string_size = sizeof(CharContainer) / sizeof(char_type);

    std::vector<LengthType> unique_prefix_lengths(local_data.size(), static_string_size);
    if (config.use_prefix_doubling) {
        timer.synchronize_and_start("string_sample_sort_prefix_doubling");
        unique_prefix_lengths =
            dsss::distinguishing_prefix::prefix_doubling(comm,
                                                         local_data,
                                                         config.inital_prefix_length);
        timer.stop();
    }

    // ensure lcp and unique prefix length are consistent with each other
    for (uint64_t i = 1; i < local_data.size(); i++) {
        lcps[i] = std::min(lcps[i], (LcpType)unique_prefix_lengths[i - 1]);
        lcps[i] = std::min(lcps[i], (LcpType)unique_prefix_lengths[i]);
    }

    uint64_t sum_unique_prefix_lengths =
        std::accumulate(unique_prefix_lengths.begin(), unique_prefix_lengths.end(), uint64_t(0));

    /***  LCP-Compression ***/
    timer.synchronize_and_start("string_sample_sort_lcp_compression");

    // set first lcp of each PE region to 0 to be able to reconstruct first element
    // if last interval is 0, segfault can happen --> 2. condition
    uint64_t index = 0;
    for (uint64_t rank = 0; rank < interval_sizes.size() && index < lcps.size(); rank++) {
        KASSERT(index < lcps.size());
        lcps[index] = 0;
        index += interval_sizes[rank];
    }

    uint64_t sum_lcp = std::accumulate(lcps.begin(), lcps.end(), uint64_t(0));
    uint64_t local_send_chars = local_data.size() + sum_unique_prefix_lengths - sum_lcp;

    // statistics
    uint64_t total_elements = mpi_util::all_reduce_sum(local_data.size(), comm);
    uint64_t total_lcps = mpi_util::all_reduce_sum(sum_lcp, comm);
    uint64_t total_send_chars = mpi_util::all_reduce_sum(local_send_chars, comm);

    int64_t total_char_bytes = total_elements * sizeof(CharContainer);
    int64_t total_char_bytes_reduced = total_send_chars * sizeof(char_type);
    int64_t total_send_bytes = total_elements * sizeof(DataType);
    int64_t total_send_bytes_reduced =
        total_send_bytes + (total_char_bytes_reduced - total_char_bytes);

    KASSERT(total_elements > 0ull);
    KASSERT(total_char_bytes > 0);
    KASSERT(total_send_bytes > 0);
    double avg_lcp = (double)total_lcps / total_elements;
    double lcp_compression_reduction = 1.0 - ((double)total_char_bytes_reduced / total_char_bytes);
    double total_reduction = 1.0 - ((double)total_send_bytes_reduced / total_send_bytes);
    double avg_prefix_length = mpi_util::avg_value(unique_prefix_lengths, comm);

    report_on_root("---> LCP-compression reduction: " + std::to_string(lcp_compression_reduction)
                       + ", avg lcp: " + std::to_string(avg_lcp)
                       + ", total reduction: " + std::to_string(total_reduction)
                       + ", avg prefix length: " + std::to_string(avg_prefix_length),
                   comm);
    if (comm.rank() == 0) {
        dsss::dcx::get_stats_instance().sample_sort_lcp_compression.push_back(
            lcp_compression_reduction);
        dsss::dcx::get_stats_instance().sample_sort_avg_lcp.push_back(avg_lcp);
        dsss::dcx::get_stats_instance().sample_sort_lcp_total_reduction.push_back(total_reduction);
        dsss::dcx::get_stats_instance().sample_sort_avg_prefix_length.push_back(avg_prefix_length);
    }

    if (!config.use_prefix_doubling
        && lcp_compression_reduction < config.lcp_compression_threshold) {
        report_on_root("--> skipping LCP-compression: " + std::to_string(lcp_compression_reduction)
                           + " < " + std::to_string(config.lcp_compression_threshold),
                       comm);
        exchange_data_directly(local_data, lcps, interval_sizes, config, comm);
        return;
    }

    // linearize chars into a single vector with separators
    std::vector<char_type> send_chars(local_send_chars);
    uint64_t write_index = 0;
    for (uint64_t i = 0; i < local_data.size(); i++) {
        auto start = local_data[i].cbegin_chars() + lcps[i];
        auto end = local_data[i].cbegin_chars() + unique_prefix_lengths[i];
        std::copy(start, end, send_chars.begin() + write_index);
        write_index += end - start;
        KASSERT(write_index <= local_send_chars);
        send_chars[write_index++] = string_separator;
    }
    KASSERT(write_index == local_send_chars);

    // compute send counts
    std::vector<int64_t> send_cnts(comm.size(), 0);
    uint64_t k = 0;
    for (uint64_t rank = 0; rank < interval_sizes.size(); rank++) {
        for (int64_t i = 0; i < interval_sizes[rank]; i++) {
            KASSERT(k < unique_prefix_lengths.size());
            KASSERT(k < lcps.size());
            send_cnts[rank] += 1 + unique_prefix_lengths[k] - lcps[k];
            k++;
        }
    }
    KASSERT(std::accumulate(send_cnts.begin(), send_cnts.end(), int64_t(0))
            == (int64_t)local_send_chars);

    // extract non-char data
    std::vector<NonCharData> non_char_data;
    non_char_data.reserve(local_data.size());
    for (auto& x: local_data) {
        non_char_data.push_back(x.get_non_char_data());
    }
    free_memory(std::move(local_data));

    timer.stop();
    /***  LCP-Compression ***/

    // exchange all data

    // non-char data
    timer.synchronize_and_start("string_sample_sort_non_char_data_alltoall");
    non_char_data = mpi_util::alltoallv_combined(non_char_data, interval_sizes, comm);
    timer.stop();

    // linearized chars
    timer.synchronize_and_start("string_sample_sort_alltoall");
    send_chars = mpi_util::alltoallv_combined(send_chars, send_cnts, comm);
    timer.stop();

    // lcps
    exchange_lcps(lcps, interval_sizes, comm);

    timer.synchronize_and_start("string_sample_sort_lcp_decompression");

    // reconstruct original strings
    local_data.resize(lcps.size());
    auto char_it = send_chars.begin();
    auto char_end = char_it;
    auto move_char_end = [&]() {
        char_end++;
        KASSERT(char_end <= send_chars.end());
        while (*char_end != string_separator) {
            char_end++;
            KASSERT(char_end <= send_chars.end());
        }
    };

    // first element has no compressed lcp
    KASSERT(lcps.size() == uint64_t(0)
            || lcps[0] == LcpType(0)); // ensure this by setting lcps to 0 before sending

    // PE might not receive any element
    if (local_data.size() > 0) {
        move_char_end();
        CharContainer chars_first = CharContainer(char_it, char_end);
        char_it = char_end + 1;
        local_data[0] = DataType(std::move(chars_first), std::move(non_char_data[0]));
    }

    for (uint64_t i = 1; i < local_data.size(); i++) {
        CharContainer chars;
        auto write_it = chars.begin();

        // copy lcp from previous string
        KASSERT((uint64_t)lcps[i] <= local_data[i - 1].chars.size());
        std::copy(local_data[i - 1].chars.begin(),
                  local_data[i - 1].chars.begin() + lcps[i],
                  write_it);
        write_it += lcps[i];

        move_char_end();
        uint64_t diff = char_end - char_it;
        KASSERT(write_it + diff <= chars.end());
        std::copy(char_it, char_end, write_it);
        char_it = char_end + 1;
        KASSERT(char_it <= send_chars.end());

        local_data[i] = DataType(std::move(chars), std::move(non_char_data[i]));
    }
    KASSERT(char_it == send_chars.end());
    timer.stop();
}


template <typename DataType>
void exchange_data_directly(std::vector<DataType>& local_data,
                            std::vector<LcpType>& lcps,
                            std::vector<int64_t> interval_sizes,
                            SampleSortConfig& config,
                            Communicator<>& comm) {
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("string_sample_sort_alltoall");
    local_data = mpi_util::alltoallv_combined(local_data, interval_sizes, comm);
    timer.stop();
    if (need_lcps(config)) {
        exchange_lcps(lcps, interval_sizes, comm);
    }
}

template <typename DataType>
void exchange_data(std::vector<DataType>& local_data,
                   std::vector<LcpType>& lcps,
                   std::vector<int64_t> interval_sizes,
                   SampleSortConfig& config,
                   Communicator<>& comm) {
    bool use_direct_exchange = !config.use_lcp_compression && !config.use_prefix_doubling;
    if (use_direct_exchange) {
        exchange_data_directly(local_data, lcps, interval_sizes, config, comm);
    } else {
        exchange_reduced_data_volume(local_data, lcps, interval_sizes, config, comm);
    }
}

template <typename DataType>
void sample_sort_strings_tie_break(std::vector<DataType>& local_data,
                                   kamping::Communicator<>& comm,
                                   SeqStringSorterWrapper sorting_wrapper,
                                   auto tie_break,
                                   auto comp,
                                   SampleSortConfig config = SampleSortConfig()) {
    auto& timer = kamping::measurements::timer();

    std::vector<LcpType> lcps;
    auto local_sorter = [&](std::vector<DataType>& local_data) {
        timer.synchronize_and_start("string_sample_sort_local_sorting");
        sorting_wrapper.sort(local_data);
        timer.stop();

        timer.synchronize_and_start("string_sample_sort_tie_break");
        tie_break(local_data);
        timer.stop();
    };

    auto local_sorter_with_lcp = [&](std::vector<DataType>& local_data) {
        lcps.resize(local_data.size());
        std::fill(lcps.begin(), lcps.end(), LcpType(0));

        timer.synchronize_and_start("string_sample_sort_local_sorting");
        sorting_wrapper.sort_with_lcps(local_data, lcps);
        timer.stop();

        // does not change lcps
        timer.synchronize_and_start("string_sample_sort_tie_break");
        tie_break(local_data);
        timer.stop();
    };
    // auto comp = std::less<>{}; // < operator of MergeSample struct has tie breaking

    auto distributed_sorter = [&](std::vector<DataType>& local_splitters) {
        SampleSortConfig config2 = config;
        config2.splitter_sorting = SplitterSorting::Central;
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
        std::mt19937_64 gen;
        int tag = 42;
        MPI_Comm mpi_comm(comm.mpi_communicator());
        if (config.use_rquick_for_splitters) {
            RQuick::sort(my_mpi_type, local_splitters, tag, gen, mpi_comm, comp);
        } else {
            sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
        }
#else
        sample_sort_strings(local_splitters, comm, sorting_wrapper, config2);
#endif
    };

    // code breaks for very small inputs --> switch to sequential sorting
    if (input_is_small(local_data, comm)) {
        report_on_root("sorting on root", comm);
        sort_on_root(local_data, comm, local_sorter);
        return;
    }

    // handle cases with empty PEs
    timer.synchronize_and_start("string_sample_sort_distribute_data");
    redistribute_imbalanced_data(local_data, comm);
    timer.stop();

    // Sort data locally
    if (need_lcps(config)) {
        local_sorter_with_lcp(local_data);

    } else {
        local_sorter(local_data);
    }

    // compute global splitters
    timer.synchronize_and_start("string_sample_sort_global_splitters");
    std::vector<DataType> global_splitters =
        get_global_splitters(local_data, local_sorter, distributed_sorter, comm, config);
    timer.stop();

    // Use the final set of splitters to find the intervals
    timer.synchronize_and_start("string_sample_sort_interval_sizes");
    std::vector<int64_t> interval_sizes =
        compute_interval_sizes(local_data, global_splitters, comm, comp, config);
    timer.stop();

    auto local_data_copy = local_data;
    // exchange data in intervals
    exchange_data(local_data, lcps, interval_sizes, config, comm);

    // merge buckets
    if (config.use_loser_tree) {
        timer.synchronize_and_start("string_sample_sort_loser_tree");
        std::vector<int64_t> receiving_sizes = comm.alltoall(kamping::send_buf(interval_sizes));
        for (uint64_t i = interval_sizes.size(); i < comm.size(); ++i) {
            interval_sizes.emplace_back(0);
        }
        multiway_merge(local_data, lcps, receiving_sizes);
        timer.stop();

        timer.synchronize_and_start("string_sample_sort_tie_break");
        tie_break(local_data);
        timer.stop();
    } else {
        local_sorter(local_data);
    }
}

template <typename DataType>
void sample_sort_strings_tie_break(std::vector<DataType>& local_data,
                                   kamping::Communicator<>& comm,
                                   SeqStringSorterWrapper sorting_wrapper,
                                   auto tie_break,
                                   SampleSortConfig config = SampleSortConfig()) {
    sample_sort_strings_tie_break(local_data,
                                  comm,
                                  sorting_wrapper,
                                  tie_break,
                                  std::less<DataType>{},
                                  config);
}


} // namespace dsss::mpi

/******************************************************************************/
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <sys/types.h>

#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"
#include "pdcx/config.hpp"
#include "pdcx/sample_string.hpp"
#include "sorters/sample_sort_common.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/binary_search.hpp"
#include "util/printing.hpp"

namespace dsss::dcx {

using namespace kamping;

template <typename char_type, typename index_type, typename DC>
struct SpaceEfficientSort {
    // X chars and one 0-character
    using Splitter = std::array<char_type, DC::X + 1>;
    // using SampleString = DCSampleString<char_type, index_type, DC>;
    using BucketMappingType = uint16_t;

    Communicator<>& comm;
    PDCXConfig const& config;

    SpaceEfficientSort(Communicator<>& _comm, PDCXConfig const& _config)
        : comm(_comm),
          config(_config) {}

    std::array<char_type, DC::X + 1> materialize_splitter(std::vector<char_type>& local_string,
                                                          uint64_t i) const {
        std::array<char_type, DC::X + 1> chars;
        std::copy(local_string.begin() + i, local_string.begin() + i + DC::X, chars.begin());
        chars.back() = 0;
        return chars;
    }

    // compute splitters for partition into blocks
    std::vector<Splitter> random_sample_splitters(uint64_t local_chars,
                                                  uint64_t blocks,
                                                  std::vector<char_type>& local_string) {
        size_t nr_splitters =
            std::max<size_t>((config.num_samples_splitters + comm.size() - 1) / comm.size(),
                             blocks);
        auto _materialize_splitter = [&](uint64_t i) {
            return materialize_splitter(local_string, i);
        };
        std::vector<Splitter> local_splitters =
            mpi::sample_random_splitters1<Splitter>(local_chars,
                                                    nr_splitters,
                                                    _materialize_splitter,
                                                    comm);

        auto cmp = [](Splitter const& a, Splitter const& b) {
            for (uint64_t i = 0; i < a.size(); i++) {
                if (a[i] != b[i])
                    return a[i] < b[i];
            }
            return false;
        };
        std::vector<Splitter> all_splitters = comm.allgatherv(kamping::send_buf(local_splitters));
        ips4o::sort(all_splitters.begin(), all_splitters.end(), cmp);

        return mpi::sample_uniform_splitters(all_splitters, blocks - 1, comm);
    }

    // compute even distributed splitters from sorted local samples
    template <typename SampleString>
    std::vector<SampleString> get_uniform_splitters(std::vector<SampleString>& local_samples,
                                                    uint64_t blocks) {
        int64_t num_samples = local_samples.size();
        int64_t samples_before = mpi_util::ex_prefix_sum(num_samples, comm);
        int64_t total_sample_size = mpi_util::all_reduce_sum(num_samples, comm);
        std::vector<SampleString> local_splitters;
        for (uint64_t i = 0; i < blocks - 1; i++) {
            int64_t global_index = (i + 1) * total_sample_size / blocks;
            int64_t x = global_index - samples_before;
            if (x >= 0 && x < num_samples) {
                local_splitters.push_back(local_samples[x]);
            }
        }
        std::vector<SampleString> global_splitters = comm.allgatherv(send_buf(local_splitters));
        return global_splitters;
    }
    std::pair<std::vector<uint64_t>, std::vector<BucketMappingType>>
    compute_sample_to_block_mapping(std::vector<char_type>& local_string,
                                    uint64_t local_chars,
                                    uint64_t blocks,
                                    auto get_kth_splitter_at) {
        std::vector<uint64_t> bucket_sizes(blocks, 0);
        std::vector<BucketMappingType> sample_to_block(local_string.size(), 0);
        KASSERT(blocks <= 255ull);

        auto cmp_substring = [&](uint64_t local_index, uint64_t splitter_nr) {
            for (uint64_t k = 0; k < DC::X - 1; k++) {
                if (local_string[local_index + k] != get_kth_splitter_at(splitter_nr, k)) {
                    return local_string[local_index + k] < get_kth_splitter_at(splitter_nr, k);
                }
            }
            return false;
        };

        // assign each substring to a block
        for (uint64_t i = 0; i < local_chars; i++) {
            BucketMappingType block_id = blocks - 1;
            auto cmp = [&](int64_t k) { return (cmp_substring(i, k)); };
            block_id = util::binary_search(0, blocks - 1, cmp);
            bucket_sizes[block_id]++;
            sample_to_block[i] = block_id;
        }
        return {bucket_sizes, sample_to_block};
    }

    template <typename DataType>
    std::pair<std::vector<uint64_t>, std::vector<BucketMappingType>>
    compute_sample_to_block_mapping(auto get_element_at,
                                    auto cmp_element,
                                    uint64_t local_size,
                                    std::vector<DataType>& global_splitters) {
        uint64_t blocks = global_splitters.size() + 1;
        std::vector<uint64_t> bucket_sizes(blocks, 0);
        std::vector<BucketMappingType> sample_to_block(local_size, 0);
        KASSERT(blocks <= 255ull);

        // assign each substring to a block
        for (uint64_t i = 0; i < local_size; i++) {
            BucketMappingType block_id = blocks - 1;
            DataType element = get_element_at(i);
            auto cmp = [&](int64_t k) { return cmp_element(element, global_splitters[k]); };
            block_id = util::binary_search(0, blocks - 1, cmp);
            bucket_sizes[block_id]++;
            sample_to_block[i] = block_id;
        }
        return {bucket_sizes, sample_to_block};
    }

    // variant for general elements
    // compute splitters for partition into blocks
    template <typename DataType>
    std::vector<DataType> general_random_sample_splitters(auto get_element_at,
                                                          auto cmp_element,
                                                          uint64_t local_size,
                                                          uint64_t blocks,
                                                          uint64_t total_samples,
                                                          bool use_rquick = false) {
        uint64_t n = comm.size();
        uint64_t local_samples = (total_samples + n - 1) / n;
        std::vector<DataType> local_splitters =
            mpi::sample_random_splitters1<DataType>(local_size,
                                                    local_samples,
                                                    get_element_at,
                                                    comm);

        if (use_rquick) {
            mpi::SortingWrapper sorter(comm);
            sorter.set_sorter(mpi::Rquick);
            sorter.sort(local_splitters, cmp_element);
            return get_uniform_splitters(local_splitters, blocks);
        } else {
            std::vector<DataType> all_splitters =
                comm.allgatherv(kamping::send_buf(local_splitters));
            ips4o::sort(all_splitters.begin(), all_splitters.end(), cmp_element);
            return mpi::sample_uniform_splitters(all_splitters, blocks - 1, comm);
        }
    }
};
inline double get_imbalance_bucket(std::vector<uint64_t> const& bucket_sizes,
                                   uint64_t total_chars,
                                   Communicator<>& comm) {
    uint64_t num_buckets = bucket_sizes.size();
    uint64_t largest_bucket = mpi_util::all_reduce_max(bucket_sizes, comm);
    double avg_buckets = (double)total_chars / (num_buckets * comm.size());
    double bucket_imbalance = ((double)largest_bucket / avg_buckets) - 1.0;
    return bucket_imbalance;
}

inline double get_max_local_bucket(std::vector<std::uint64_t> const& bucket_sizes) {
    std::uint64_t const num_buckets = bucket_sizes.size();
    if (bucket_sizes.empty()) {
        return 0.;
    }
    return *std::max_element(bucket_sizes.begin(), bucket_sizes.end());
}

inline double get_max_local_imbalance(std::vector<std::uint64_t> const& bucket_sizes,
                                      std::uint64_t total_elements,
                                      std::size_t comm_size) {
    std::uint64_t const num_buckets = bucket_sizes.size();
    if (bucket_sizes.empty()) {
        return 0.;
    }
    std::uint64_t const local_max = *std::max_element(bucket_sizes.begin(), bucket_sizes.end());
    double const avg_bucket_load = static_cast<double>(total_elements) / (num_buckets * comm_size);
    double const bucket_imbalance = static_cast<double>(local_max) / avg_bucket_load;
    return bucket_imbalance;
}
} // namespace dsss::dcx

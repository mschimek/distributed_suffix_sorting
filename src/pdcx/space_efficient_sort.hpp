#pragma once

#include <random>
#include <vector>

#include "kamping/communicator.hpp"
#include "pdcx/config.hpp"
#include "pdcx/sample_string.hpp"
#include "sorters/sample_sort_common.hpp"
#include "util/printing.hpp"

namespace dsss::dcx {

using namespace kamping;

template <typename char_type, typename index_type, typename DC>
struct SpaceEfficientSort {
    using Splitter = typename DCSampleString<char_type, index_type, DC>::SampleStringLetters;

    Communicator<>& comm;
    PDCXConfig& config;

    SpaceEfficientSort(Communicator<>& _comm, PDCXConfig& _config) : comm(_comm), config(_config) {}

    // compute splitters for partition into blocks
    std::vector<Splitter>
    random_sample_splitters(uint64_t local_chars, uint64_t blocks, auto materialize_sample) {
        size_t nr_splitters =
            std::max<size_t>((config.num_samples_splitters + comm.size() - 1) / comm.size(),
                             blocks);
        std::vector<Splitter> local_splitters =
            mpi::sample_random_splitters1<Splitter>(local_chars,
                                                    nr_splitters,
                                                    materialize_sample,
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

    std::pair<std::vector<uint64_t>, std::vector<uint8_t>>
    compute_sample_to_block_mapping(std::vector<char_type>& local_string,
                                    uint64_t local_chars,
                                    std::vector<Splitter>& global_splitters) {
        int64_t blocks = global_splitters.size() + 1;
        std::vector<uint64_t> bucket_sizes(blocks, 0);
        std::vector<uint8_t> sample_to_block(local_string.size(), 0);
        KASSERT(blocks <= 255);

        // assign each substring to a block
        for (uint64_t i = 0; i < local_chars; i++) {
            uint8_t block_id = blocks - 1;
            for (int64_t j = 0; j < blocks - 1; j++) {
                if (cmp_index_substring(local_string, i, global_splitters[j], DC::X - 1)) {
                    block_id = j;
                    break;
                }
            }
            bucket_sizes[block_id]++;
            sample_to_block[i] = block_id;
        }
        return {bucket_sizes, sample_to_block};
    }
};
double get_imbalance_bucket(std::vector<uint64_t>& bucket_sizes,
                            uint64_t total_chars,
                            Communicator<>& comm) {
    uint64_t num_buckets = bucket_sizes.size();
    uint64_t largest_bucket = mpi_util::all_reduce_max(bucket_sizes, comm);
    double avg_buckets = (double)total_chars / (num_buckets * comm.size());
    double bucket_imbalance = ((double)largest_bucket / avg_buckets) - 1.0;
    return bucket_imbalance;
}
} // namespace dsss::dcx

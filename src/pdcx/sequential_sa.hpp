#pragma once

#include <vector>

#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "mpi/distribute.hpp"
#include "pdcx/compute_ranks.hpp"
#include "util/printing.hpp"
#include "util/string_util.hpp"

namespace dsss::dcx {

using namespace kamping;


template <typename char_type, typename index_type>
std::vector<index_type> compute_sa_on_root(std::vector<char_type>& local_string,
                                           Communicator<>& comm) {
    std::vector<char_type> global_string = comm.gatherv(send_buf(local_string));
    std::vector<index_type> SA;
    if (comm.rank() == 0) {
        SA = slow_suffixarray<char_type, index_type>(global_string);
    }
    mpi_util::distribute_data_custom(SA, local_string.size(), comm);
    return SA;
}

// skip recursion and compute SA with sequential algorithm on root process
template <typename char_type, typename index_type, typename DC>

void sequential_sa_on_local_ranks(std::vector<DCRankIndex<char_type, index_type, DC>>& local_ranks,
                                  uint64_t local_sample_size,
                                  auto map_back_func,
                                  Communicator<>& comm) {
    using RankIndex = DCRankIndex<char_type, index_type, DC>;

    std::vector<RankIndex> global_ranks = comm.gatherv(send_buf(local_ranks));
    std::vector<index_type> SA;
    if (comm.rank() == 0) {
        std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_mod_div);
        auto get_rank = [](RankIndex& r) -> index_type { return r.rank; };
        std::vector<index_type> ranks =
            extract_attribute<RankIndex, index_type>(global_ranks, get_rank);

        // TODO: better sequential SACA
        SA = slow_suffixarray<index_type, index_type>(ranks);
        global_ranks.clear();

        for (uint64_t i = 0; i < SA.size(); i++) {
            index_type global_index = map_back_func(SA[i]);
            global_ranks.emplace_back(i + 1, global_index);
        }
        std::sort(global_ranks.begin(), global_ranks.end(), RankIndex::cmp_by_index);
    }

    local_ranks.clear();
    local_ranks = mpi_util::distribute_data_custom(global_ranks, local_sample_size, comm);
}
} // namespace dsss::dcx
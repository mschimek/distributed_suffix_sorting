#pragma once


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"
#include "util/printing.hpp"
#include "util/random.hpp"

namespace dsss::test {


void test_sorting(int repeats,
                  int local_size,
                  auto distributed_sorter,
                  kamping::Communicator<>& comm) {
    using namespace kamping;

    int rank = comm.rank();
    int size = comm.size();
    int max_value = 1e6;
    for (int i = 0; i < repeats; i++) {
        int seed = i * size + rank;
        std::vector<int> local_data =
            dsss::random::generate_random_data<int>(local_size, max_value, seed);
        comm.barrier();
        distributed_sorter(local_data, comm);
        auto sorted_sequence = comm.gatherv(send_buf(local_data));
        KASSERT(std::is_sorted(sorted_sequence.begin(), sorted_sequence.end()));
    }
}

void test_distribute_data_custom(int repeats, kamping::Communicator<>& comm) {
    int avg_size = 100;
    for (int r = 0; r < repeats; r++) {
        std::vector<int> initial_size(comm.size());
        std::vector<int> target_size(comm.size());

        // random distribution of intial and target sizes
        int seed = 0; // same seed for all processes
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 2 * avg_size);
        for (uint i = 0; i < comm.size(); i++) {
            initial_size[i] = dist(rng);
            target_size[i] = dist(rng);
        }

        // ensure sum of sizes are equal
        int sum_inital = std::accumulate(initial_size.begin(), initial_size.end(), 0);
        int sum_target = std::accumulate(target_size.begin(), target_size.end(), 0);
        if (sum_target > sum_inital) {
            initial_size.back() += sum_target - sum_inital;
        } else if (sum_target < sum_inital) {
            target_size.back() += sum_inital - sum_target;
        }
        std::vector<int> local_data(initial_size[comm.rank()], comm.rank());
        std::vector<int> rcv_data =
            mpi_util::distribute_data_custom(local_data, target_size[comm.rank()], comm);

        std::vector<int> global_data = comm.allgatherv(kamping::send_buf(local_data));
        std::vector<int> global_rcv_data = comm.allgatherv(kamping::send_buf(rcv_data));
        KASSERT(target_size[comm.rank()] == (int)rcv_data.size());
        KASSERT(global_data == global_rcv_data);
    }
}
} // namespace dsss::test

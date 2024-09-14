#pragma once


#include <algorithm>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>

#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "printing.hpp"

namespace dsss::test {
std::vector<int> generate_random_data(int n, int max_value, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, max_value);
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) {
        v[i] = dist(rng);
    }
    return v;
}

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
        std::vector<int> local_data = generate_random_data(local_size, max_value, seed);
        comm.barrier();
        distributed_sorter(local_data, comm);
        auto sorted_sequence = comm.gatherv(send_buf(local_data));
        KASSERT(std::is_sorted(sorted_sequence.begin(), sorted_sequence.end()));
    }
}

} // namespace test

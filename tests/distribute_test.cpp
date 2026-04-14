#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"

using namespace dsss;

class DistributeTest : public ::testing::Test {
protected:
    kamping::Communicator<> comm;
};

TEST_F(DistributeTest, DistributeDataEvenly_PreservesGlobalData) {
    int local_size = 100 + comm.rank() * 13; // intentionally uneven
    std::vector<int> local_data(local_size);
    std::iota(local_data.begin(), local_data.end(), comm.rank() * 10000);

    namespace kmp = kamping::params;
    auto before = comm.allgatherv(kmp::send_buf(local_data));
    std::sort(before.begin(), before.end());

    auto result = mpi_util::distribute_data(local_data, comm);

    auto after = comm.allgatherv(kmp::send_buf(result));
    std::sort(after.begin(), after.end());

    EXPECT_EQ(before, after);
}

TEST_F(DistributeTest, DistributeDataEvenly_BalancedSizes) {
    int local_size = 100 + comm.rank() * 7;
    std::vector<int> local_data(local_size, comm.rank());

    auto result = mpi_util::distribute_data(local_data, comm);

    size_t total = local_data.size();
    size_t global_total = 0;
    MPI_Allreduce(&total, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm.mpi_communicator());

    // distribute_data gives each rank total/p elements, last rank gets remainder
    size_t expected = std::max<size_t>(1, global_total / comm.size());
    if (comm.rank() + 1 == comm.size()) {
        // last rank gets whatever is left
        EXPECT_EQ(result.size(), global_total - expected * (comm.size() - 1));
    } else {
        EXPECT_EQ(result.size(), expected);
    }
}

TEST_F(DistributeTest, DistributeDataCustom_PreservesGlobalData) {
    int avg_size = 100;
    std::vector<int> initial_size(comm.size());
    std::vector<int> target_size(comm.size());

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0, 2 * avg_size);
    for (int i = 0; i < (int)comm.size(); i++) {
        initial_size[i] = dist(rng);
        target_size[i] = dist(rng);
    }

    // balance totals
    int sum_initial = std::accumulate(initial_size.begin(), initial_size.end(), 0);
    int sum_target = std::accumulate(target_size.begin(), target_size.end(), 0);
    if (sum_target > sum_initial) {
        initial_size.back() += sum_target - sum_initial;
    } else if (sum_target < sum_initial) {
        target_size.back() += sum_initial - sum_target;
    }

    std::vector<int> local_data(initial_size[comm.rank()], comm.rank());

    namespace kmp = kamping::params;
    auto before = comm.allgatherv(kmp::send_buf(local_data));
    std::sort(before.begin(), before.end());

    auto result = mpi_util::distribute_data_custom(local_data, target_size[comm.rank()], comm);

    EXPECT_EQ((int)result.size(), target_size[comm.rank()]);

    auto after = comm.allgatherv(kmp::send_buf(result));
    std::sort(after.begin(), after.end());

    EXPECT_EQ(before, after);
}

#include <algorithm>
#include <functional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/communicator.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/random.hpp"

using namespace dsss;

class SortingTest : public ::testing::Test {
protected:
    kamping::Communicator<> comm;
};

TEST_F(SortingTest, SortRandomIntegers_Small) {
    int n = 100;
    int max_value = 1000;
    int seed = comm.rank();
    auto local_data = dsss::random::generate_random_data<int>(n, max_value, seed);

    mpi::SortingWrapper sorter(comm);
    sorter.set_sorter(mpi::AtomicSorters::Ams);
    sorter.finalize_setting();
    sorter.sort(local_data, std::less<int>{});

    namespace kmp = kamping::params;
    auto global = comm.allgatherv(kmp::send_buf(local_data));
    EXPECT_TRUE(std::is_sorted(global.begin(), global.end()));
}

TEST_F(SortingTest, SortRandomIntegers_Large) {
    int n = 10000;
    int max_value = 1000000;
    int seed = 42 + comm.rank();
    auto local_data = dsss::random::generate_random_data<int>(n, max_value, seed);

    mpi::SortingWrapper sorter(comm);
    sorter.set_sorter(mpi::AtomicSorters::Ams);
    sorter.finalize_setting();
    sorter.sort(local_data, std::less<int>{});

    namespace kmp = kamping::params;
    auto global = comm.allgatherv(kmp::send_buf(local_data));
    EXPECT_TRUE(std::is_sorted(global.begin(), global.end()));
}

TEST_F(SortingTest, SortPreservesElements) {
    int n = 1000;
    int max_value = 500;
    int seed = comm.rank();
    auto local_data = dsss::random::generate_random_data<int>(n, max_value, seed);

    namespace kmp = kamping::params;
    auto before = comm.allgatherv(kmp::send_buf(local_data));
    std::sort(before.begin(), before.end());

    mpi::SortingWrapper sorter(comm);
    sorter.set_sorter(mpi::AtomicSorters::Ams);
    sorter.finalize_setting();
    sorter.sort(local_data, std::less<int>{});

    auto after = comm.allgatherv(kmp::send_buf(local_data));
    EXPECT_EQ(before, after);
}

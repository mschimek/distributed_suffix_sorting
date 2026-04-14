#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "kamping/communicator.hpp"
#include "dcx/dcx_instantiations.hpp"
#include "sa_check.hpp"
#include "util/random.hpp"

using namespace dsss;

namespace {

template <typename DCXVariant>
bool run_sa_test(int n, int alphabet_size, int seed, kamping::Communicator<>& comm) {
    dcx::PDCXConfig config;
    config.atomic_sorter = mpi::AtomicSorters::Ams;
    config.print_phases = false;

    using char_type = uint8_t;
    auto local_data = dsss::random::generate_random_data<char_type>(n, alphabet_size, seed);
    DCXVariant algo(config);
    auto sa = algo.compute_suffix_array(local_data, comm);
    return check_suffixarray(sa, local_data, comm);
}

} // namespace

class SuffixArrayTest : public ::testing::Test {
protected:
    kamping::Communicator<> comm;
};

TEST_F(SuffixArrayTest, DC39_Unpacked_SmallInput_BinaryAlphabet) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8>(100, 2, seed, comm)));
    }
}

TEST_F(SuffixArrayTest, DC39_Unpacked_SmallInput_LargeAlphabet) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8>(100, 32, seed, comm)));
    }
}

TEST_F(SuffixArrayTest, DC39_Unpacked_MediumInput) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8>(1000, 8, seed, comm)));
    }
}

TEST_F(SuffixArrayTest, DC39_Packed8bit_SmallInput) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8_8bit>(100, 8, seed, comm)));
    }
}

TEST_F(SuffixArrayTest, DC39_Packed5bit_SmallInput) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8_5bit>(100, 8, seed, comm)));
    }
}

TEST_F(SuffixArrayTest, DC39_Packed3bit_SmallInput) {
    for (int seed = 0; seed < 3; ++seed) {
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8_3bit>(100, 4, seed, comm)));
    }
}

// Test alignment: input sizes that cover all remainders mod X
TEST_F(SuffixArrayTest, DC39_Unpacked_AlignmentTest) {
    constexpr int X = 39;
    for (int d = 0; d < X; ++d) {
        int n = 100 + (comm.rank() == 0 ? d : 0);
        int seed = d * comm.size() + comm.rank();
        EXPECT_TRUE((run_sa_test<dcx::DC39_u8>(n, 8, seed, comm)))
            << "Failed for alignment offset d=" << d;
    }
}

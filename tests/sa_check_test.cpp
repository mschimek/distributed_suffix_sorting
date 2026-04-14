#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"
#include "sa_check.hpp"
#include "util/random.hpp"
#include "util/string_util.hpp"

using namespace dsss;

using char_type = uint8_t;
using index_type = uint32_t;

// Parameterized fixture: (text_size_per_pe, alphabet_size)
class SACheckTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    kamping::Communicator<> comm;

    int text_size() const { return std::get<0>(GetParam()); }
    int alphabet_size() const { return std::get<1>(GetParam()); }

    // Build a correct suffix array using the naive O(n^2 log n) algorithm on root,
    // then distribute it to match the local text distribution.
    std::vector<index_type> naive_sa(std::vector<char_type>& local_text) {
        namespace kmp = kamping::params;
        auto global_text = comm.gatherv(kmp::send_buf(local_text));

        std::vector<index_type> global_sa;
        if (comm.rank() == 0) {
            global_sa = slow_suffixarray<char_type, index_type>(global_text);
        }

        return mpi_util::distribute_data_custom(global_sa, local_text.size(), comm);
    }
};

// --- Positive tests: correct suffix arrays must pass ---

TEST_P(SACheckTest, NaiveSA_MatchesChecker) {
    auto local_text =
        dsss::random::generate_random_data<char_type>(text_size(), alphabet_size(), comm.rank());
    auto sa = naive_sa(local_text);
    EXPECT_TRUE(check_suffixarray(sa, local_text, comm));
}

TEST_P(SACheckTest, NaiveSA_AllSameCharacter) {
    std::vector<char_type> local_text(text_size(), 'a');
    auto sa = naive_sa(local_text);
    EXPECT_TRUE(check_suffixarray(sa, local_text, comm));
}

// --- Negative tests: incorrect suffix arrays must fail ---

TEST_P(SACheckTest, ReversedSA_Fails) {
    auto local_text =
        dsss::random::generate_random_data<char_type>(text_size(), alphabet_size(), comm.rank());
    auto sa = naive_sa(local_text);

    namespace kmp = kamping::params;
    auto global_sa = comm.gatherv(kmp::send_buf(sa));
    if (comm.rank() == 0) {
        std::reverse(global_sa.begin(), global_sa.end());
    }
    auto bad_sa = mpi_util::distribute_data_custom(global_sa, local_text.size(), comm);

    EXPECT_FALSE(check_suffixarray(bad_sa, local_text, comm));
}

TEST_P(SACheckTest, SwappedAdjacentEntries_Fails) {
    auto local_text =
        dsss::random::generate_random_data<char_type>(text_size(), alphabet_size(), comm.rank());
    auto sa = naive_sa(local_text);

    if (comm.rank() == 0 && sa.size() >= 2) {
        std::swap(sa[0], sa[1]);
    }

    EXPECT_FALSE(check_suffixarray(sa, local_text, comm));
}

TEST_P(SACheckTest, DuplicateIndex_Fails) {
    auto local_text =
        dsss::random::generate_random_data<char_type>(text_size(), alphabet_size(), comm.rank());
    auto sa = naive_sa(local_text);

    if (comm.rank() == 0 && sa.size() >= 2) {
        sa.back() = sa.front();
    }

    EXPECT_FALSE(check_suffixarray(sa, local_text, comm));
}

INSTANTIATE_TEST_SUITE_P(Sizes,
                         SACheckTest,
                         ::testing::Values(std::make_tuple(50, 2),
                                           std::make_tuple(500, 200),
                                           std::make_tuple(2000, 8)),
                         [](const auto& info) {
                             return "n" + std::to_string(std::get<0>(info.param))
                                    + "_a" + std::to_string(std::get<1>(info.param));
                         });

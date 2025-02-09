#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include <mpi.h>
#include <sys/types.h>

#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"
#include "options.hpp"
#include "pdcx/config.hpp"
#include "sa_check.hpp"
#include "sorters/sample_sort_strings.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/lcp_type.hpp"
#include "strings/merging.hpp"
#include "strings/string_set.hpp"
#include "test.hpp"
#include "tlx/sort/strings/string_ptr.hpp"
#include "util/printing.hpp"
#include "util/random.hpp"

using namespace dsss;
using namespace kamping;


void test_sorting(Communicator<>& comm) {
    int repeats = 10;
    int n = 1e4;
    mpi::SortingWrapper sorting_wrapper(comm);
    auto call_sorter = [&](std::vector<int>& local_data, Communicator<>& comm) {
        sorting_wrapper.sort(local_data, std::less<>{});
    };
    test::test_sorting(repeats, n, call_sorter, comm);
}

template <typename PDCX, typename char_type, typename index_type>
void test_pdcx(int repeats, int n, int alphabet_size, Communicator<>& comm) {
    int cnt_correct = 0;
    int print_limit = 200;
    dcx::PDCXConfig pdcx_config;
    pdcx_config.atomic_sorter = mpi::AtomicSorters::SampleSort;
    pdcx_config.print_phases = false;

    for (int i = 0; i < repeats; i++) {
        int seed = i * comm.size() + comm.rank();
        std::vector<char_type> local_data =
            dsss::random::generate_random_data<char_type>(n, alphabet_size, seed);
        PDCX pdcx(pdcx_config, comm);
        std::vector<index_type> SA = pdcx.compute_sa(local_data);
        bool sa_ok = check_suffixarray(SA, local_data, comm);
        cnt_correct += sa_ok;

        if (!sa_ok) {
            if (n <= print_limit) {
                print_concatenated(SA, comm, "SA:");
                std::vector<char_type> global_data = comm.gatherv(send_buf(local_data));

                if (comm.rank() == 0) {
                    std::cout << "SA incorrect with seed " << seed << std::endl;
                    std::cout << "input: \n";
                    print_vector(global_data);
                    std::vector<index_type> global_sa =
                        slow_suffixarray<char_type, index_type>(global_data);
                    std::cout << "correct SA: \n";
                    print_vector(global_sa);
                }
            }
            return;
        }
    }
    if (comm.rank() == 0) {
        std::cout << "SA correct: " << cnt_correct << "/" << repeats << std::endl;
    }
}

template <typename PDCX, typename char_type, typename index_type>
void run_tests_pdcx(Communicator<>& comm, std::string test_name = "") {
    if (comm.rank() == 0) {
        std::cout << "Running Tests " + test_name << std::endl;
    }
    std::vector<int> alphabet_size = {2, 8, 32};
    std::vector<std::pair<int, int>> sizes_repeats{{100, 250}, {1000, 100}, {10000, 10}};

    for (auto alpha: alphabet_size) {
        for (auto [n, r]: sizes_repeats) {
            if (comm.rank() == 0) {
                std::cout << "alphabet size: " << alpha << ", n: " << n << ", repeats: " << r
                          << ", ";
            }
            test_pdcx<PDCX, char_type, index_type>(r, n, alpha, comm);
        }
    }
    std::cout << "\n";
}

// test to cover all remainders of total chars mod X
template <typename PDCX, typename char_type, typename index_type>
void run_alignment_tests_pdcx(Communicator<>& comm, std::string test_name = "") {
    if (comm.rank() == 0) {
        std::cout << "Running Alignment Tests " + test_name << std::endl;
    }
    std::vector<int> alphabet_size = {2, 8, 32};
    int n = 1000;
    int r = 5;
    int X = 15;

    for (auto alpha: alphabet_size) {
        for (int d = 0; d < X; d++) {
            int _n = n;
            if (comm.rank() == 0) {
                _n += d;
                std::cout << "alphabet size: " << alpha << ", n: " << _n << ", repeats: " << r
                          << ", ";
            }
            test_pdcx<PDCX, char_type, index_type>(r, _n, alpha, comm);
        }
    }
    std::cout << "\n";
}

void start_tests(Communicator<>& comm) {
    // using char_type = uint16_t;
    // using index_type = uint32_t;

    // run_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC3Param>, char_type, index_type>(
    //     comm,
    //     "pdcx-3");
    // run_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC7Param>, char_type, index_type>(
    //     comm,
    //     "pdcx-7");
    // run_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC13Param>, char_type, index_type>(
    //     comm,
    //     "pdcx-13");

    // run_alignment_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC3Param>, char_type,
    // index_type>(
    //     comm,
    //     "pdcx-3");
    // run_alignment_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC7Param>, char_type,
    // index_type>(
    //     comm,
    //     "pdcx-7");
    // run_alignment_tests_pdcx<dcx::PDCX<char_type, index_type, dcx::DC13Param>, char_type,
    // index_type>(
    //     comm,
    //     "pdcx-13");
}

template <typename PDCX, typename char_type, typename index_type>
void run_pdcx(uint64_t n, uint32_t alphabet_size, Communicator<>& comm) {
    std::vector<char_type> local_data =
        dsss::random::generate_random_data<char_type>(n, alphabet_size, comm.rank());

    dcx::PDCXConfig config;
    PDCX pdcx(config, comm);
    std::vector<index_type> SA = pdcx.compute_sa(local_data);

    pdcx.report_time();
    pdcx.report_stats();
    pdcx.reset();
    bool sa_ok = check_suffixarray(SA, local_data, comm);
    if (comm.rank() == 0) {
        std::cout << "SA_ok=" << sa_ok << "\n";
        std::cout << "\n";
    }
}

void test_transpose_blocks_balanced(Communicator<>& comm) {
    int blocks = 4;
    // int n = 4;
    // int n = 4 + comm.rank();
    // std::vector<int> ns = {0, 0, 10, 10};
    // int n = ns[comm.rank()];
    // std::vector<int> local_data(n * blocks);
    // for (int i = 0; i < blocks; i++) {
    //     for(int j = 0; j < n; j++) {
    //         local_data[i * n + j] = 100 * i + 10 * comm.rank() + j;
    //     }
    // }

    std::vector<std::vector<uint64_t>> B = {{2, 4, 8, 16},
                                            {0, 0, 0, 0},
                                            {0, 0, 0, 0},
                                            {2, 4, 8, 16}};
    std::vector<uint64_t> block_size = B[comm.rank()];
    std::vector<int> local_data;
    for (int i = 0; i < blocks; i++) {
        for (uint64_t j = 0; j < block_size[i]; j++) {
            int x = 1000 * i + 100 * comm.rank() + 10 * j;
            local_data.push_back(x);
        }
    }

    // print_concatenated(local_data, comm, "input_data");
    mpi_util::transpose_blocks_balanced(local_data, block_size, comm);
}

int main() {
    Environment e;
    Communicator comm;

    return 0;
}
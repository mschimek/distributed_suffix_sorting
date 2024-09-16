#include <cstdint>
#include <functional>
#include <iostream>

#include <mpi.h>
#include <sys/types.h>

#include "kamping/communicator.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/named_parameters.hpp"
#include "pdc3.hpp"
#include "printing.hpp"
#include "sa_check.hpp"
#include "sort.hpp"
#include "test.hpp"
#include "util.hpp"

using namespace dsss;
using namespace kamping;


void test_sorting(Communicator<>& comm) {
    int repeats = 10;
    int n = 1e4;
    auto call_sorter = [&](std::vector<int>& local_data, Communicator<>& comm) {
        dsss::mpi::sort(local_data, std::less<>{}, comm);
    };
    test::test_sorting(repeats, n, call_sorter, comm);
}

void test_pdc3(int repeats, int n, int alphabet_size, Communicator<>& comm) {
    int rank = comm.rank();
    int cnt_correct = 0;

    for (int i = 0; i < repeats; i++) {
        int seed = i * comm.size() + rank;
        std::vector<uint32_t> local_data =
            test::generate_random_data<uint32_t>(n, alphabet_size, seed);
        std::vector<uint32_t> local_data_copy = local_data;
        std::vector<uint32_t> global_data = comm.gatherv(send_buf(local_data_copy));

        dc3::PDC3<uint32_t, uint32_t> pdc3(comm);
        auto sa = pdc3.call_pdc3(local_data);

        bool sa_ok = check_suffixarray(sa, local_data_copy, comm);
        cnt_correct += sa_ok;
        if (!sa_ok) {
            print_concatenated(sa, comm);

            if (rank == 0) {
                std::cout << "SA incorrect with seed " << seed << std::endl;
                std::cout << "input: \n";
                print_vector(global_data);

                std::vector<uint32_t> global_sa = slow_suffixarray<uint32_t, uint32_t>(global_data);
                std::cout << "correct SA: \n";
                print_vector(global_sa);
            }
            return;
        }
    }
    if (rank == 0) {
        std::cout << "SA correct: " << cnt_correct << "/" << repeats << std::endl;
    }
}

void run_pdc3(std::vector<uint32_t>& local_data, Communicator<>& comm) {
    // copy without padding, checker should not receive padding
    std::vector<uint32_t> local_data_copy = local_data;
    std::vector<uint32_t> global_data = comm.gatherv(send_buf(local_data_copy));

    if (comm.rank() == 0) {
        print_substrings(global_data);
    }
    print_concatenated(local_data, comm, "local data");

    dc3::PDC3<uint32_t, uint32_t> pdc3(comm);
    auto sa = pdc3.call_pdc3(local_data);

    print_concatenated(sa, comm, "SA");

    bool sa_ok = check_suffixarray(sa, local_data_copy, comm);
    if (comm.rank() == 0) {
        std::cout << "SA ok: " << sa_ok << std::endl;

        std::vector<uint32_t> global_sa = slow_suffixarray<uint32_t, uint32_t>(global_data);
        std::cout << "correct SA: \n";
        print_vector(global_sa);
    }
}

void run_tests_pdc3(Communicator<>& comm) {
    test_pdc3(100, 99, 2, comm);
    test_pdc3(100, 100, 2, comm);
    test_pdc3(100, 101, 2, comm);
    test_pdc3(100, 99, 6, comm);
    test_pdc3(100, 100, 6, comm);
    test_pdc3(100, 101, 6, comm);

    test_pdc3(10, 1e4, 2, comm);
    test_pdc3(10, 1e4, 6, comm);
}

void run_pdc3(Communicator<>& comm) {
    using char_type = uint8_t;
    using index_type = uint32_t;

    int n = 1e8 / comm.size();
    int alphabet_size = 3;
    int seed = comm.rank();
    std::vector<char_type> local_data =
        test::generate_random_data<char_type>(n, alphabet_size, seed);

    dc3::PDC3<char_type, index_type> pdc3(comm);
    pdc3.reset();
    auto sa = pdc3.call_pdc3(local_data);

    pdc3.report_time();
    pdc3.report_memory();
    pdc3.report_stats();

    bool sa_ok = check_suffixarray(sa, local_data, comm);
    if (comm.rank() == 0) {
        std::cout << "sa_ok: " << sa_ok << "\n";
    }
}

int main() {
    Environment e;
    Communicator comm;

    // run_tests_pdc3(comm);
    run_pdc3(comm);

    return 0;
}
// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#include <functional>
#include <iostream>

#include <mpi.h>

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

void test_sorting(kamping::Communicator<>& comm) {
    int repeats = 10;
    int n = 1e4;
    auto call_sorter = [&](std::vector<int>& local_data, kamping::Communicator<>& comm) {
        dsss::mpi::sort(local_data, std::less<>{}, comm);
    };
    test::test_sorting(repeats, n, call_sorter, comm);
}

void test_pdc3(int repeats, int n, int alphabet_size, kamping::Communicator<>& comm) {
    int rank = comm.rank();
    int cnt_correct = 0;

    for (int i = 0; i < repeats; i++) {
        int seed = i * comm.size() + rank;
        std::vector<int> local_data = test::generate_random_data(n, alphabet_size, seed);
        std::vector<int> local_data_copy = local_data;
        std::vector<int> global_data = comm.gatherv(kamping::send_buf(local_data_copy));

        dc3::PDC3 pdc3(comm);
        auto sa = pdc3.call_pdc3(local_data);

        bool sa_ok = check_suffixarray(sa, local_data_copy, comm);
        cnt_correct += sa_ok;
        if (!sa_ok) {
            kamping::print_concatenated(sa, comm);

            if (rank == 0) {
                std::cout << "SA incorrect with seed " << seed << std::endl;
                std::cout << "input: \n";
                kamping::print_vector(global_data);

                std::vector<int> global_sa = slow_suffixarray(global_data);
                std::cout << "correct SA: \n";
                kamping::print_vector(global_sa);
            }
            return;
        }
    }
    if (rank == 0) {
        std::cout << "SA correct: " << cnt_correct << "/" << repeats << std::endl;
    }
}

void run_pdc3(std::vector<int>& local_data, kamping::Communicator<>& comm) {
    // copy without padding, checker should not receive padding
    std::vector<int> local_data_copy = local_data;
    std::vector<int> global_data = comm.gatherv(kamping::send_buf(local_data_copy));

    if (comm.rank() == 0) {
        print_substrings(global_data);
    }
    print_concatenated(local_data, comm, "local data");

    dc3::PDC3 pdc3(comm);
    auto sa = pdc3.call_pdc3(local_data);

    print_concatenated(sa, comm, "SA");

    bool sa_ok = check_suffixarray(sa, local_data_copy, comm);
    if (comm.rank() == 0) {
        std::cout << "SA ok: " << sa_ok << std::endl;

        std::vector<int> global_sa = slow_suffixarray(global_data);
        std::cout << "correct SA: \n";
        kamping::print_vector(global_sa);
    }
}

void run_tests_pdc3(kamping::Communicator<>& comm) {
    test_pdc3(100, 99, 2, comm);
    test_pdc3(100, 100, 2, comm);
    test_pdc3(100, 101, 2, comm);
    test_pdc3(100, 99, 6, comm);
    test_pdc3(100, 100, 6, comm);
    test_pdc3(100, 101, 6, comm);
}


int main() {
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    run_tests_pdc3(comm);


    // int n = 1e5 / comm.size();
    // int alphabet_size = 3;
    // int seed = comm.rank();
    // std::vector<int> local_data = test::generate_random_data(n, alphabet_size, seed);

    // dc3::PDC3 pdc3(comm);
    // auto sa = pdc3.call_pdc3(local_data);

    // bool sa_ok = check_suffixarray(sa, local_data, comm);
    // if (comm.rank() == 0) {
    //     std::cout << "sa_ok: " << sa_ok << "\n";
    // }


    return 0;
}
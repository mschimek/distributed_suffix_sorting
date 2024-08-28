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

#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "pdc3.hpp"
#include "printing.hpp"
#include "sort.hpp"
#include "test.hpp"


void test_sorting(kamping::Communicator<>& comm) {
    int repeats = 10;
    int n = 1e4;
    auto call_sorter = [&](std::vector<int>& local_data, kamping::Communicator<>& comm) {
        dsss::mpi::sort(local_data, std::less<>{}, comm);
    };
    test::test_sorting(repeats, n, call_sorter, comm);
}

int main() {
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    int n = 10;
    int rank = comm.rank();
    int size = comm.size();
    std::vector<int> local_data(n, rank);

    // add padding
    if (rank == size - 1) {
        local_data.push_back(0);
        local_data.push_back(0);
        local_data.push_back(0);
    }

    pdc3(local_data, comm);

    return 0;
}
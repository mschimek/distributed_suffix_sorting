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

#include <mpi.h>

#include <functional>
#include <iostream>
#include <numeric>

#include "helpers_for_examples.hpp"
#include "sort.hpp"

int main() {
  using namespace kamping;

  kamping::Environment e;
  Communicator comm;

  int n = 1000;
  std::vector<int> local_data(n, 0);
  std::iota(local_data.begin(), local_data.end(), 0);

  print_result("Before", comm);
  print_result(local_data.size(), comm);

  dsss::mpi::sort(local_data, std::less<>{}, comm);

  print_result("After", comm);
  print_result(local_data.size(), comm);

  return 0;
}
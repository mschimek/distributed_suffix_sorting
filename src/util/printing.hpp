// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation, either version 3 of
// the License, or (at your option) any later version. KaMPIng is distributed in the hope that it
// will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If
// not, see <https://www.gnu.org/licenses/>.
#pragma once

#include <string>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/communicator.hpp"
#include "mpi/reduce.hpp"


#define V(x) std::string(#x "=") << (x) << " " //"x=...

#define DEBUG_SIZE false
// #define DEBUG true
#define DEBUG false
#define DBG(x) if constexpr(DEBUG) report_on_root(x, comm)


namespace kamping {
/// @brief Print all elements in a container, prefixed with the rank of the current PE.
/// @tparam ContainerType Type of the communicator's default container.
/// @tparam Plugins Types of the communicator's plugins.
/// @tparam T Type of the elements contained in the container.
/// @param result The container whose elements are printed.
/// @param comm KaMPIng communicator to get the rank of the PE.
template <template <typename...> typename ContainerType,
          template <typename, template <typename...> typename>
          typename... Plugins,
          typename T>
void print_result(ContainerType<T> const& result,
                  Communicator<ContainerType, Plugins...> const& comm) {
    for (auto const& elem: result) {
        std::cout << "[PE " << comm.rank() << "] " << elem << "\n";
    }
    std::cout << std::flush;
}

/// @brief Print the given element, prefixed with the rank of the current PE.
/// @tparam T Type of the element.
/// @tparam ContainerType Type of the communicator's default container.
/// @tparam Plugins Types of the communicator's plugins.
/// @param result The elements to be printed. Streamed to std::cout.
/// @param comm KaMPIng communicator to get the rank of the PE.
template <template <typename...> typename ContainerType,
          template <typename, template <typename...> typename>
          typename... Plugins,
          typename T>
void print_result(T const& result, Communicator<ContainerType, Plugins...> const& comm) {
    std::cout << "[PE " << comm.rank() << "] " << result << std::endl;
}

/// @brief Print all elements in a container only on the root PE.
/// @tparam T Type of the elements contained in the container.
/// @param result The container whose elements are printed on the root PE.
/// @param comm KaMPIng communicator to determine which PE is the root PE.
template <template <typename...> typename ContainerType, typename T, typename Comm>
void print_result_on_root(ContainerType<T> const& result, Comm const& comm) {
    if (comm.is_root()) {
        print_result(result, comm);
    }
}

/// @brief Print the given string only on the root PE.
/// @tparam Communicator Type of communicator (has to be a KaMPIng communicator).
/// @param str The string to be printed.
/// @param comm KaMPIng communicator to determine which PE is the root PE.
template <typename Communicator>
void print_on_root(std::string const& str, Communicator const& comm) {
    if (comm.is_root()) {
        print_result(str, comm);
    }
}

// concatenated local vectors and print contents
template <typename T>
void print_concatenated(T const& local_data,
                        Communicator<> const& comm,
                        std::string msg = "",
                        std::string sep = " ") {
    auto [recv_buffer, recv_counts] = comm.allgatherv(send_buf(local_data), recv_counts_out());
    if (comm.is_root()) {
        int i = 0;
        int rank = 0;
        std::cout << msg << "\n";
        for (int cnt: recv_counts) {
            std::cout << "[PE " << rank << "] ";
            for (int j = 0; j < cnt; j++) {
                std::cout << recv_buffer[i++] << sep;
            }
            std::cout << "\n";
            rank++;
        }
        std::cout << std::endl;
    }
}


void print_concatenated_string(std::string const& local_str,
                               Communicator<> const& comm,
                               std::string msg = "") {
    std::vector<char> chars(local_str.begin(), local_str.end());
    print_concatenated(chars, comm, msg, "");
}

// T must implement to_string
template <typename T>
void print_concatenated_string(std::vector<T> const& local_data,
                               Communicator<> const& comm,
                               std::string msg = "",
                               std::string sep = " ") {
    std::string local_str;
    bool is_first = true;
    for (auto& x: local_data) {
        if (!is_first) {
            local_str += sep;
        }
        local_str += x.to_string();
        is_first = false;
    }
    print_concatenated_string(local_str, comm, msg);
}

template <typename T>
void print_concatenated_size(std::vector<T> const& local_data,
                               Communicator<> const& comm,
                               std::string msg = "") {
    print_concatenated_string(std::to_string(local_data.size()), comm, msg);
}

// template <typename Container, typename T>
void print_vector(auto& vec, std::string sep = " ") {
    bool is_first = true;
    for (auto x: vec) {
        if (!is_first) {
            std::cout << sep;
        }
        is_first = false;
        std::cout << std::fixed << std::setprecision(3) << x;
    }
    std::cout << std::endl;
}

void report_min_max_avg(uint64_t local_size,
                        Communicator<>& comm,
                        std::string str = "",
                        uint64_t level = 0) {
    uint64_t total_size = dsss::mpi_util::all_reduce_sum(local_size, comm);
    uint64_t smallest_size = dsss::mpi_util::all_reduce_min(local_size, comm);
    uint64_t largest_size = dsss::mpi_util::all_reduce_max(local_size, comm);
    if (comm.is_root()) {
        std::string pad(level * 2, ' ');
        std::cout << pad << "--> " << str << ": "
                  << "min=" << smallest_size << " max=" << largest_size
                  << " avg=" << (double)total_size / comm.size() << std::endl;
    }
}

void report_on_root(std::string const& str,
                    Communicator<> const& comm,
                    uint64_t level = 0,
                    bool print = true) {
    if (!print)
        return;
    if (comm.is_root()) {
        std::string pad(level * 2, ' ');
        std::cout << pad << str << std::endl;
    }
}

template<typename T>
void report_on_pe(T const& x,
                    Communicator<> const& comm,
                    std::string msg = "") {
    
    std::string s = msg + " " + std::to_string(x);
    print_result(s, comm); 
}

} // namespace kamping
#pragma once

#include <iostream>

#include "kamping/communicator.hpp"

namespace options {

// set flag with cmake .. -DOPTIMIZE_DATA_TYPES=ON
void report_compile_flags(kamping::Communicator<>& comm) {
    if (comm.rank() == 0) {
        std::cout << "Compile Flags: \n";
#ifdef OPTIMIZE_DATA_TYPES
        std::cout << "OPTIMIZE_DATA_TYPES is enabled\n";
        std::cout << "OPTIMIZE_DATA_TYPES=1\n";
#else
        std::cout << "OPTIMIZE_DATA_TYPES is disabled\n";
        std::cout << "OPTIMIZE_DATA_TYPES=0\n";
#endif
#ifdef INCLUDE_ALL_SORTERS
        std::cout << "INCLUDE_ALL_SORTERS is enabled\n";
        std::cout << "INCLUDE_ALL_SORTERS=1\n";
#else
        std::cout << "INCLUDE_ALL_SORTERS is disabled. Using sample sort as fallback.\n";
        std::cout << "INCLUDE_ALL_SORTERS=0\n";
#endif
        std::cout << "\n";
    }
}
} // namespace options
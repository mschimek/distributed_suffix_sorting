#pragma once

#include <iostream>

#include "util/printing.hpp"
namespace dsss::dcx {

struct PDCXConfig {
    double discarding_threshold = 0.5;
    bool use_old_discarding = false;

    void print_config() const {
        std::cout << "PDCXConfig\n";
        std::cout << V(discarding_threshold) << "\n";
        std::cout << V(use_old_discarding) << "\n";
        std::cout << "\n";
    }
};

} // namespace dsss::dcx
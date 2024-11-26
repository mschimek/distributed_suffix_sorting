#pragma once

#include <type_traits>
#include <vector>

#include <sys/resource.h>

namespace dsss {
template <typename T>
void free_memory(T&& to_drop) {
    std::remove_reference_t<T>(std::move(to_drop));
}

long get_max_mem_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

long get_max_mem_bytes() { return 1000 * get_max_mem_kb(); }


} // namespace dsss
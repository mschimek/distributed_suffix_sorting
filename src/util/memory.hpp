#pragma once

#include <type_traits>
#include <vector>

namespace dsss {
template <typename T>
void free_memory(T&& to_drop) {
    std::remove_reference_t<T>(std::move(to_drop));
}

} // namespace dsss
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dsss::monitor {


struct MemoryKey {
    int64_t memory;
    std::string key;

    int64_t get_memory_bytes() const { return memory; }

    double get_memory_mb() const {
        double memory_mb = (double)memory / 1e6;
        return memory_mb;
    }

    std::string get_memory_mb_string() const {
        double memory_mb = get_memory_mb();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << memory_mb;
        return ss.str();
    }

    std::string to_string_mb() {
        std::string str = get_memory_mb_string() + " MB at " + key;
        return str;
    }

    bool operator<(MemoryKey const& other) const { return memory < other.memory; }
};

struct MemoryMonitor {
    MemoryMonitor() : current_memory(0), memory_history() {}

    void reset() {
        current_memory = 0;
        memory_history.clear();
    }

    template <typename T>
    int64_t get_memory_vector(std::vector<T>& v) const {
        return sizeof(T) * v.capacity() + sizeof(std::vector<T>);
    }

    template <typename T>
    void add_memory(std::vector<T>& vec, std::string key = "") {
        int64_t memory = get_memory_vector(vec);
        current_memory += memory;
        memory_history.push_back({current_memory, "add_" + key});
    }

    template <typename T>
    void remove_memory(std::vector<T>& vec, std::string key = "") {
        int64_t memory = get_memory_vector(vec);
        current_memory -= memory;
        memory_history.push_back({current_memory, "rm_" + key});
    }

    std::string history_mb_to_string() {
        std::string out = "";
        for (auto& m: memory_history) {
            out += m.to_string_mb() + "\n";
        }
        return out;
    }

    MemoryKey get_peak_memory() const {
        return *std::max_element(memory_history.begin(), memory_history.end());
    }

    // in bytes
    int64_t current_memory;
    std::vector<MemoryKey> memory_history;
};

// singleton instance
inline MemoryMonitor& get_monitor_instance() {
    static MemoryMonitor monitor;
    return monitor;
}
} // namespace dsss::monitor
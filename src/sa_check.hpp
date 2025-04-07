#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/distribute.hpp"
#include "mpi/reduce.hpp"
#include "mpi/shift.hpp"
#include "mpi/zip.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/lcp_type.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"

namespace dsss {

inline void report_memory_usage(kamping::Communicator<> const& comm, const std::string& msg, bool output_rss_from_all_pes) {
    kamping::report_on_root("Memory Usage:" + msg, comm);
    uint64_t max_mem = dsss::get_max_mem_bytes();
    uint64_t max_rss = comm.allreduce_single(kamping::send_buf(max_mem), kamping::op(kamping::ops::max<>{}));
    kamping::report_on_root("max_rss_pe=" + std::to_string(max_rss), comm);

    if (output_rss_from_all_pes) {
        auto all_mem = comm.gather(kamping::send_buf(max_mem));
        if (comm.rank() == 0) {
            std::cout << "max_mem_pe=";
            kamping::print_vector(all_mem, ",");
            std::cout << std::endl;
        }
    }
}

inline void report_size(kamping::Communicator<> const& comm, const std::string& msg, std::size_t size) {
    kamping::report_on_root("Size Dist:" + msg, comm);
    uint64_t max_rss = comm.allreduce_single(kamping::send_buf(size), kamping::op(kamping::ops::max<>{}));
    kamping::report_on_root("max_size_pe=" + std::to_string(max_rss), comm);

    if (true) {
        auto all_mem = comm.gather(kamping::send_buf(size));
        if (comm.rank() == 0) {
            std::cout << "max_size_pe=";
            kamping::print_vector(all_mem, ",");
            std::cout << std::endl;
        }
    }
}

// adapated from: https://github.com/kurpicz/dsss/blob/master/dsss/suffix_sorting/sa_check.hpp
// Roman Dementiev, Juha Kärkkäinen, Jens Mehnert, and Peter Sanders. 2008. Better external memory
// suffix array construction.
template <typename IndexType, typename CharType>
bool check_suffixarray(std::vector<IndexType>& sa,
                       std::vector<CharType>& text,
                       kamping::Communicator<>& comm) {
    using namespace kamping;

    mpi::SortingWrapper sorting_wrapper(comm);
    sorting_wrapper.set_sorter(mpi::AtomicSorters::Ams);
    sorting_wrapper.finalize_setting();

    bool is_correct = true;
    size_t local_size_sa = sa.size();
    size_t local_size_text = text.size();
    size_t global_size_sa = mpi_util::all_reduce_sum(local_size_sa, comm);
    size_t global_size_text = mpi_util::all_reduce_sum(local_size_text, comm);

    if (global_size_text != global_size_sa) {
        print_on_root("SA and text size don't match: " + std::to_string(global_size_sa)
                          + " != " + std::to_string(global_size_text),
                      comm);
        return false;
    }

    struct sa_tuple {
        IndexType rank;
        IndexType sa;
    };

    struct rank_triple {
        IndexType rank1;
        IndexType rank2;
        CharType chr;

        bool operator<(const rank_triple& other) const {
            return std::tie(chr, rank2) < std::tie(other.chr, other.rank2);
        }

        bool operator<=(const rank_triple& other) const {
            return std::tie(chr, rank2) <= std::tie(other.chr, other.rank2);
        }
    };

    // index sa with 1, ..., n
    auto index_function = [](uint64_t idx, IndexType sa_idx) {
        return sa_tuple{1 + IndexType(idx), sa_idx};
    };

    std::vector<sa_tuple> sa_tuples =
        mpi_util::zip_with_index<IndexType, sa_tuple>(sa, index_function, comm);

    report_memory_usage(kamping::comm_world(), "before sort 1", false);

    sorting_wrapper.sort(sa_tuples,
                         [](const sa_tuple& a, const sa_tuple& b) { return a.sa < b.sa; });

    report_memory_usage(kamping::comm_world(), "after sort 1", false);
    sa_tuples = mpi_util::distribute_data(sa_tuples, comm);
    report_memory_usage(kamping::comm_world(), "after redistribute 1", false);
    text = mpi_util::distribute_data(text, comm);
    report_memory_usage(kamping::comm_world(), "after redistribute 2", false);
    comm.barrier();

    size_t local_size = sa_tuples.size();
    size_t offset = mpi_util::ex_prefix_sum(local_size, comm);

    bool is_permutation = true;
    for (size_t i = 0; i < local_size; ++i) {
        is_permutation &= (sa_tuples[i].sa == IndexType(i + offset));
    }

    is_correct = mpi_util::all_reduce_and(is_permutation, comm);
    if (!is_correct) {
        print_on_root("no permutation", comm);
        return false;
    }

    sa_tuple tuple_to_right = mpi_util::shift_left(sa_tuples.front(), comm);
    if (comm.rank() + 1 < comm.size()) {
        sa_tuples.emplace_back(tuple_to_right);
    } else {
        sa_tuples.emplace_back(sa_tuple{0, 0});
    }

    report_memory_usage(kamping::comm_world(), "after shift", false);
    
    report_size(kamping::comm_world(), "local size", local_size);
    report_size(kamping::comm_world(), "local size sa tuples", sa_tuples.size());
    std::vector<rank_triple> rts;
    for (size_t i = 0; i < local_size; ++i) {
        rts.emplace_back(rank_triple{sa_tuples[i].rank, sa_tuples[i + 1].rank, text[i]});
    }
    report_memory_usage(kamping::comm_world(), "before free", false);
    free_memory(std::move(sa_tuples));
    report_size(kamping::comm_world(), "before sort rank triple", rts.size());

    sorting_wrapper.sort(rts, [](const rank_triple& a, const rank_triple& b) {
        return a.rank1 < b.rank1;
    });
    report_memory_usage(kamping::comm_world(), "after sort rank triple", false);

    // avoid empty PEs for small input
    rts = mpi_util::distribute_data(rts, comm);
    report_memory_usage(kamping::comm_world(), "after redistribute data", false);

    local_size = rts.size();
    bool is_sorted = true;
    for (int64_t i = 0; i < int64_t(local_size) - 1; ++i) {
        is_sorted &= (rts[i] <= rts[i + 1]);
    }

    KASSERT(rts.size() > 0ull);
    auto smaller_triple = mpi_util::shift_right(rts.back(), comm);
    auto larger_triple = mpi_util::shift_left(rts.front(), comm);

    report_memory_usage(kamping::comm_world(), "after shift 2", false);
    if (comm.rank() > 0) {
        is_sorted &= (smaller_triple < rts.front());
    }
    if (comm.rank() + 1 < comm.size()) {
        is_sorted &= (rts.back() < larger_triple);
    }

    is_correct = mpi_util::all_reduce_and(is_sorted, comm);

    if (!is_correct) {
        print_on_root("not sorted", comm);
    }
    return is_correct;
}

// a little more space efficient version of SA-checking algorithm
// write sa_tuple in rank_triple vector
template <typename IndexType, typename CharType>
bool check_suffixarray2(std::vector<IndexType>& sa,
                        std::vector<CharType>& text,
                        kamping::Communicator<>& comm) {
    using namespace kamping;

    mpi::SortingWrapper sorting_wrapper(comm);
    sorting_wrapper.set_sorter(mpi::AtomicSorters::Ams);

    bool is_correct = true;

    size_t local_size_sa = sa.size();
    size_t local_size_text = text.size();
    size_t global_size_sa = mpi_util::all_reduce_sum(local_size_sa, comm);
    size_t global_size_text = mpi_util::all_reduce_sum(local_size_text, comm);

    if (global_size_text != global_size_sa) {
        print_on_root("SA and text size don't match: " + std::to_string(global_size_sa)
                          + " != " + std::to_string(global_size_text),
                      comm);
        return false;
    }

    struct rank_triple {
        IndexType rank1; // or rank
        IndexType rank2; // or sa
        CharType chr;

        bool operator<(const rank_triple& other) const {
            return std::tie(chr, rank2) < std::tie(other.chr, other.rank2);
        }

        bool operator<=(const rank_triple& other) const {
            return std::tie(chr, rank2) <= std::tie(other.chr, other.rank2);
        }

        // assume rank is store in rank1
        IndexType get_rank() const { return rank1; }

        // assume sa is store in rank2
        IndexType get_sa() const { return rank2; }

        static bool cmp_sa(const rank_triple& a, const rank_triple& b) {
            return a.get_sa() < b.get_sa();
        }
    };

    // index sa with 1, ..., n
    auto index_function = [](uint64_t idx, IndexType sa_idx) {
        return rank_triple{1 + IndexType(idx), sa_idx, 0};
    };

    // sa_tuples: (rank, sa, _)
    std::vector<rank_triple> rts =
        mpi_util::zip_with_index<IndexType, rank_triple>(sa, index_function, comm);

    sorting_wrapper.sort(rts, rank_triple::cmp_sa);

    rts = mpi_util::distribute_data(rts, comm);
    text = mpi_util::distribute_data(text, comm);
    comm.barrier();

    size_t local_size = rts.size();
    size_t offset = mpi_util::ex_prefix_sum(local_size, comm);

    bool is_permutation = true;
    for (size_t i = 0; i < local_size; ++i) {
        is_permutation &= (rts[i].get_sa() == IndexType(i + offset));
    }

    is_correct = mpi_util::all_reduce_and(is_permutation, comm);
    if (!is_correct) {
        print_on_root("no permutation", comm);
        return false;
    }

    rank_triple tuple_to_right = mpi_util::shift_left(rts.front(), comm);
    rts.reserve(rts.size() + 1);

    if (comm.rank() + 1 < comm.size()) {
        rts.emplace_back(tuple_to_right);
    } else {
        rts.emplace_back(rank_triple{0, 0, 0});
    }

    for (size_t i = 0; i < local_size; ++i) {
        rts[i] = {rts[i].get_rank(), rts[i + 1].get_rank(), text[i]};
    }
    // pop dummy
    rts.pop_back();
    rts.shrink_to_fit();

    sorting_wrapper.sort(rts, [](const rank_triple& a, const rank_triple& b) {
        return a.rank1 < b.rank1;
    });

    local_size = rts.size();
    bool is_sorted = true;
    for (size_t i = 0; i < local_size - 1; ++i) {
        is_sorted &= (rts[i] <= rts[i + 1]);
    }

    auto smaller_triple = mpi_util::shift_right(rts.back(), comm);
    auto larger_triple = mpi_util::shift_left(rts.front(), comm);

    if (comm.rank() > 0) {
        is_sorted &= (smaller_triple < rts.front());
    }
    if (comm.rank() + 1 < comm.size()) {
        is_sorted &= (rts.back() < larger_triple);
    }

    is_correct = mpi_util::all_reduce_and(is_sorted, comm);

    if (!is_correct) {
        print_on_root("not sorted", comm);
    }
    return is_correct;
}


template <typename T>
bool check_sorted(std::vector<T>& v, auto less, kamping::Communicator<>& comm) {
    using namespace kamping;

    uint64_t total_size = mpi_util::all_reduce_sum(v.size(), comm);
    if(total_size <= 10000) {
        auto w = comm.allgatherv(send_buf(v));
        bool ok = std::is_sorted(w.begin(), w.end(), less);
        return ok;
    }

    if (v.size() == 0) {
        std::cout << "Warning: empty vector in sorted check" << std::endl;
    }
    KASSERT(v.size() > 0ull);

    T next = mpi_util::shift_left(v.back(), comm);
    bool ok = true;

    for (int64_t i = 0; i < (int64_t)v.size() - 1; i++) {
        ok &= !less(v[i + 1], v[i]);
        if (!ok) {
            std::cout << "[PE " << comm.rank() << "] " << i << " " << v[i].to_string() << " "
                      << v[i + 1].to_string() << std::endl;
            break;
        }
    }
    if (ok && comm.rank() < comm.size() - 1) {
        ok &= !less(next, v.back());
        if (!ok) {
            std::cout << "[PE " << comm.rank() << "] --> overlapping" << v.back().to_string() << " "
                      << next.to_string() << std::endl;
        }
    }
    bool all_ok = mpi_util::all_reduce_and(ok, comm);
    return all_ok;
}

template <typename T>
bool check_vector_same(std::vector<T>& a,
                       std::vector<T>& b,
                       auto eq,
                       kamping::Communicator<>& comm) {
    // check for same total size
    uint64_t total_size_a = mpi_util::all_reduce_sum(a.size(), comm);
    uint64_t total_size_b = mpi_util::all_reduce_sum(b.size(), comm);
    bool ok = total_size_a == total_size_b;
    if (!ok) {
        kamping::report_on_root("vector sizes not equal: " + std::to_string(total_size_a) + " "
                                    + std::to_string(total_size_b),
                                comm);
        return false;
    }

    // align local vector and compare elementwise
    uint64_t local_size_a = a.size();
    uint64_t local_size_b = b.size();
    a = mpi_util::distribute_data(a, comm);
    b = mpi_util::distribute_data(b, comm);
    for (uint64_t i = 0; i < a.size(); i++) {
        ok &= eq(a[i], b[i]);
        if (!ok) {
            kamping::report_on_root("vector elements not equal at " + std::to_string(i) + " "
                                        + a[i].to_string() + " " + b[i].to_string(),
                                    comm);
            break;
        }
    }
    bool all_ok = mpi_util::all_reduce_and(ok, comm);

    // reverse modification made to vectors
    a = mpi_util::distribute_data_custom(a, local_size_a, comm);
    b = mpi_util::distribute_data_custom(b, local_size_b, comm);
    return all_ok;
}

template <typename char_type, typename lcp_type>
bool check_lcp_values(std::vector<char_type>& local_string,
                      std::vector<lcp_type>& lcps,
                      kamping::Communicator<>& comm,
                      bool strict = true) {
    KASSERT(local_string.size() == lcps.size());
    bool ok = local_string.size() == lcps.size();

    for (uint64_t i = 1; i < local_string.size(); i++) {
        LcpType common_prefix = 0;
        int64_t result = string_cmp(local_string[i - 1], local_string[i], common_prefix);
        if (result == 0) {
            common_prefix++; // tlx also counts 0-character at the end
        }
        if (strict) {
            ok &= (common_prefix == lcps[i]);
        } else {
            // lcp is exact, but does not yield wrong results
            ok &= (common_prefix >= lcps[i]);
        }
        // if(!ok) {
        if (common_prefix < lcps[i]) {
            // if(common_prefix != lcps[i]) {
            std::string msg = "lcp values wrong at " + std::to_string(i)
                              + " lcp: " + std::to_string(lcps[i])
                              + ", correct lcp: " + std::to_string(common_prefix);
            msg += " " + local_string[i - 1].to_string() + " " + local_string[i].to_string();
            kamping::print_result(msg, comm);
            break;
        }
    }

    bool all_ok = mpi_util::all_reduce_and(ok, comm);
    return all_ok;
}

} // namespace dsss

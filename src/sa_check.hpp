#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "mpi_util.hpp"
#include "printing.hpp"
#include "sort.hpp"

namespace dsss {

// adapated from: https://github.com/kurpicz/dsss/blob/master/dsss/suffix_sorting/sa_check.hpp
// Roman Dementiev, Juha Kärkkäinen, Jens Mehnert, and Peter Sanders. 2008. Better external memory
// suffix array construction.
template <typename IndexType, typename CharType>
bool check_suffixarray(std::vector<IndexType>& sa,
                       std::vector<CharType>& text,
                       kamping::Communicator<>& comm) {
    using namespace kamping;

    bool is_correct = true;

    if (sa.size() == 0) {
        print_on_root("SA is empty", comm);
        return false;
    }
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
    auto index_function = [](IndexType idx, IndexType sa_idx) { return sa_tuple{1 + idx, sa_idx}; };
    std::vector<sa_tuple> sa_tuples = mpi_util::zip_with_index<IndexType, sa_tuple>(sa, index_function, comm);

    mpi::sort(
        sa_tuples,
        [](const sa_tuple& a, const sa_tuple& b) { return a.sa < b.sa; },
        comm);
    sa_tuples = mpi_util::distribute_data(sa_tuples, comm);
    text = mpi_util::distribute_data(text, comm);
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

    std::vector<rank_triple> rts;
    for (size_t i = 0; i < local_size; ++i) {
        rts.emplace_back(rank_triple{sa_tuples[i].rank, sa_tuples[i + 1].rank, text[i]});
    }

    mpi::sort(
        rts,
        [](const rank_triple& a, const rank_triple& b) { return a.rank1 < b.rank1; },
        comm);

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

} // namespace dsss

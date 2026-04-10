#include "dcx/dcx_instantiations.hpp"

#include "pdcx/pdcx.hpp"
#include "strings/char_container.hpp"

std::vector<DC39Algorithm_uint8_3bit_packing::index_t>
DC39Algorithm_uint8_3bit_packing::compute_suffix_array(std::vector<char_t>& local_string,
                                                       kamping::Communicator<>& comm) {
    using DCXParam = dsss::dcx::DC39Param;
    using CharContainer = KPackedInteger<2, char_t, BITS_CHAR, uint64_t>;
    using PDCXVariant = dsss::dcx::PDCX<char_t, index_t, DCXParam, CharContainer, CharContainer>;

    auto algo = PDCXVariant(pdcx_config, comm);
    auto local_suffix_array = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
    return local_suffix_array;
}

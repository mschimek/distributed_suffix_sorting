#include "dcx/dcx_instantiations.hpp"

#include "pdcx/pdcx.hpp"

std::vector<DC39Algorithm_uint8_unpacked::index_t>
DC39Algorithm_uint8_unpacked::compute_suffix_array(std::vector<char_t>& local_string,
                                                   kamping::Communicator<>& comm) {
    using DCXParam = dsss::dcx::DC39Param;
    using PDCXVariant = dsss::dcx::PDCX<char_t, index_t, DCXParam>;

    auto algo = PDCXVariant(pdcx_config, comm);
    auto local_suffix_array = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
    return local_suffix_array;
}

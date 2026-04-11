#include "dcx/dcx_instantiations.hpp"

#include "pdcx/pdcx.hpp"

using namespace dsss::dcx;

template <>
std::vector<DC39_u8::index_t>
DC39_u8::compute_suffix_array(std::vector<uint8_t>& local_string,
                              kamping::Communicator<>& comm) {
    using PDCXVariant = dsss::dcx::PDCX<uint8_t, index_t, dsss::dcx::DC39Param>;

    auto algo = PDCXVariant(pdcx_config, comm);
    auto local_suffix_array = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
    return local_suffix_array;
}

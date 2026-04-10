
#include "dcx/dcx.hpp"
#include "dcx_common.hpp"

namespace dsss::dcx {

std::vector<dsss::UIntPair<uint8_t>> get_sa(
    std::vector<uint8_t>& input,
    kamping::Communicator<>& comm,
    int32_t argc,
    char const* argv[]) {
    PDCXConfig config = common::parse_pdcx_config(argc, argv, /*allow_extras=*/true);
    std::vector<dsss::UIntPair<uint8_t>> local_sa;
    common::compute_sa(comm, config, input, local_sa);
    return local_sa;
}

std::vector<dsss::UIntPair<uint8_t>> get_sa(
    std::vector<dsss::UIntPair<uint8_t>>& input,
    kamping::Communicator<>& comm,
    int32_t argc,
    char const* argv[]) {
    PDCXConfig config = common::parse_pdcx_config(argc, argv, /*allow_extras=*/true);
    DC39_u40 algo(config);
    return algo.compute_suffix_array(input, comm);
}

} // namespace dsss::dcx

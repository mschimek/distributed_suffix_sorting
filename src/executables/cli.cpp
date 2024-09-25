
#include <tlx/cmdline_parser.hpp>

#include "kamping/communicator.hpp"
#include "mpi/io.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/pdcx.hpp"
#include "sa_check.hpp"
#include "util/random.hpp"
#include "util/uint_types.hpp"

#define V(x) std::string(#x "=") << (x) << " " //"x=...

using char_type = uint8_t;
using index_type = dsss::uint40;

size_t string_size = {0};
size_t alphabet_size = {2};
size_t seed = {0};
std::string input_path = "";
std::string output_path = "";
std::string dcx_variant = "dc3";
bool check = false;

tlx::CmdlineParser cp;
std::vector<char_type> local_string;
std::vector<index_type> local_sa;

void configure_cli() {
    cp.set_description("Distributed Suffix Array Construction using pDCX");
    cp.set_author("Manuel Haag <uozeb@student.kit.edu>");

    cp.add_param_string("input",
                        input_path,
                        "Path to input file. The special input 'random' generates a random text of "
                        "the size given by parameter '-s'.");

    cp.add_bytes('s',
                 "size",
                 string_size,
                 "Size (in bytes unless stated "
                 "otherwise) of the string that use to test our suffix array "
                 "construction algorithms.");

    cp.add_bytes('a', "alphabet_size", alphabet_size, "Size of the alphbet used for random.");
    cp.add_bytes('e', "seed", seed, "Seed to be used for random. PE i uses seed: seed + i");

    cp.add_flag('c', "check", check, "Check if the SA has been constructed correctly.");

    cp.add_string('o',
                  "output",
                  "<F>",
                  output_path,
                  "Filename for the output (SA). Note that the output is five times larger than "
                  "the input file.");

    cp.add_string('x',
                  "dcx",
                  "<F>",
                  dcx_variant,
                  "pDCX variant to use. Available options: dc3, dc7, dc13.");
}

void report_arguments(kamping::Communicator<>& comm) {
    if (comm.rank() == 0) {
        std::cout << "Arguments:\n";
        std::cout << V(string_size) << "\n";
        std::cout << V(alphabet_size) << "\n";
        std::cout << V(seed) << "\n";
        std::cout << V(input_path) << "\n";
        std::cout << V(output_path) << "\n";
        std::cout << V(dcx_variant) << "\n";
        std::cout << V(check) << "\n";
        std::cout << "\n";
    }
}

void read_input(kamping::Communicator<>& comm) {
    if (!input_path.compare("random")) {
        string_size /= comm.size();
        uint64_t local_seed = seed + comm.rank();
        local_string =
            dsss::random::generate_random_data<char_type>(string_size, alphabet_size, local_seed);
    } else {
        if (string_size > 0) {
            local_string = dsss::mpi::read_and_distribute_string(input_path, comm, string_size);
        } else {
            local_string = dsss::mpi::read_and_distribute_string(input_path, comm);
        }
    }
}

template <typename PDCX, typename char_type, typename index_type>
void run_pdcx(kamping::Communicator<>& comm) {
    auto algo = PDCX(comm);
    local_sa = algo.compute_sa(local_string);
    algo.report_stats();
    comm.barrier();
    algo.report_time();
}

void compute_sa(kamping::Communicator<>& comm) {
    if (dcx_variant == "dc3") {
        run_pdcx<dsss::dcx::PDCX<char_type, index_type, DC3Param>, char_type, index_type>(comm);
    }
    // else if (dcx_variant == "dc7") {
    //     run_pdcx<dsss::dcx::PDCX<char_type, index_type, DC7Param>, char_type,
    //     index_type>(comm);
    // } else {
    //     run_pdcx<dsss::dcx::PDCX<char_type, index_type, DC13Param>, char_type,
    //     index_type>(comm);
    // }
}

void write_sa(kamping::Communicator<>& comm) {
    if (!output_path.empty()) {
        if (comm.rank() == 0) {
            std::cout << "Writing the SA to " << output_path << std::endl;
        }
        dsss::mpi::write_data(local_sa, output_path, comm);
        comm.barrier();
        if (comm.rank() == 0) {
            std::cout << "Finished writing the SA" << std::endl;
        }
    }
}

void check_sa(kamping::Communicator<>& comm) {
    if (check) {
        if (comm.rank() == 0) {
            std::cout << "Checking SA ... ";
        }
        // assuming algorithm did not change local string
        bool correct = dsss::check_suffixarray(local_sa, local_string, comm);

        if (comm.rank() == 0) {
            std::string msg = correct ? "Correct SA!" : "ERROR: Not a correct SA!";
            std::cout << msg << "\n";
            std::cout << "SA_ok=" << correct << "\n";
        }
    }
}

int main(int32_t argc, char const* argv[]) {
    kamping::Environment e;
    kamping::Communicator comm;

    configure_cli();
    if (!cp.process(argc, argv)) {
        return -1;
    }
    report_arguments(comm);
    read_input(comm);
    compute_sa(comm);
    check_sa(comm);
    write_sa(comm);

    return 0;
}

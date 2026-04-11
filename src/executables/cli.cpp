
#include <cstdint>
#include <string>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/named_parameters.hpp>

#include "CLI_mpi.hpp"
#include "dcx_common.hpp"
#include "mpi/io.hpp"
#include "mpi/reduce.hpp"
#include "options.hpp"
#include "pdcx/statistics.hpp"
#include "sa_check.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/random.hpp"
#include "util/uint_types.hpp"

using namespace dsss;

inline void print_as_jsonlist_to_file(std::vector<std::string> objects, std::string filename) {
    std::ofstream outstream(filename);
    outstream << "[" << std::endl;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (i > 0) {
            outstream << "," << std::endl;
        }
        outstream << objects[i];
    }
    outstream << std::endl << "]" << std::endl;
}

struct Parameters {
    size_t textsize = 0u;
    size_t alphabet_size = 2u;
    size_t external_iteration = 0;
    std::string algorithm = "DCX";
    std::string input_path = "";
    std::string output_path = "";
    std::string json_output_path = "";
    std::string dcx_variant = "dc3";
    std::size_t num_pe = 0u;
    bool check = false;
    dcx::PDCXConfig pdcx_config;


    std::vector<std::pair<std::string, std::string>> config() const {
        std::vector<std::pair<std::string, std::string>> config_vector;
        config_vector.emplace_back("textsize", std::to_string(textsize));
        config_vector.emplace_back("alphabet_size", std::to_string(alphabet_size));
        config_vector.emplace_back("num_pe", std::to_string(num_pe));
        config_vector.emplace_back("input_path", input_path);
        config_vector.emplace_back("output_path", output_path);
        config_vector.emplace_back("json_output_path", json_output_path);
        config_vector.emplace_back("dcx_variant", dcx_variant);
        config_vector.emplace_back("algorithm", algorithm);
        config_vector.emplace_back("external_iteration", std::to_string(external_iteration));
        {
            auto pdcx_confi_vector = pdcx_config.config();
            config_vector.insert(config_vector.end(),
                                 pdcx_confi_vector.begin(),
                                 pdcx_confi_vector.end());
        }

        return config_vector;
    }

    friend std::ostream& operator<<(std::ostream& out, Parameters const& params) {
        for (auto const& config_entry: params.config()) {
            out << config_entry.first << "=" << config_entry.second << " ";
        }
        return out;
    }
};


Parameters read_cli_parameters(int argc, char const** argv) {
    Parameters parameters;
    CLI::App app{"Suffix Sorting Benchmark"};
    app.add_option("--input",
                   parameters.input_path,
                   "Path to input file. The special input 'random' generates a random text of the "
                   "size given by parameter '-s'.");
    app.add_option("--textsize",
                   parameters.textsize,
                   "Size (in bytes unless stated otherwise) of the string that use to test our "
                   "suffix array construction algorithms.");
    app.add_option("--iteration",
                   parameters.external_iteration,
                   "Helper Argument for benchmarking.");
    app.add_option("--alphabet_size",
                   parameters.alphabet_size,
                   "Size of the alphbet used for random.");
    app.add_option("--output",
                   parameters.output_path,
                   "Filename for the output (SA). Note that the output is five times larger than "
                   "the input file.");
    app.add_option("--json-output-path", parameters.json_output_path, "path to json output");

    // CLI-specific options
    app.add_flag("--check", parameters.check, "Check if the SA has been constructed correctly.");
    app.add_option("--dcx",
                   parameters.dcx_variant,
                   "DCX variant to use. Available options: dc3, dc7, dc13, ..., dc133.");

    // PDCXConfig options (shared with library)
    dcx::common::add_pdcx_options(app, parameters.pdcx_config);

    CLI11_PARSE_MPI(app, argc, argv);
    parameters.num_pe = kamping::comm_world().size();
    return parameters;
}

template <typename char_t>
std::vector<char_t> read_input(kamping::Communicator<>& comm, Parameters const& parameters) {
    if (parameters.input_path != "random" && !mpi::file_exists(parameters.input_path)) {
        if (comm.rank() == 0) {
            std::cerr << "File " << parameters.input_path << " does not exist!" << std::endl;
        }
        exit(1);
    }
    std::vector<char_t> local_string;
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("io");
    local_string =
        mpi::read_and_distribute_string<char_t>(parameters.input_path, comm, parameters.textsize);
    timer.stop();
    kamping::report_on_root("\n", comm);
    return local_string;
}

// compress_alphabet, run_pdcx, run_packed_dcx_variant, select_dcx_variant,
// select_packed_dcx_variant, and compute_sa are provided by dcx_common.hpp
using dcx::common::compute_sa;

template <typename index_t>
void write_sa(kamping::Communicator<>& comm,
              Parameters const& params,
              std::vector<index_t>& local_sa) {
    if (!params.output_path.empty()) {
        kamping::report_on_root("Writing the SA to " + params.output_path + "\n", comm);
        mpi::write_data(local_sa, params.output_path, comm);
        comm.barrier();
        kamping::report_on_root("Finished writing the SA \n", comm);
    }
}

template <typename char_t, typename index_t>
void check_sa(kamping::Communicator<>& comm,
              Parameters const& params,
              std::vector<char_t>& local_string,
              std::vector<index_t>& local_sa) {
    using namespace kamping;
    if (params.check) {
        measurements::Timer<Communicator<>> check_timer;
        measurements::timer().synchronize_and_start("check_SA");
        check_timer.synchronize_and_start("check_SA");

        // TODO maybe read again
        local_string = read_input<char_t>(comm, params);

        kamping::report_on_root("Checking SA ... ", comm);
        // assuming algorithm did not change local string
        bool correct = check_suffixarray(local_sa, local_string, comm);
        // bool correct = check_suffixarray2(local_sa, local_string, comm);
        // bool correct2 = check_suffixarray2(local_sa, local_string, comm);

        if (comm.rank() == 0) {
            std::string msg = correct ? "Correct SA!" : "ERROR: Not a correct SA!";
            std::cout << msg << std::endl;
            std::cout << "SA_ok=" << correct << std::endl;
            // std::cout << "SA_ok2=" << correct2 << std::endl;
        }
        check_timer.stop();
        measurements::timer().stop();
        check_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
        kamping::report_on_root("\n", comm);
    }
}

template <typename char_t>
void report_memory_usage(kamping::Communicator<>& comm,
                         std::vector<char_t> const& local_string,
                         bool output_rss_from_all_pes = false) {
    kamping::report_on_root("Memory Usage:", comm);
    uint64_t max_mem = dsss::get_max_mem_bytes();
    uint64_t max_rss = mpi_util::all_reduce_max(max_mem, comm);
    double blowup = (double)max_rss / local_string.size();
    kamping::report_on_root("max_rss_pe=" + std::to_string(max_rss), comm);
    kamping::report_on_root("blowup_pe=" + std::to_string(blowup), comm);

    if (output_rss_from_all_pes) {
        auto all_mem = comm.gather(kamping::send_buf(max_mem));
        if (comm.rank() == 0) {
            std::cout << "max_mem_pe=";
            kamping::print_vector(all_mem, ",");
            std::cout << std::endl;
        }
    }
}

template <typename char_t, typename index_t>
void run_pipeline(kamping::Communicator<>& comm, Parameters const& params) {
    uint64_t max_mem_before_input = dsss::get_max_mem_bytes();
    auto& timer = kamping::measurements::timer();
    timer.clear();
    kamping::measurements::counter().add("mem_before_reading_input",
                                         max_mem_before_input,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    std::vector<char_t> local_string = read_input<char_t>(comm, params);
    std::vector<index_t> local_sa;
    kamping::measurements::counter().add("mem_before_sa_construction",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    compute_sa(comm, params.pdcx_config, local_string, local_sa);
    dsss::dcx::get_local_stats_instance().commit();
    dsss::dcx::get_local_stats_instance().reset();
    kamping::measurements::counter().add("mem_after_sa_construction",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    report_memory_usage(comm, local_string);
    check_sa(comm, params, local_string, local_sa);

    kamping::measurements::counter().add("mem_after_sa_check",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    write_sa(comm, params, local_sa);
}

int main(int32_t argc, char const* argv[]) {
    uint64_t max_mem_start = dsss::get_max_mem_bytes();
    kamping::Environment e;
    kamping::Communicator comm;
    kamping::measurements::counter().clear();
    uint64_t max_mem_init = dsss::get_max_mem_bytes();
    kamping::measurements::counter().add("mem_program_start",
                                         max_mem_start,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    kamping::measurements::counter().add("mem_program_after_init",
                                         max_mem_init,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    options::report_compile_flags(comm);

    Parameters const params = read_cli_parameters(argc, argv);

    run_pipeline<uint8_t, uint40>(comm, params);

    // print
    auto config_vector = params.config();
    std::stringstream sstream_counter;
    std::stringstream sstream_timer;
    kamping::measurements::SimpleJsonPrinter<double> printer_timer(sstream_timer, config_vector);
    kamping::measurements::SimpleJsonPrinter<std::int64_t> printer_counter(sstream_counter,
                                                                           config_vector);
    kamping::measurements::timer().aggregate_and_print(printer_timer);
    kamping::measurements::counter().aggregate_and_print(printer_counter);

    if (comm.rank() == 0) {
        print_as_jsonlist_to_file({sstream_timer.str()}, params.json_output_path + "_timer.json");
        print_as_jsonlist_to_file({sstream_counter.str()},
                                  params.json_output_path + "_counter.json");
    }

    return 0;
}

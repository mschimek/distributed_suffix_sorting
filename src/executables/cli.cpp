
#include <algorithm>
#include <cstdint>
#include <string>

#include <tlx/cmdline_parser.hpp>

#include "kamping/communicator.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/io.hpp"
#include "options.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/pdcx.hpp"
#include "sa_check.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "util/printing.hpp"
#include "util/random.hpp"
#include "util/uint_types.hpp"

#define V(x) std::string(#x "=") << (x) << " " //"x=...

using namespace dsss;

using char_type = uint8_t;
using index_type = uint40;

size_t string_size = {0};
size_t alphabet_size = {2};
size_t seed = {0};
std::string input_path = "";
std::string output_path = "";
std::string dcx_variant = "dc3";
bool check = false;

dcx::PDCXConfig pdcx_config;
std::string atomic_sorter = "sample_sort";
std::string string_sorter = "radix_sort_ci3";
std::string buckets_samples;
std::string buckets_merging;

dsss::mpi::SampleSortConfig sample_sort_config;
std::string splitter_sampling = "uniform";
std::string splitter_sorting = "central";

tlx::CmdlineParser cp;
std::vector<char_type> local_string;
std::vector<index_type> local_sa;

uint64_t input_alphabet_size = 0;

void configure_cli() {
    // basic information
    cp.set_description("Distributed Suffix Array Construction using pDCX");
    cp.set_author("Manuel Haag <uozeb@student.kit.edu>");

    // input and output
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
    cp.add_string('o',
                  "output",
                  "<F>",
                  output_path,
                  "Filename for the output (SA). Note that the output is five times larger than "
                  "the input file.");

    // pdcx configuration
    cp.add_flag('c', "check", check, "Check if the SA has been constructed correctly.");
    cp.add_string('x',
                  "dcx",
                  "<F>",
                  dcx_variant,
                  "pDCX variant to use. Available options: dc3, dc7, dc13.");
    cp.add_double('t',
                  "discarding_threshold",
                  pdcx_config.discarding_threshold,
                  "Value between [0, 1], threshold when to use discarding optimization.");
    cp.add_flag('C',
                "use_lcps_tie_breaking",
                pdcx_config.use_lcps_tie_breaking,
                "Compute LCPs in string sorting and use them speedup comparison of strings in tie "
                "breaking.");
    cp.add_bytes('m',
                 "num_samples_splitters",
                 pdcx_config.num_samples_splitters,
                 "Total number of random samples to use to determine block splitters in space "
                 "efficient sort.");
    cp.add_flag('U',
                "use_random_sampling_splitters",
                pdcx_config.use_random_sampling_splitters,
                "Use random sampling to determine block splitters in space efficient sort.");
    cp.add_flag('B',
                "balance_blocks_space_efficient_sort",
                pdcx_config.balance_blocks_space_efficient_sort,
                "Balance blocks after materialization in space efficient sorting.");
    cp.add_string('P',
                  "buckets_samples",
                  "<F>",
                  buckets_samples,
                  "Number of buckets to use for space efficient sorting of samples on each "
                  "recursion level. Missing values default to 1. Example: 16,8,4");
    cp.add_string('M',
                  "buckets_merging",
                  "<F>",
                  buckets_merging,
                  "Number of buckets to use for space efficient sorting in merging phase on each "
                  "recursion level. Missing values default to 1. Example: 16,8,4");

    // sorter configuration
    cp.add_string('r',
                  "atomic_sorter",
                  "<F>",
                  atomic_sorter,
                  "Atomic sorter to be used. [sample_sort, rquick, ams, bitonic, rfis]");
    cp.add_bytes('l', "ams_levels", pdcx_config.ams_levels, "Number of levels to be used in ams.");
    cp.add_string('p',
                  "splitter_sampling",
                  "<F>",
                  splitter_sampling,
                  "Splitter sampling method in sample sort. [uniform, random]");
    cp.add_string('T',
                  "splitter_sorting",
                  "<F>",
                  splitter_sorting,
                  "Splitter sorting method in sample sort [central, distributed]");
    cp.add_string('n',
                  "string_sorter",
                  string_sorter,
                  "String sorter to be used. [multi_key_qsort, radix_sort_ci2, radix_sort_ci3]");
    cp.add_flag('S',
                "use_string_sort",
                pdcx_config.use_string_sort,
                "Use string sorting instead of atomic sorting.");
    cp.add_flag('L',
                "use_loser_tree",
                sample_sort_config.use_loser_tree,
                "Use loser tree in merging step of sample sort.");
    cp.add_flag('R',
                "use_rquick_for_splitters",
                sample_sort_config.use_rquick_for_splitters,
                "Use Rquick to sort splitter.");
    cp.add_flag('b',
                "use_binary_search_for_splitters",
                sample_sort_config.use_binary_search_for_splitters,
                "Use binary search instead of linear scan to find intervals in sample sort.");
}

template <typename EnumType>
EnumType get_enum(std::string s, std::vector<std::string> names, kamping::Communicator<>& comm) {
    for (uint i = 0; i < names.size(); i++) {
        if (s == names[i]) {
            return static_cast<EnumType>(i);
        }
    }
    if (comm.rank() == 0) {
        std::cout << "Invalid enum: " << s << std::endl;
        std::cout << "Available options: ";
        bool is_first = true;
        for (std::string& s: names) {
            if (!is_first) {
                std::cout << ", ";
            }
            std::cout << s;
            is_first = false;
        }
        std::cout << std::endl;
    }
    exit(1);
}

void map_strings_to_enum(kamping::Communicator<>& comm) {
    // pdcx
    pdcx_config.atomic_sorter =
        get_enum<mpi::AtomicSorters>(atomic_sorter, mpi::atomic_sorter_names, comm);
    pdcx_config.string_sorter =
        get_enum<dsss::SeqStringSorter>(string_sorter, dsss::string_sorter_names, comm);

    // sample sort
    sample_sort_config.splitter_sorting =
        get_enum<dsss::mpi::SplitterSorting>(splitter_sorting,
                                             dsss::mpi::splitter_sorting_names,
                                             comm);
    sample_sort_config.splitter_sampling =
        get_enum<dsss::mpi::SplitterSampling>(splitter_sampling,
                                              dsss::mpi::splitter_sampling_names,
                                              comm);

    pdcx_config.sample_sort_config = sample_sort_config;
}

std::vector<uint32_t> parse_list_of_ints(std::string s) {
    char separator = ',';
    std::replace(s.begin(), s.end(), separator, ' ');

    std::vector<uint32_t> numbers;
    std::stringstream ss(s);
    uint32_t temp;
    while (ss >> temp) {
        numbers.push_back(temp);
    }
    return numbers;
}

void check_limit(std::vector<uint32_t>& vec,
                 uint32_t limit,
                 std::string name,
                 kamping::Communicator<>& comm) {
    if (vec.size() == 0)
        return;
    uint32_t _max = *std::max_element(vec.begin(), vec.end());
    if (_max > limit) {
        kamping::report_on_root(name + " must be <= " + std::to_string(limit) + ".", comm);
        exit(1);
    }
}

void parse_enums_and_lists(kamping::Communicator<>& comm) {
    map_strings_to_enum(comm);
    pdcx_config.buckets_samples = parse_list_of_ints(buckets_samples);
    pdcx_config.buckets_merging = parse_list_of_ints(buckets_merging);
    check_limit(pdcx_config.buckets_samples, 255, "buckets_samples", comm);
    check_limit(pdcx_config.buckets_merging, 255, "buckets_merging", comm);
}

void report_arguments(kamping::Communicator<>& comm) {
    comm.barrier();
    if (comm.rank() == 0) {
        std::cout << "Arguments:\n";
        std::cout << V(string_size) << "\n";
        std::cout << V(alphabet_size) << "\n";
        std::cout << V(seed) << "\n";
        std::cout << V(input_path) << "\n";
        std::cout << V(output_path) << "\n";
        std::cout << V(dcx_variant) << "\n";
        std::cout << V(check) << "\n";
        std::cout << std::endl;
        pdcx_config.print_config();
    }
    comm.barrier();
}

void read_input(kamping::Communicator<>& comm) {
    if (input_path != "random" && !mpi::file_exists(input_path)) {
        if (comm.rank() == 0) {
            std::cerr << "File " << input_path << " does not exist!" << std::endl;
        }
        exit(1);
    }
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("io");
    if (!input_path.compare("random")) {
        string_size /= comm.size();
        uint64_t local_seed = seed + comm.rank();
        local_string =
            random::generate_random_data<char_type>(string_size, alphabet_size, local_seed);
    } else {
        if (string_size > 0) {
            local_string = mpi::read_and_distribute_string(input_path, comm, string_size);
        } else {
            local_string = mpi::read_and_distribute_string(input_path, comm);
        }
    }
    timer.stop();
    timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
    timer.clear();
    kamping::report_on_root("\n", comm);
}

void compress_alphabet(std::vector<char_type>& input, kamping::Communicator<>& comm) {
    uint64_t max_alphabet_size = 1 << (sizeof(char_type) * 8);

    // determine character frequencies
    std::vector<uint64_t> local_counts(max_alphabet_size, 0);
    for (auto c: input) {
        local_counts[c]++;
    }
    std::vector<uint64_t> global_counts =
        comm.allreduce(kamping::send_buf(local_counts), kamping::op(kamping::ops::plus<>{}));
    uint64_t alphabet_size =
        max_alphabet_size - std::count(global_counts.begin(), global_counts.end(), 0);

    if (alphabet_size == max_alphabet_size) {
        kamping::report_on_root("Can only process alphabets with not more than 255 distinct "
                                "characters. Change char_type.",
                                comm);
        exit(1);
    }

    // reserve character 0 for padding
    uint64_t next_char = 1;
    std::vector<uint64_t> map_char(max_alphabet_size);
    for (uint64_t i = 0; i < max_alphabet_size; i++) {
        if (global_counts[i] > 0) {
            map_char[i] = next_char++;
        }
    }

    // map input alphabet to compressed alphabet
    for (uint64_t i = 0; i < input.size(); i++) {
        input[i] = map_char[input[i]];
    }
    kamping::report_on_root("input_alphabet_size=" + std::to_string(alphabet_size), comm);
    input_alphabet_size = alphabet_size;
}

template <typename PDCX, typename char_type, typename index_type>
void run_pdcx(kamping::Communicator<>& comm) {
    auto algo = PDCX(pdcx_config, comm);
    local_sa = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
}

void compute_sa(kamping::Communicator<>& comm) {
    using namespace dcx;

    measurements::Timer<Communicator<>> algo_timer;
    algo_timer.synchronize_and_start("total_time");

    auto& timer = kamping::measurements::timer();
    timer.clear();

    timer.synchronize_and_start("compress_alphabet");
    compress_alphabet(local_string, comm);
    timer.stop();

    // run_pdcx<PDCX<char_type, index_type, DC7Param>, char_type, index_type>(comm);
    run_pdcx<PDCX<char_type, index_type, DC21Param>, char_type, index_type>(comm);
    // run_pdcx<PDCX<char_type, index_type, DC31Param>, char_type, index_type>(comm);
    // run_pdcx<PDCX<char_type, index_type, DC133Param>, char_type, index_type>(comm);
    // if (dcx_variant == "dc3") {
    //     run_pdcx<PDCX<char_type, index_type, DC3Param>, char_type, index_type>(comm);
    // } else
    // if (dcx_variant == "dc7") {
    //     run_pdcx<PDCX<char_type, index_type, DC7Param>, char_type, index_type>(comm);
    // } else if (dcx_variant == "dc13") {
    //     run_pdcx<PDCX<char_type, index_type, DC13Param>, char_type, index_type>(comm);
    // } else if (dcx_variant == "dc21") {
    //     run_pdcx<PDCX<char_type, index_type, DC21Param>, char_type, index_type>(comm);
    // } else if (dcx_variant == "dc31") {
    //     run_pdcx<PDCX<char_type, index_type, DC31Param>, char_type, index_type>(comm);
    // } else {
    //     std::cerr << "dcx variant " << dcx_variant
    //               << " not supported. Must be in [dc3, dc7, dc13, dc21, dc31]. \n";
    // }

    algo_timer.stop();
    algo_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
    kamping::report_on_root("\n", comm);
}

void write_sa(kamping::Communicator<>& comm) {
    if (!output_path.empty()) {
        kamping::report_on_root("Writing the SA to " + output_path + "\n", comm);
        mpi::write_data(local_sa, output_path, comm);
        comm.barrier();
        kamping::report_on_root("Finished writing the SA \n", comm);
    }
}

void check_sa(kamping::Communicator<>& comm) {
    using namespace kamping;
    if (check) {
        measurements::Timer<Communicator<>> check_timer;
        check_timer.synchronize_and_start("check_SA");

        // TODO maybe read again
        // read_input(comm);

        kamping::report_on_root("Checking SA ... ", comm);
        // assuming algorithm did not change local string
        bool correct = check_suffixarray(local_sa, local_string, comm);

        if (comm.rank() == 0) {
            std::string msg = correct ? "Correct SA!" : "ERROR: Not a correct SA!";
            std::cout << msg << std::endl;
            std::cout << "SA_ok=" << correct << std::endl;
        }
        check_timer.stop();
        check_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
        kamping::report_on_root("\n", comm);
    }
}

int main(int32_t argc, char const* argv[]) {
    kamping::Environment e;
    kamping::Communicator comm;

    options::report_compile_flags(comm);

    configure_cli();
    if (!cp.process(argc, argv)) {
        return -1;
    }
    parse_enums_and_lists(comm);
    report_arguments(comm);
    read_input(comm);
    compute_sa(comm);
    check_sa(comm);
    write_sa(comm);

    return 0;
}

#pragma once

#include <random>
#include <vector>

#include "sorters/sample_sort_config.hpp"

#ifdef INCLUDE_ALL_SORTERS
#include "AmsSort/AmsSort.hpp"
#include "Bitonic/Bitonic.hpp"
// #include "HSS/Hss.hpp"
#include "RFis/RFis.hpp"
#include "RQuick/RQuick.hpp"
#include "kamping/mpi_datatype.hpp"
#endif

#include "RBC.hpp"
#include "kamping/communicator.hpp"
#include "sorters/sample_sort.hpp"

namespace dsss::mpi {

using namespace kamping;

enum AtomicSorters { SampleSort, Rquick, Ams, Bitonic, RFis };
static std::vector<std::string> atomic_sorter_names = {
    "sample_sort", "rquick", "ams", "bitonic", "rfis"};

struct SortingWrapper {
    SortingWrapper(Communicator<>& _comm)
        : comm(_comm),
          mpi_comm(comm.mpi_communicator()),
          data_seed(3469931 + comm.rank()),
          gen{data_seed},
          num_levels(1),
          tag(123),
          print(false),
          sorter(AtomicSorters::SampleSort),
          sample_sort_config(SampleSortConfig()) {}

    void set_sorter(AtomicSorters new_sorter) { sorter = new_sorter; }
    void set_num_levels(int new_num_levels) { num_levels = new_num_levels; }
    void set_sample_sort_config(SampleSortConfig config) { sample_sort_config = config; }
    void finalize_setting() {
        if (sorter == AtomicSorters::Ams && num_levels == 1) {
            is_init = true;
            RBC::Create_Comm_from_MPI(mpi_comm, &rcomm, true, true, true);
        } else {
            is_init = true;
            RBC::Create_Comm_from_MPI(mpi_comm, &rcomm);
        }
    }


    void set_print(bool new_print) { print = new_print; }

    template <typename DataType, class Compare>
    inline void sort(std::vector<DataType>& local_data, Compare comp) {
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
        switch (sorter) {
            case SampleSort:
                sample_sort(local_data, comp, comm, sample_sort_config);
                break;
            case Rquick:
                RQuick::sort(my_mpi_type, local_data, tag, gen, mpi_comm, comp);
                break;
            case Bitonic:
                if(!is_init) {
                std::cout << "is init: " << is_init << std::endl;
                }
                Bitonic::Sort(local_data, my_mpi_type, tag, rcomm, comp);
                break;
            case RFis:
                RFis::Sort(my_mpi_type, local_data, rcomm, comp);
                break;
            case Ams:
                if(!is_init) {
                std::cout << "is init: " << is_init << std::endl;
                }
                Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp);
                break;
            default:
                sample_sort(local_data, comp, comm);
        }
#else
        sample_sort(local_data, comp, comm, sample_sort_config);
#endif

        // some problems with operators
        // case Hss:
        //     Hss::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp); //
        //     compile break;
    }


    Communicator<>& comm;
    MPI_Comm mpi_comm;
    RBC::Comm rcomm;

    int data_seed;
    std::mt19937_64 gen;
    int num_levels;
    int tag;
    bool print;
    bool is_init = false;

    AtomicSorters sorter;
    SampleSortConfig sample_sort_config;
};
} // namespace dsss::mpi

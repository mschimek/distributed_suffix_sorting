#pragma once

#include <random>
#include <vector>

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
          num_levels(1),
          tag(123),
          sorter(AtomicSorters::SampleSort) {
        RBC::Create_Comm_from_MPI(mpi_comm, &rcomm);
    }

    void set_sorter(AtomicSorters new_sorter) { sorter = new_sorter; }
    void set_num_levels(int new_num_levels) { num_levels = new_num_levels; }

    template <typename DataType, class Compare>
    inline void sort(std::vector<DataType>& local_data, Compare comp) {
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
        switch (sorter) {
            case SampleSort:
                sample_sort(local_data, comp, comm);
                break;
            case Rquick:
                RQuick::sort(my_mpi_type, local_data, tag, gen, mpi_comm, comp);
                break;
            case Ams:
                Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp);
                break;
            case Bitonic:
                Bitonic::Sort(local_data, my_mpi_type, tag, rcomm, comp);
                break;
            case RFis:
                RFis::Sort(my_mpi_type, local_data, rcomm, comp);
                break;
            default:
                sample_sort(local_data, comp, comm);
        }
#else
        sample_sort(local_data, comp, comm);
#endif

        // some problems with operators
        // case Hss:
        //     Hss::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp); //
        //     compile break;
    }


    Communicator<>& comm;
    MPI_Comm mpi_comm;
    RBC::Comm rcomm;

    std::mt19937_64 gen;
    int data_seed;
    int num_levels;
    int tag;

    AtomicSorters sorter;
};
} // namespace dsss::mpi
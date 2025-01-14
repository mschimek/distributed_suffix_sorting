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
          num_levels(1),
          tag(123),
          print(false),
          sorter(AtomicSorters::SampleSort),
          sample_sort_config(SampleSortConfig()) {
        RBC::Create_Comm_from_MPI(mpi_comm, &rcomm);
    }

    void set_sorter(AtomicSorters new_sorter) { sorter = new_sorter; }
    void set_num_levels(int new_num_levels) { num_levels = new_num_levels; }
    void set_sample_sort_config(SampleSortConfig config) { sample_sort_config = config; }


    void set_print(bool new_print) { print = new_print; }

    template <typename DataType, class Compare>
    inline void sort(std::vector<DataType>& local_data, Compare comp) {
#ifdef INCLUDE_ALL_SORTERS
        MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();

        // enum PartitioningStrategy {
        //     INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
        //     INPLACE_PARTITIONING
        // };

        // enum DistributionStrategy {
        // EXCHANGE_WITHOUT_RECV_SIZES,
        // EXCHANGE_WITH_RECV_SIZES,
        // EXCHANGE_WITH_RECV_SIZES_AND_PORTS
        // };


        // AMS parameters
        double imbalance = 1.10;
        bool use_dma = true;
        Ams::PartitioningStrategy part =
            Ams::PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING;
        Ams::DistributionStrategy distr =
            Ams::DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS;
        bool use_ips4o = true;
        bool use_two_tree = true;
        bool ams_print = print;
        // temp
        /*
        if (sample_sort_config.ams_partition_strategy == 0) {
            part = Ams::PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING;
        } else {
            part = Ams::PartitioningStrategy::INPLACE_PARTITIONING;
        }

        if (sample_sort_config.ams_distributiong_strategy == 0) {
            distr = Ams::DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES;
        } else if (sample_sort_config.ams_distributiong_strategy == 1) {
            distr = Ams::DistributionStrategy::EXCHANGE_WITH_RECV_SIZES;
        } else {
            distr = Ams::DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS;
        }
        */


        switch (sorter) {
            case SampleSort:
                sample_sort(local_data, comp, comm, sample_sort_config);
                break;
            case Rquick:
                RQuick::sort(my_mpi_type, local_data, tag, gen, mpi_comm, comp);
                break;
            case Bitonic:
                Bitonic::Sort(local_data, my_mpi_type, tag, rcomm, comp);
                break;
            case RFis:
                RFis::Sort(my_mpi_type, local_data, rcomm, comp);
                break;
            case Ams:
                std::cout << comm.rank() << " "
                          << "calling AMS, print=" << print << "\n";
                // Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp);
                Ams::sortLevel(my_mpi_type,
                               local_data,
                               num_levels,
                               gen,
                               rcomm,
                               comp,
                               imbalance,
                               use_dma,
                               part,
                               distr,
                               use_ips4o,
                               use_two_tree,
                               ams_print);

                // void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                //                std::mt19937_64& async_gen,
                //                const RBC::Comm& comm,
                //                Comp comp = Comp(),
                //                double imbalance = 1.10,
                //                bool use_dma = true,
                //                PartitioningStrategy part =
                //                  PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                //                DistributionStrategy distr =
                //                  DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                //                bool use_ips4o = true,
                //                bool use_two_tree = true,
                //                bool print = false);

                // Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp, print); //
                // BIG BUG
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

    std::mt19937_64 gen;
    int data_seed;
    int num_levels;
    int tag;
    bool print;

    AtomicSorters sorter;
    SampleSortConfig sample_sort_config;
};
} // namespace dsss::mpi

# Distributed DCX
Implementation of distributed Difference Cover Modulo X (DCX) suffix array construction algorithm.

## Compiling

To compile the code use the following instructions:
```
git clone --recursive git@github.com:HaagManuel/distributed_suffix_sorting.git
cmake -B release -DCMAKE_BUILD_TYPE=Release -DINCLUDE_ALL_SORTERS=ON -DOPTIMIZE_DATA_TYPES=OFF -DIPS4O_DISABLE_PARALLEL=ON
cd release && make -j 16
```

## Usage

To execute our currently best configuration run:

```sh
mpirun -n $NTASK release/cli $INPUT_FILE -c -x dc21 -s $TOTAL_INPUT_SIZE_BYTES -r ams -t 0.7 -l 2 -P 16,16 -M 64,64 -b -e 0 -Z -z 10000 -g -G -E -u 
```


## Notes
We tested our implementation with gcc 12.2.0 and IntelMPI 2023.1.0.

To reduce the memory footprint of MPI we use the following environment variables.

```sh
export I_MPI_SHM_CELL_FWD_NUM=0
export I_MPI_SHM_CELL_EXT_NUM_TOTAL=0
export I_MPI_SHM_CELL_BWD_SIZE=65536
export I_MPI_SHM_CELL_BWD_NUM=64
export I_MPI_MALLOC=0
export I_MPI_SHM_HEAP_VSIZE=0
```



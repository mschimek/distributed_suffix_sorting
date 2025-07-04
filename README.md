
# Distributed DCX
Implementation of our lightweight distributed Difference Cover Modulo X (DCX) suffix array construction algorithm.

## Compiling

To compile the code use the following instructions:
```
git clone --recursive git@github.com:HaagManuel/distributed_suffix_sorting.git
cmake -B release -DCMAKE_BUILD_TYPE=Release -DINCLUDE_ALL_SORTERS=ON -DOPTIMIZE_DATA_TYPES=OFF -DIPS4O_DISABLE_PARALLEL=ON
cmake --build build --parallel --target cli
```

## Usage

To execute our currently best configuration run:

```sh
mpiexec -n <number ranks> ./build/cli --input <file> --textsize <size in bytes> --dcx <dc39> --atomic-sorter ams --discarding-threshold 0.7 --ams-levels 2 --splitter-sampling random --splitter-sorting central --use-random-sampling-splitters --num-samples-splitters 20000 --buckets-sample-phase 16,16 --buckets-merging-phase 64,64,16 --use-binary-search-for-splitters --use-randomized-chunks --avg-chunks-pe 10000 --use-char-packing-samples --use-char-packing-merging --buckets-phase3 1 --samples-buckets-phase3 10000 --rearrange-buckets-balanced --use-compressed-buckets --pack-extra-words 0 --json-output-path <output-path-for-logs>
```
On small scale using fewer buckets might be beneficial. See `--help` for more information.

## Notes
We tested our implementation with gcc 12.2.0 and IntelMPI 2021.11.

To reduce the memory footprint of MPI, we use the following environment variables.

```sh
export I_MPI_SHM_CELL_FWD_NUM=0
export I_MPI_SHM_CELL_EXT_NUM_TOTAL=0
export I_MPI_SHM_CELL_BWD_SIZE=65536
export I_MPI_SHM_CELL_BWD_NUM=64
export I_MPI_MALLOC=0
export I_MPI_SHM_HEAP_VSIZE=0
```



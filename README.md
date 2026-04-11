
# Distributed DCX
Implementation of our lightweight distributed Difference Cover Modulo X (DCX) suffix array construction algorithm as presented in the paper:

> Manuel Haag, Florian Kurpicz, Peter Sanders, and Matthias Schimek. "Fast and Lightweight Distributed Suffix Array Construction." In Proceedings of the 33rd Annual European Symposium on Algorithms (ESA 2025). LIPIcs, Vol. 332, pp. 47:1–47:19. DOI: [10.4230/LIPIcs.ESA.2025.47](https://doi.org/10.4230/LIPIcs.ESA.2025.47)

If you use this code in the context of an academic publication, we kindly ask you to cite the paper.

## Compiling

To compile the code use the following instructions:
```sh
git clone --recursive git@github.com:mschimek/distributed_suffix_sorting.git
cd distributed_suffix_sorting
cmake --preset experiments
cmake --build --preset experiments --target cli
```

Available presets:
- `experiments` — Release build for benchmarking and experiments
- `dev` — Debug build for development

## Usage

To execute our currently best configuration run:

```sh
mpiexec -n <number ranks> ./build/experiments/cli --input <file> --textsize <size in bytes> --dcx <dc39> --atomic-sorter ams --discarding-threshold 0.7 --ams-levels 2 --splitter-sampling random --splitter-sorting central --use-random-sampling-splitters --num-samples-splitters 20000 --buckets-sample-phase 16,16 --buckets-merging-phase 64,64,16 --use-binary-search-for-splitters --use-randomized-chunks --avg-chunks-pe 10000 --use-char-packing-samples --use-char-packing-merging --buckets-phase3 1 --samples-buckets-phase3 10000 --rearrange-buckets-balanced --use-compressed-buckets --pack-extra-words 0 --json-output-path <output-path-for-logs>
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



# ALCF-4 QMCPACK Benchmark

This is the ALCF-4 benchmark for QMCPACK (https://qmcpack.org). This document describes obtaining the QMCPACK benchmark on heterogeneous systems for ALCF-4. The general ALCF-4 benchmark run rules apply except where explicitly modified in this document and should be reviewed before running this benchmark.

## QMCPACK Overview

QMCPACK, is a modern high-performance open-source Quantum Monte Carlo (QMC) simulation code. Its main applications are electronic structure calculations of molecular, quasi-2D and solid-state systems. Variational Monte Carlo (VMC), diffusion Monte Carlo (DMC) and a number of other advanced QMC algorithms are implemented. By directly solving the Schrodinger equation, QMC methods offer greater accuracy than methods such as density functional theory, but at a trade-off of much greater computational expense.

QMCPACK is written in C++ and designed with the modularity afforded by object-oriented programming. For parallelization QMCPACK utilizes a fully hybrid MPI + OpenMP threading approach + vendor(CUDA/HIP/SYCL) to optimize memory usage and to take advantage of the growing number of cores per SMP node or GPUs. High parallel and computational efficiencies are achievable on the largest supercomputers.

The benchmarks are particularly sensitive to floating point, memory bandwidth and memory latency performance. To obtain high
performance, the compiler’s ability of optimize and vectorize the application is critical. Strategies to place more of the walker data
higher in the memory hierarchy are likely to increase performance. 

## Code Access

The required QMCPACK version for this benchmark is commit bc5045303e6cbb74596f235e7a4e4a85999bb30a from the GitHub repository at https://github.com/QMCPACK/qmcpack.
Official documenation about prerequisites, building, testing QMCPACK can be found at https://qmcpack.readthedocs.io/en/develop/installation.html.
Build recipes for a few supercomputers are provided at the QMCPACK github repo https://github.com/QMCPACK/qmcpack/tree/develop/config.

## Benchmark

This benchmark runs a VMC simulation of fcc phase NiO in a 256-atom cell. The input file is provided as `benchmark/NiO-a256-vmc.xml`.
Please modify it before using

1. Download `NiO-fcc-supertwist111-supershift000-S64.h5` and set its path to `$orbpath`. Download instruction can be found at https://github.com/QMCPACK/qmcpack/tree/develop/tests/performance/NiO
2. Download `Ni.opt.xml` and `O.ncpp.xml` from https://github.com/QMCPACK/qmcpack/tree/develop/tests/pseudopotentials_for_tests
3. Edit `$walkers` to a desired walker count for each MPI rank.

### FOM

FOM is throughput based,
```zsh
FOM = number_of_MPI_ranks * walkers_per_rank * steps / walltime
```
For this benchmark, the total number of steps and walltime can be found as the call count of the `VMCBatched::RunSteps` timer entry in the QMCPACK output.


### QMCPACK configuration used in Aurora benchmark
```bash
cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS="-mllvm -vpo-paropt-atomic-free-reduction-slm=true" \
      -DENABLE_OFFLOAD=ON -DENABLE_SYCL=ON -DQMC_MIXED_PRECISION=ON ..
```

### Running the benchmark on Aurora
```bash
NNODES=`wc -l < $PBS_NODEFILE` # access the compute node count from PBS
NRANKS=12                      # Number of MPI ranks per node.
NDEPTH=8                       # Number of hardware threads per rank, spacing between MPI ranks on a node
export OMP_NUM_THREADS=8       # OpenMP threads. (required to be <= NDEPTH)
# CPU binding. the first 6 ranks are placed on the first socket and the last 6 ranks on the second socket.
CPU_BIND=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100

NTOTRANKS=$(( NNODES * NRANKS ))
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE # work with gpu_tile_compact.sh script to distribute one GPU tile per MPI rank.

mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind $CPU_BIND ../gpu_tile_compact.sh $exe_bin/qmcpack --enable-timers=fine NiO-a256-vmc.xml > NiO-a256-vmc.out
```
More info about how to run QMCPACK efficiently can be found at https://qmcpack.readthedocs.io/en/develop/running.html.

### Results
Walker count scanning is a needed step to maximize accelerator performance.

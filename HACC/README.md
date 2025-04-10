# ALCF-4 HACC Benchmark

This is the ALCF-4 benchmark for HACC. This document describes obtaining the HACC benchmark on heterogeneous systems for ALCF-4. The general ALCF-4 benchmark run rules apply except where explicitly modified in this document and should be reviewed before running this benchmark.

## HACC Overview

The Hardware Accelerated Cosmology Code (HACC) framework uses N-body techniques to simulate the formation of structure in collisionless fluids under the influence of gravity in an expanding universe. The main scientific goal is to simulate the evolution of the Universe from its early times to today and to advance our understanding of dark energy and dark matter, the two components that make up 95% of our Universe.

The HACC framework has been designed with great flexibility in mind – it is easily portable between different high-performance computing platforms. An overview of the code structure is given in Habib et al. Journal of Physics: Conf. Series, 180, 012019 (2009) and Pope et al. Comp. Sci. Eng. 12, 17 (2010). HACC has three distinct phases in the computation - their relative ratios to total run time strongly depend on the parameters of the simulation. The short force evaluation kernel is compute intensive with regular stride one memory accesses. This kernel can be fully vectorized and/or threaded. The tree walk phase has essentially irregular indirect memory accesses, and has very high number of branching and integer operations. The 3D FFT phase is implemented with point-to-point communication operations and is executed only every long time step; thus significantly reducing the overall communication complexity of the code.

## Code Access

The source code for the ALCF4 HACC benchmark is provided as a submodule of the alcf4_benchmarks project. The easiest way to obtain the code is to recursively clone submodules when cloning the main project:
```
git clone --recurse-submodules https://github.com/argonne-lcf/alcf4_benchmarks.git
```

The submodules can also be obtained after the initial clone of the main project via the following:
```
git clone https://github.com/argonne-lcf/alcf4_benchmarks.git
cd alcf4_benchmarks
git submodule update --init
```

A convenience script that will perform the same action is also provided:
```
git clone https://github.com/argonne-lcf/alcf4_benchmarks.git
cd alcf4_benchmarks/HACC
./git_submodule_init.sh
```

The ALCF4 HACC benchmark source code can also be obtained directly from the publicly-accessible project on the Argonne CELS gitlab: https://git.cels.anl.gov/hacc/HACC-B24

### FOM

The reported figure of merit is calculated as the total number of particles divided by the run time.
```
FOM = (np*np*np) / time
```
The FOM must be reported for `nstep=3` and `nsub=5`.

### Software prerequisites

* FFTW 

### Building HACC

```
cd HACC-B24/src
source env/bashrc.aurora.sycl
cd cpu
make
```

### Running the benchmark on Aurora
```
export RANKS_PER_NODE=96				    # Number of MPI ranks per node
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZEX_NUMBER_OF_CCS=0:4
export OMP_NUM_THREADS=1
CPU_BIND=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=$(( NNODES * RANKS_PER_NODE ))
mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind $CPU_BIND \
gpu_tile_compact.sh ./hacc_tpm indat cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -a final -f refresh -t 48x32x32
```

### Results
The results for 512 nodes of Aurora using np=6912. 

```
ACCUMULATED STATS
step   max  1.835e+02 s  avg  1.817e+02 s  min  1.800e+02 s
calc   max  2.965e+01 s  avg  2.299e+01 s  min  1.449e+01 s
build  max  4.744e+01 s  avg  2.110e+01 s  min  1.733e+01 s
srt    max  1.320e+01 s  avg  1.065e+01 s  min  8.569e+00 s
t_f1   max  6.265e+00 s  avg  4.744e+00 s  min  3.347e+00 s
t_f2   max  1.511e+01 s  avg  1.239e+01 s  min  8.989e+00 s
t_b1   max  2.112e+01 s  avg  1.945e+01 s  min  1.782e+01 s
t_b2   max  5.428e+01 s  avg  5.250e+01 s  min  5.094e+01 s
t_sub  max  7.941e+01 s  avg  5.741e+01 s  min  4.566e+01 s
t_s1   max  4.200e+01 s  avg  3.045e+01 s  min  8.344e+00 s
t_s2   max  0.000e+00 s  avg  0.000e+00 s  min  0.000e+00 s
t_a1   max  2.638e+00 s  avg  1.595e+00 s  min  7.542e-01 s
t_a2   max  6.105e+00 s  avg  3.077e+00 s  min  6.252e-01 s
```

FOM = (6912*6912*6912) / 1.835e+02 = 1799596417.046


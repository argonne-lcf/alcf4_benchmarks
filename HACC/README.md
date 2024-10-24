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

### HACC configuration used in Aurora benchmark

### FOM

### Software prerequisites

### Building HACC

### Testing The Build

### Running the benchmark

### Results

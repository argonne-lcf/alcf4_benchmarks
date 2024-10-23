# ALCF-4 HACC Benchmark

## HACC Overview

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

We also provide a convenience script that will perform the same action:
```
git clone https://github.com/argonne-lcf/alcf4_benchmarks.git
cd alcf4_benchmarks/HACC
./git_submodule_init.sh
```

### HACC configuration used in Aurora benchmark

### FOM

### Software prerequisites

### Building HACC

### Testing The Build

### Running the benchmark

### Results

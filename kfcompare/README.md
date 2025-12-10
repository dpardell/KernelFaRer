# KFCompare

A framework for comparing baseline kernel implementations with KernelFaRer-transformed versions. It helps identify discrepancies or errors by checking if both implementations produce matching outputs (within a set tolerance), and also provides benchmarking capabilities.

## Overview

KFCompare compiles each kernel twice:
1. **Baseline**: Original kernel compiled without the KernelFaRer plugin
2. **KFaRer**: Same kernel compiled with the KernelFaRer LLVM plugin

Both versions are linked into a single test executable that runs them with identical inputs and compares the outputs.

## Directory Structure

```
kfcompare/
├── include/
│   ├── KFCompare.h      # Main header with Matrix class and test utilities
│   └── KFCompare.inl    # Template implementations
├── src/
│   └── KFCompare.cpp    # Non-template implementations (timing, etc.)
├── kernels/             # Kernel source files (*.cc)
│   ├── naive_gemm.cc
│   ├── skip_gemm_*.cc
│   ├── sm_gemm_*.cc
│   └── trmm_*.cc
└── tests/
    ├── CMakeLists.txt   # Build configuration
    └── test_*.cpp       # Test drivers
```

## Building

```bash
cd kfcompare/tests
cmake -B build -DLDFLAGS_BLAS=openblas
cmake --build build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `PLUGIN` | Auto-detected | Path to KernelFaRer.so/.dylib |
| `REPLACEMENT_MODE` | `cblas-interface` | KernelFaRer replacement mode |
| `LDFLAGS_BLAS` | `openblas` | BLAS library to link (openblas, mkl, accelerate) |

Example with custom options:
```bash
cmake -B build \
  -DPLUGIN=/path/to/KernelFaRer.dylib \
  -DLDFLAGS_BLAS=accelerate
```

## Running Tests

### Run all tests with CTest
```bash
cd build
ctest --output-on-failure
```

### Run individual tests
```bash
./build/test_naive_gemm
./build/test_skip_gemm_i2 --size 256
./build/test_trmm_0 --stress 100
```

## Test Executable Options

Each test executable supports these command-line options:

| Option | Description | Example |
|--------|-------------|---------|
| `--size N` | Set M=N=K=N (square matrices) | `--size 512` |
| `--size MxNxK` | Set dimensions explicitly | `--size 64x128x256` |
| `--stress ITERS` | Run stress test with random inputs | `--stress 100` |
| `--bench` | Benchmark mode (5 runs, avg middle 3) | `--bench` |

### Examples

```bash
# Quick correctness check with default sizes
./test_naive_gemm

# Test with larger matrices
./test_naive_gemm --size 1024

# Stress test with 100 random inputs
./test_naive_gemm --stress 100

# Benchmark with specific dimensions
./test_naive_gemm --size 512x512x512 --bench
```

## Adding a New Test

### 1. Create the kernel file

Add your kernel to `kernels/your_kernel.cc`. The kernel function must use `extern "C"` linkage:

```cpp
extern "C" void your_kernel_func(/* parameters */) {
    // Implementation
}
```

### 2. Create the test driver

Add `tests/test_your_kernel.cpp`:

```cpp
#include "KFCompare.h"

// Declare both versions (suffixes added by build system)
extern "C" void your_kernel_func_baseline(/* params */);
extern "C" void your_kernel_func_kfarer(/* params */);

int main(int argc, char** argv) {
    auto cfg = kfcompare::parse_args(argc, argv);
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    
    kfcompare::Matrix<float> A(M, K), B(K, N), C(M, N);
    
    return kfcompare::run<float>(cfg, "your_kernel",
        [&]{ your_kernel_func_baseline(/* args */); },
        [&]{ your_kernel_func_kfarer(/* args */); },
        [&]{ C.zero(); },  // Reset output before each run
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-5  // Output pointer, size, tolerance
    );
}
```

### 3. Rebuild

The CMake configuration auto-discovers `test_*.cpp` files and their matching `kernels/*.cc`:

```bash
cmake --build build
```

## KFCompare API

### Matrix Class
```cpp
kfcompare::Matrix<T> mat(rows, cols);
mat.zero();                           // Fill with zeros
mat.randomize(seed, min, max);        // Random floats in [min, max]
mat.randomize_ints(seed, max);        // Random integers in [0, max]
mat.copy_from(other);                 // Copy from another matrix
mat.data();                           // Get raw pointer
```

### Test Functions
```cpp
// Unified entry point - handles regular test, stress test, or benchmark
kfcompare::run<T>(cfg, name, baseline_fn, kfarer_fn, reset_fn, randomize_fn,
                  output_ptr, output_size, tolerance);

// Lower-level functions
kfcompare::run_test<T>(...);    // Single comparison
kfcompare::stress_test<T>(...); // Multiple random iterations
kfcompare::compare<T>(a, b, n); // Compare arrays, return max diff
```


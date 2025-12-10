// KFCompare - Simple kernel comparison library for KernelFaRer testing
#ifndef KFCOMPARE_H
#define KFCOMPARE_H

#include <cstddef>
#include <functional>

namespace kfcompare {

// Test configuration parsed from command line
struct TestConfig {
    int M = 64;
    int N = 128;
    int K = 256;
    int stress_iters = 0;
    bool bench = false;  // Benchmark mode: 5 runs, average middle 3
};

// Parse command line arguments for --size and --stress
// Usage: --size N (sets M=K=N) or --size MxKxN
//        --stress ITERS
TestConfig parse_args(int argc, char** argv);

// Simple matrix class for test data
template<typename T>
class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();
    
    // No copy, allow move
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    
    void zero();
    void randomize(unsigned seed, T min_val = T(-1), T max_val = T(1));
    void randomize_ints(unsigned seed, int max_val = 3);
    void copy_from(const Matrix& other);
    
    // Debug: print matrix (first max_rows x max_cols elements)
    void print(const char* name, int max_rows = 8, int max_cols = 8) const;

private:
    T* data_ = nullptr;
    size_t rows_, cols_;
};

// Compare two arrays, return max absolute difference
template<typename T>
double compare(const T* a, const T* b, size_t n);

// Time a kernel function, returns median of 'runs' executions
double time_kernel(std::function<void()> func, int runs = 5);

// Benchmark timing result
struct BenchResult {
    double times[5];      // All 5 run times (sorted)
    double avg_mid3;      // Average of middle 3
};

// Benchmark a kernel: 5 runs, returns all times and average of middle 3
BenchResult benchmark_kernel(std::function<void()> func);

// Run a comparison test and print results. Returns 0 on pass, 1 on fail.
// - baseline: function to run as reference
// - kfarer: function to compare against baseline (potentially transformed)
// - reset: called before each run to reset output state
// - output: pointer to output buffer to compare
// - output_size: number of elements in output
// - tolerance: max allowed difference
// - bench: if true, use benchmark mode (5 runs, average middle 3)
template<typename T>
int run_test(const char* name,
             std::function<void()> baseline,
             std::function<void()> kfarer,
             std::function<void()> reset,
             const T* output, size_t output_size,
             double tolerance = 1e-6,
             bool bench = false);

// Stress test: run comparison many times with different random inputs.
template<typename T>
int stress_test(const char* name,
                std::function<void()> baseline,
                std::function<void()> kfarer,
                std::function<void()> reset,
                std::function<void(unsigned)> randomize,
                const T* output, size_t output_size,
                double tolerance = 1e-6,
                int iterations = 100);

// Unified entry point: runs test or stress test based on config.
// Call this from main() with all the lambdas.
template<typename T>
int run(const TestConfig& cfg,
        const char* name,
        std::function<void()> baseline,
        std::function<void()> kfarer,
        std::function<void()> reset,
        std::function<void(unsigned)> randomize,
        const T* output, size_t output_size,
        double tolerance = 1e-6);

} // namespace kfcompare

// Include template implementations
#include "KFCompare.inl"

#endif // KFCOMPARE_H


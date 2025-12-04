// KFCompare - Simple kernel comparison library for KernelFaRer testing
#ifndef KFCOMPARE_H
#define KFCOMPARE_H

#include <cstddef>
#include <functional>

namespace kfcompare {

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

// Time a kernel function, return median time in milliseconds
double time_kernel(std::function<void()> func, int runs = 5);

// Run a comparison test and print results. Returns 0 on pass, 1 on fail.
// - baseline: function to run as reference
// - kfarer: function to compare against baseline (potentially transformed)
// - reset: called before each run to reset output state
// - output: pointer to output buffer to compare
// - output_size: number of elements in output
// - tolerance: max allowed difference
template<typename T>
int run_test(const char* name,
             std::function<void()> baseline,
             std::function<void()> kfarer,
             std::function<void()> reset,
             const T* output, size_t output_size,
             double tolerance = 1e-6);

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

// Unified entry point: parses argc/argv for --stress N, runs appropriate mode.
// Call this from main() with all the lambdas.
template<typename T>
int run(int argc, char** argv,
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


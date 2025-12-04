// KFCompare template implementations
#ifndef KFCOMPARE_INL
#define KFCOMPARE_INL

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <functional>

namespace kfcompare {

inline TestConfig parse_args(int argc, char** argv) {
    TestConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            const char* size_str = argv[++i];
            // Try MxNxK format first
            int m, n, k;
            if (std::sscanf(size_str, "%dx%dx%d", &m, &n, &k) == 3) {
                cfg.M = m; cfg.N = n; cfg.K = k;
            } else if (std::sscanf(size_str, "%d", &m) == 1) {
                // Single number: square matrices
                cfg.M = cfg.N = cfg.K = m;
            }
        } else if (std::strcmp(argv[i], "--stress") == 0 && i + 1 < argc) {
            cfg.stress_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            cfg.bench = true;
        }
    }
    return cfg;
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    data_ = static_cast<T*>(std::malloc(sizeof(T) * rows * cols));
}

template<typename T>
Matrix<T>::~Matrix() {
    if (data_) std::free(data_);
}

template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept 
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
    other.data_ = nullptr;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        if (data_) std::free(data_);
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.data_ = nullptr;
    }
    return *this;
}

template<typename T>
void Matrix<T>::zero() {
    std::memset(data_, 0, sizeof(T) * rows_ * cols_);
}

template<typename T>
void Matrix<T>::randomize(unsigned seed, T min_val, T max_val) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        data_[i] = dist(gen);
    }
}

template<typename T>
void Matrix<T>::randomize_ints(unsigned seed, int max_val) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, max_val);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        data_[i] = static_cast<T>(dist(gen));
    }
}

template<typename T>
void Matrix<T>::copy_from(const Matrix& other) {
    std::memcpy(data_, other.data_, sizeof(T) * rows_ * cols_);
}

template<typename T>
void Matrix<T>::print(const char* name, int max_rows, int max_cols) const {
    std::printf("%s [%zu x %zu]:\n", name, rows_, cols_);
    int r_end = std::min(static_cast<int>(rows_), max_rows);
    int c_end = std::min(static_cast<int>(cols_), max_cols);
    for (int r = 0; r < r_end; ++r) {
        std::printf("  ");
        for (int c = 0; c < c_end; ++c) {
            // Assume row-major for printing
            std::printf("%8.3f ", static_cast<double>(data_[r * cols_ + c]));
        }
        if (c_end < static_cast<int>(cols_)) std::printf("...");
        std::printf("\n");
    }
    if (r_end < static_cast<int>(rows_)) std::printf("  ...\n");
}

template<typename T>
double compare(const T* a, const T* b, size_t n) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

template<typename T>
int run_test(const char* name,
             std::function<void()> baseline,
             std::function<void()> kfarer,
             std::function<void()> reset,
             const T* output, size_t output_size,
             double tolerance,
             bool bench) {
    // Run baseline and save result
    reset();
    baseline();
    std::vector<T> baseline_result(output, output + output_size);
    
    // Run kfarer and compare
    reset();
    kfarer();
    double max_diff = compare(baseline_result.data(), output, output_size);
    
    const char* status = (max_diff <= tolerance) ? "PASS" : "FAIL";
    std::printf("=== %s ===\n", name);
    std::printf("[%s] max_diff=%.2e (tol=%.2e)\n", status, max_diff, tolerance);
    
    if (bench) {
        // Benchmark mode: 5 runs, show all times, average middle 3
        reset();
        BenchResult base = benchmark_kernel(baseline);
        reset();
        BenchResult kf = benchmark_kernel(kfarer);
        
        double speedup = base.avg_mid3 / kf.avg_mid3;
        
        // Compute min/max speedup from individual middle-3 runs for confidence interval
        double min_speedup = base.times[1] / kf.times[3];  // slowest base / fastest kf
        double max_speedup = base.times[3] / kf.times[1];  // fastest base / slowest kf
        
        std::printf("\n  BENCHMARK (5 runs, avg of middle 3):\n");
        std::printf("  %-10s", "baseline:");
        for (int i = 0; i < 5; ++i) {
            if (i == 0 || i == 4) std::printf(" [%6.2f]", base.times[i]);  // dropped
            else std::printf("  %6.2f ", base.times[i]);
        }
        std::printf(" => %7.3f ms\n", base.avg_mid3);
        
        std::printf("  %-10s", "kfarer:");
        for (int i = 0; i < 5; ++i) {
            if (i == 0 || i == 4) std::printf(" [%6.2f]", kf.times[i]);  // dropped
            else std::printf("  %6.2f ", kf.times[i]);
        }
        std::printf(" => %7.3f ms\n", kf.avg_mid3);
        
        std::printf("\n  SPEEDUP: %.3f ms / %.3f ms = %.2fx", base.avg_mid3, kf.avg_mid3, speedup);
        std::printf("  (range: %.2fx - %.2fx)\n", min_speedup, max_speedup);
    } else {
        // Quick mode: median timing
        reset();
        double time_base = time_kernel(baseline);
        reset();
        double time_kf = time_kernel(kfarer);
        
        std::printf("  baseline: %.3f ms\n", time_base);
        std::printf("  kfarer:   %.3f ms (%.2fx)\n", time_kf, time_base / time_kf);
    }
    
    return (max_diff <= tolerance) ? 0 : 1;
}

template<typename T>
int stress_test(const char* name,
                std::function<void()> baseline,
                std::function<void()> kfarer,
                std::function<void()> reset,
                std::function<void(unsigned)> randomize,
                const T* output, size_t output_size,
                double tolerance,
                int iterations) {
    std::printf("=== %s (stress test, %d iterations) ===\n", name, iterations);
    
    double worst_diff = 0.0;
    int failed_iter = -1;
    
    for (int i = 0; i < iterations; ++i) {
        unsigned seed = static_cast<unsigned>(i * 12345 + 42);
        randomize(seed);
        
        // Run baseline
        reset();
        baseline();
        std::vector<T> baseline_result(output, output + output_size);
        
        // Run kfarer
        reset();
        kfarer();
        double diff = compare(baseline_result.data(), output, output_size);
        
        if (diff > worst_diff) worst_diff = diff;
        if (diff > tolerance && failed_iter < 0) failed_iter = i;
        
        // Progress indicator
        if ((i + 1) % 10 == 0 || i == iterations - 1) {
            std::printf("\r  [%d/%d] worst_diff=%.2e", i + 1, iterations, worst_diff);
            std::fflush(stdout);
        }
    }
    std::printf("\n");
    
    const char* status = (worst_diff <= tolerance) ? "PASS" : "FAIL";
    std::printf("[%s] worst_diff=%.2e (tol=%.2e)", status, worst_diff, tolerance);
    if (failed_iter >= 0) {
        std::printf(" (first fail at iter %d)", failed_iter);
    }
    std::printf("\n");
    
    return (worst_diff <= tolerance) ? 0 : 1;
}

template<typename T>
int run(const TestConfig& cfg,
        const char* name,
        std::function<void()> baseline,
        std::function<void()> kfarer,
        std::function<void()> reset,
        std::function<void(unsigned)> randomize,
        const T* output, size_t output_size,
        double tolerance) {
    // Build name with size info
    char full_name[256];
    std::snprintf(full_name, sizeof(full_name), "%s [%dx%dx%d]", name, cfg.M, cfg.N, cfg.K);
    
    if (cfg.stress_iters > 0) {
        return stress_test<T>(full_name, baseline, kfarer, reset, randomize,
                              output, output_size, tolerance, cfg.stress_iters);
    } else {
        // Initialize with default seed
        randomize(42);
        return run_test<T>(full_name, baseline, kfarer, reset,
                           output, output_size, tolerance, cfg.bench);
    }
}

} // namespace kfcompare

#endif // KFCOMPARE_INL


// KFCompare non-template implementations
#include "KFCompare.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace kfcompare {

TestConfig parse_args(int argc, char** argv) {
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

double time_kernel(std::function<void()> func, int runs) {
    std::vector<double> times;
    times.reserve(runs);
    
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        func();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }
    
    std::sort(times.begin(), times.end());
    return times[runs / 2]; // median
}

BenchResult benchmark_kernel(std::function<void()> func) {
    BenchResult result;
    std::vector<double> times;
    times.reserve(5);
    
    for (int i = 0; i < 5; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        func();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }
    
    std::sort(times.begin(), times.end());
    for (int i = 0; i < 5; ++i) result.times[i] = times[i];
    result.avg_mid3 = (times[1] + times[2] + times[3]) / 3.0;
    return result;
}

// Explicit template instantiations for common types
template class Matrix<float>;
template class Matrix<double>;
template double compare<float>(const float*, const float*, size_t);
template double compare<double>(const double*, const double*, size_t);
template int run_test<float>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, const float*, size_t, double, bool);
template int run_test<double>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, const double*, size_t, double, bool);
template int stress_test<float>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const float*, size_t, double, int);
template int stress_test<double>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const double*, size_t, double, int);
template int run<float>(const TestConfig&, const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const float*, size_t, double);
template int run<double>(const TestConfig&, const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const double*, size_t, double);

} // namespace kfcompare


// KFCompare non-template implementations
#include "KFCompare.h"
#include <chrono>
#include <vector>
#include <algorithm>

namespace kfcompare {

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

// Explicit template instantiations for common types
template class Matrix<float>;
template class Matrix<double>;
template double compare<float>(const float*, const float*, size_t);
template double compare<double>(const double*, const double*, size_t);
template int run_test<float>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, const float*, size_t, double);
template int run_test<double>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, const double*, size_t, double);
template int stress_test<float>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const float*, size_t, double, int);
template int stress_test<double>(const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const double*, size_t, double, int);
template int run<float>(int, char**, const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const float*, size_t, double);
template int run<double>(int, char**, const char*, std::function<void()>, std::function<void()>,
    std::function<void()>, std::function<void(unsigned)>, const double*, size_t, double);

} // namespace kfcompare


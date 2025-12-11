// test_trmm_1.cpp - Compare baseline vs KernelFaRer for trmm_1.cc
// Usage: ./test_trmm_1 [--size N | --size MxN] [--stress ITERS]
// Note: For TRMM, K is ignored; uses M for the triangular matrix dimension
#include "KFCompare.h"

extern "C" void basicStrmmUpper_baseline(int m, int n, float alpha, const float *A, int lda, float *B, int ldb);
extern "C" void basicStrmmUpper_kfarer(int m, int n, float alpha, const float *A, int lda, float *B, int ldb);

int main(int argc, char** argv) {
    auto cfg = kfcompare::parse_args(argc, argv);
    const int M = cfg.M, N = cfg.N;
    const float alpha = 1.0f;
    
    kfcompare::Matrix<float> A(M, M), B(M, N), B_init(M, N);
    
    return kfcompare::run<float>(cfg, "trmm_1 (basicStrmmUpper)",
        [&]{ basicStrmmUpper_baseline(M, N, alpha, A.data(), M, B.data(), M); },
        [&]{ basicStrmmUpper_kfarer(M, N, alpha, A.data(), M, B.data(), M); },
        [&]{ B.copy_from(B_init); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B_init.randomize_ints(seed+1, 3); B.copy_from(B_init); },
        B.data(), B.size(), 1e-5
    );
}
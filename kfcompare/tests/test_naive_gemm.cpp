// test_naive_gemm.cpp - Compare baseline vs KernelFaRer for naive_gemm.cc
// Usage: ./test_naive_gemm [--size N | --size MxNxK] [--stress ITERS]
#include "KFCompare.h"

extern "C" void basicDgemm_baseline(int m, int n, int k, double alpha,
    const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
extern "C" void basicDgemm_kfarer(int m, int n, int k, double alpha,
    const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

int main(int argc, char** argv) {
    auto cfg = kfcompare::parse_args(argc, argv);
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    const double alpha = 1.0, beta = 0.0;
    
    kfcompare::Matrix<double> A(M, K), B(K, N), C(M, N);
    
    return kfcompare::run<double>(cfg, "naive_gemm (basicDgemm)",
        [&]{ basicDgemm_baseline(M, N, K, alpha, A.data(), M, B.data(), N, beta, C.data(), M); },
        [&]{ basicDgemm_kfarer(M, N, K, alpha, A.data(), M, B.data(), K, beta, C.data(), M); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-5
    );
}

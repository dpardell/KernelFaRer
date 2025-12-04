// test_skip_gemm_i2.cpp - Compare baseline vs KernelFaRer for skip_gemm_i2.cc
// Stride 2 on i dimension
// Usage: ./test_skip_gemm_i2 [--size N | --size MxNxK] [--stress ITERS]
#include "KFCompare.h"

extern "C" void striped_gemm_baseline(float *A, float *B, float *C,
    int M, int N, int K, int lda, int ldb, int ldc);
extern "C" void striped_gemm_kfarer(float *A, float *B, float *C,
    int M, int N, int K, int lda, int ldb, int ldc);

int main(int argc, char** argv) {
    auto cfg = kfcompare::parse_args(argc, argv);
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    const int lda = K, ldb = N, ldc = N;
    
    kfcompare::Matrix<float> A(M, K), B(K, N), C(M, N);
    
    return kfcompare::run<float>(cfg, "skip_gemm_i2 (striped_gemm)",
        [&]{ striped_gemm_baseline(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ striped_gemm_kfarer(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-5
    );
}

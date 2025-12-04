// test_gemm_0.cpp - Compare baseline vs KernelFaRer for gemm-0.cc
// Usage: ./test_gemm_0 [--stress N]
#include "KFCompare.h"

void basicSgemm_baseline(int m, int n, int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
void basicSgemm_kfarer(int m, int n, int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

int main(int argc, char** argv) {
    const int M = 256, N = 256, K = 256;
    const float alpha = 1.0f, beta = 0.0f;
    
    kfcompare::Matrix<float> A(M, K), B(K, N), C(M, N);
    
    return kfcompare::run<float>(argc, argv, "gemm-0 (basicSgemm)",
        [&]{ basicSgemm_baseline(M, N, K, alpha, A.data(), M, B.data(), K, beta, C.data(), M); },
        [&]{ basicSgemm_kfarer(M, N, K, alpha, A.data(), M, B.data(), K, beta, C.data(), M); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-5
    );
}

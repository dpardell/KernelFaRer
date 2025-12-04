// test_sm_gemm_1.cpp - Compare baseline vs KernelFaRer for sm-gemm-1.cc
#include "KFCompare.h"

extern "C" void submatrix_gemm_baseline(int m, int n, int k, float alpha,
    const float *A, int lda, int offsetA, const float *B, int ldb, int offsetB,
    float beta, float *C, int ldc, int offsetC);
extern "C" void submatrix_gemm_kfarer(int m, int n, int k, float alpha,
    const float *A, int lda, int offsetA, const float *B, int ldb, int offsetB,
    float beta, float *C, int ldc, int offsetC);

int main(int argc, char** argv) {
    const int M = 128, N = 128, K = 128;
    const int lda = K, ldb = N, ldc = N;
    const float alpha = 1.0f, beta = 0.0f;
    
    kfcompare::Matrix<float> A(M, K), B(K, N), C(M, N);
    return kfcompare::run<float>(argc, argv, "sm-gemm-1 (submatrix_gemm)",
        [&]{ submatrix_gemm_baseline(M, N, K, alpha, A.data(), lda, 0, B.data(), ldb, 0, beta, C.data(), ldc, 0); },
        [&]{ submatrix_gemm_kfarer(M, N, K, alpha, A.data(), lda, 0, B.data(), ldb, 0, beta, C.data(), ldc, 0); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-5);
}


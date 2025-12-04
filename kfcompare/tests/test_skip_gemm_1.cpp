// test_skip_gemm_1.cpp - Compare baseline vs KernelFaRer for skip-gemm-1.cc
#include "KFCompare.h"

void striped_gemm_baseline(double *A, double *B, double *C, int M, int N, int K, int lda, int ldb, int ldc);
void striped_gemm_kfarer(double *A, double *B, double *C, int M, int N, int K, int lda, int ldb, int ldc);

int main(int argc, char** argv) {
    const int M = 256, N = 256, K = 256;
    const int lda = K, ldb = N, ldc = N;
    kfcompare::Matrix<double> A(M, K), B(K, N), C(M, N);
    return kfcompare::run<double>(argc, argv, "skip-gemm-1 (striped_gemm)",
        [&]{ striped_gemm_baseline(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ striped_gemm_kfarer(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-10);
}


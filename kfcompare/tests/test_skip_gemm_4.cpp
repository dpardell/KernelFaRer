// test_skip_gemm_4.cpp - Compare baseline vs KernelFaRer for skip-gemm-4.cc
// Note: This kernel uses column-major layout internally
#include "KFCompare.h"

void striped_gemm_baseline(double *A, double *B, double *C, int m, int n, int k, int lda, int ldb, int ldc);
void striped_gemm_kfarer(double *A, double *B, double *C, int m, int n, int k, int lda, int ldb, int ldc);

int main(int argc, char** argv) {
    const int M = 256, N = 256, K = 256;
    // Column-major: lda=M, ldb=K, ldc=M (as used in the kernel)
    const int lda = M, ldb = K, ldc = M;
    kfcompare::Matrix<double> A(M, K), B(K, N), C(M, N);
    return kfcompare::run<double>(argc, argv, "skip-gemm-4 (striped_gemm)",
        [&]{ striped_gemm_baseline(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ striped_gemm_kfarer(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc); },
        [&]{ C.zero(); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B.randomize_ints(seed+1, 3); },
        C.data(), C.size(), 1e-10);
}


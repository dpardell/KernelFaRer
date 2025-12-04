// test_trmm_0.cpp - Compare baseline vs KernelFaRer for trmm-0.cc
// Usage: ./test_trmm_0 [--stress N]
#include "KFCompare.h"

void basicStrmmLower_baseline(int m, int n, float alpha, const float *A, int lda, float *B, int ldb);
void basicStrmmLower_kfarer(int m, int n, float alpha, const float *A, int lda, float *B, int ldb);

int main(int argc, char** argv) {
    const int M = 256, N = 256;
    const float alpha = 1.0f;
    
    kfcompare::Matrix<float> A(M, M), B(M, N), B_init(M, N);
    
    return kfcompare::run<float>(argc, argv, "trmm-0 (basicStrmmLower)",
        [&]{ basicStrmmLower_baseline(M, N, alpha, A.data(), M, B.data(), M); },
        [&]{ basicStrmmLower_kfarer(M, N, alpha, A.data(), M, B.data(), M); },
        [&]{ B.copy_from(B_init); },
        [&](unsigned seed){ A.randomize_ints(seed, 3); B_init.randomize_ints(seed+1, 3); B.copy_from(B_init); },
        B.data(), B.size(), 1e-5
    );
}

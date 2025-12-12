extern "C" void striped_gemm(float *A, float *B, float *C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k += 64) {
                float a_val = A[k * lda + i]; // col major
                float b_val = B[k * ldb + j]; // row major
                C[i * ldc + j] += a_val * b_val; // row major
            }
        }
    }
}

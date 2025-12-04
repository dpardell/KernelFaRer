extern "C" void striped_gemm(float *A, float *B, float *C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i+=64) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                float a_val = A[i * lda + k];
                float b_val = B[k * ldb + j];
                C[i * ldc + j] += a_val * b_val;
            }
        }
    }
}


extern "C" void striped_gemm(float *A, float *B, float *C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j+=64) {
            for (int k = 0; k < K; k++) {
                float a_val = A[k * lda + i]; // col major 
                float b_val = B[j * ldb + k]; // col major 
                C[j * ldc + i] += a_val * b_val; // col major
            }
        }
    }
}


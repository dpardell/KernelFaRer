void striped_gemm(double *A, double *B, double *C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j+=2) {
            for (int k = 0; k < K; k++) {
                double a_val = A[i * lda + k];
                double b_val = B[j * ldb + k];
                C[j * ldc + i] += a_val * b_val;
            }
        }
    }
}

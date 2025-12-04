extern "C" void submatrix_gemm(int m, int n, int k, float alpha, const float *A, int lda, int offsetA, const float *B, int ldb, int offsetB, float beta, float *C, int ldc, int offsetC) {
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[offsetA + mm * lda + i];
        float b = B[offsetB + i * ldb + nn];
        c += a * b;
      }
      C[offsetC + mm * ldc + nn] = alpha * c + beta * C[offsetC + mm * ldc + nn];
    }
  }
}


extern "C" void submatrix_gemm(int m, int n, int k, float alpha, const float *A, int lda, int offsetA, const float *B, int ldb, int offsetB, float beta, float *C, int ldc, int offsetC) {
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        float a = A[offsetA + mm * lda + kk]; // row major
        float b = B[offsetB + kk * ldb + nn]; // row major
        c += a * b;
      }
      C[offsetC + mm * ldc + nn] = alpha * c + beta * C[offsetC + mm * ldc + nn]; // row major
    }
  }
}


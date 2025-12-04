extern "C" void submatrix_gemm(int m, int n, int k, float alpha, 
                       const float *A, int lda, int row_A, int col_A,
                       const float *B, int ldb, int row_B, int col_B,
                       float beta, float *C, int ldc, int row_C, int col_C) {
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[(row_A + mm) * lda + (col_A + i)];
        float b = B[(row_B + i) * ldb + (col_B + nn)];
        c += a * b;
      }
      C[(row_C + mm) * ldc + (col_C + nn)] = alpha * c + beta * C[(row_C + mm) * ldc + (col_C + nn)];
    }
  }
}


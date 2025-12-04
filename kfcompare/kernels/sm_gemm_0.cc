extern "C" void submatrix_gemm(int m, int n, int k, float alpha, 
                    const float *A, int lda, int offsetA,
                    const float *B, int ldb, int offsetB, 
                    float beta, float *C, int ldc, int offsetC) {
  A += offsetA;
  B += offsetB;
  C += offsetC;
  
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        float a = A[mm * lda + kk]; 
        float b = B[kk * ldb + nn];    
        sum += a * b;
      }
      C[mm * ldc + nn] = alpha * sum + beta * C[mm * ldc + nn];
    }
  }
}


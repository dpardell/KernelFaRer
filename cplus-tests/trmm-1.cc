void basicStrmmUpper(int m, int n, float alpha, const float *A, int lda, float *B, int ldb)
{
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) { 
      float sum = 0.0f;
      for (int k = mm; k < m; ++k) {  
        float a = A[mm * lda + k];    // row major
        float b = B[k * ldb + nn];    // row major
        sum += a * b; 
      }
      B[mm * ldb + nn] = alpha * sum; // row major
    }
  }
}
void basicStrmmUpper(int m, int n, float alpha, const float *A, int lda, float *B, int ldb)
{
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) { 
      float sum = 0.0f;
      for (int k = mm; k < m; ++k) { 
        float a = A[mm + k * lda];
        float b = B[k + nn * ldb];
        sum += a * b; 
      }
      B[mm + nn * ldb] = alpha * sum;
    }
  }
}
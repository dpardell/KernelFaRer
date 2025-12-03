void basicStrmmLower(int m, int n, float alpha, const float *A, int lda, float *B, int ldb)
{
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = m - 1; mm >= 0; --mm) {
      float sum = 0.0f;
      for (int k = 0; k <= mm; ++k) {
        float a = A[mm + k * lda];
        float b = B[k + nn * ldb];
        sum += a * b;
      }
      B[mm + nn * ldb] = alpha * sum;
    }
  }
}
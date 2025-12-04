extern "C" void basicDgemm(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc )
{
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      double c = 0.0;
      for (int i = 0; i < k; ++i) {
        double a = A[mm + i * lda];
        double b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}


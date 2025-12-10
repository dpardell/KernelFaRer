extern "C" void striped_gemm(float *A, float *B, float *C,
    int m, int n, int k,
    int lda, int ldb, int ldc) {
    
    constexpr int i_blk = 128;
    constexpr int j_blk = 64;
    constexpr int p_blk = 128;
    
    // Naive + Interchange + Blocking GEMM
    for (int i = 0; i < m; i+=i_blk) {
        for (int p = 0; p < k; p+=p_blk) {
            for (int j = 0; j < n; j+=j_blk) {
                const int ii_end = ((i + i_blk) < m) ? (i + i_blk) : m;
                for (long ii = i; ii < ii_end; ii+=2) {
                    const int pp_end = ((p + p_blk) < k) ? (p + p_blk) : k;
                    for (long pp = p; pp < pp_end; pp++) {
                        const int jj_end = ((j + j_blk) < n) ? (j + j_blk) : n;
                        for (long jj = j; jj < jj_end; jj++) { 
                            float a = A[ii * lda + pp]; // row major
                            float b =  B[pp * ldb + jj]; // row major
                            C[ii * ldc + jj] += a * b; // row major
                        }
                    }
                }
            }
        }
    }
}


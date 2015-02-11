#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 192
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /*for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) 
    {
      //double cij = C[i+j*lda];
      int aindex = k*lda;
      int cindex = j*lda;
      int bindex = k+cindex;
      for (int i = 0; i < M; ++i)
        C[i+cindex] += A[i+aindex] * B[bindex];
	      //cij += A[i+k*lda] * B[k+j*lda];
      //C[i+j*lda] = cij;
    }*/

  int m = 0;
  if (M%2 != 0)
    m = M-1;
  else
    m = M;
  //printf("n: %d; N: %d", n, N);
  for (int j = 0; j < N; ++j) {
    int cindex = j*lda;
    for (int k = 0; k < K; ++k) {
      //double cij = C[i+j*n];
      int aindex = k*lda;
      //int cindex = j*n;
      int bindex = k+cindex;
      __m128d bcol = _mm_load1_pd(B+bindex);
      int i = 0;
      for(; i < m; i = i+2) {
         //cij += A[i+k*n] * B[k+j*n];
         //C[i+cindex] += A[i+aindex] * B[bindex];
        __m128d a = _mm_loadu_pd(A+i+aindex);
        __m128d c = _mm_loadu_pd(C+i+cindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, bcol));
        _mm_storeu_pd(C+i+cindex, c);
      }
      for (; i < M; ++i) {
        C[i+cindex] += A[i+aindex] * B[bindex];
      }
      //C[i+j*n] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int j = 0; j < lda; j += BLOCK_SIZE)
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
	       /* Correct block dimensions if block "goes off edge of" the matrix */
	       int M = min (BLOCK_SIZE, lda-i);
	       int N = min (BLOCK_SIZE, lda-j);
	       int K = min (BLOCK_SIZE, lda-k);

	       /* Perform individual block dgemm */
	       do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

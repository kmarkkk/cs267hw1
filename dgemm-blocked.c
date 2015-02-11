#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
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

  /*int m = 0;
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
  }*/

  int M2, N4, K_unroll, aindex, bindex, bindex1, bindex2, bindex3, bindex4, 
  cindex, cindex1, cindex2, cindex3, cindex4, i, j, k;
  //__m128d a, b, b1, b2, b3, b4, c, c1, c2, c3, c4;
  __m128d b, b1, b2, b3, b4;
  register __m128d a, c, c1, c2, c3, c4;

  if (M%2 != 0)
    M2 = M-1;
  else
    M2 = M;

  N4 = N - (N % 4);
  K_unroll = K - (K % 4);

  j = 0;
  for (; j < N4; j+=4) {
    cindex1 = j*lda;
    cindex2 = (j+1)*lda;
    cindex3 = (j+2)*lda;
    cindex4 = (j+3)*lda;
    //for (int i = 0; i < n; ++i) {
    i = 0;
    for (; i < M2; i+=2) {
      //double cij = C[i+j*n];
      //int aindex = k*n;
      //int cindex = j*n;
      //int bindex = k+cindex;
      //__m128d bcol = _mm_load1_pd(B+bindex);
      c1 = _mm_loadu_pd(C+i+cindex1);
      c2 = _mm_loadu_pd(C+i+cindex2);
      c3 = _mm_loadu_pd(C+i+cindex3);
      c4 = _mm_loadu_pd(C+i+cindex4);
      //int k = 0;
      //for(; k < N; k = k+2) {
      k = 0;
      for (; k < K_unroll; k+=4) {
         //cij += A[i+k*n] * B[k+j*n];
         //C[i+cindex] += A[i+aindex] * B[bindex];
        aindex = k*lda;
        bindex1 = k+cindex1;
        bindex2 = k+cindex2;
        bindex3 = k+cindex3;
        bindex4 = k+cindex4;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));

        aindex = (k+1)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));
        
        aindex = (k+2)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));

        aindex = (k+3)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));
      }

      for (; k < K; ++k) {
        aindex = k*lda;
        bindex1 = k+cindex1;
        bindex2 = k+cindex2;
        bindex3 = k+cindex3;
        bindex4 = k+cindex4;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));
      }
      //for (; i < n; ++i) {
        //C[i+cindex] += A[i+aindex] * B[bindex];
      //}
      //C[i+j*n] = cij;
      _mm_storeu_pd(C+i+cindex1, c1);
      _mm_storeu_pd(C+i+cindex2, c2);
      _mm_storeu_pd(C+i+cindex3, c3);
      _mm_storeu_pd(C+i+cindex4, c4);
    }
    for (; i < M; ++i) {
      //for (int k = 0; k < n; ++k) {
        //C[i+cindex] += A[i+k*n] * B[k+j*n];
      //}

      k = 0;
      for (; k < K_unroll; k+=4) {
        aindex = k*lda;
        bindex1 = k+cindex1;
        bindex2 = k+cindex2;
        bindex3 = k+cindex3;
        bindex4 = k+cindex4;

        C[i+cindex1] += A[i+aindex] * B[bindex1];
        C[i+cindex2] += A[i+aindex] * B[bindex2];
        C[i+cindex3] += A[i+aindex] * B[bindex3];
        C[i+cindex4] += A[i+aindex] * B[bindex4];

        aindex = (k+1)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;

        C[i+cindex1] += A[i+aindex] * B[bindex1];
        C[i+cindex2] += A[i+aindex] * B[bindex2];
        C[i+cindex3] += A[i+aindex] * B[bindex3];
        C[i+cindex4] += A[i+aindex] * B[bindex4];

        aindex = (k+2)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;

        C[i+cindex1] += A[i+aindex] * B[bindex1];
        C[i+cindex2] += A[i+aindex] * B[bindex2];
        C[i+cindex3] += A[i+aindex] * B[bindex3];
        C[i+cindex4] += A[i+aindex] * B[bindex4];

        aindex = (k+3)*lda;
        bindex1 += 1;
        bindex2 += 1;
        bindex3 += 1;
        bindex4 += 1;

        C[i+cindex1] += A[i+aindex] * B[bindex1];
        C[i+cindex2] += A[i+aindex] * B[bindex2];
        C[i+cindex3] += A[i+aindex] * B[bindex3];
        C[i+cindex4] += A[i+aindex] * B[bindex4];
      }

      for (; k < K; ++k) {
        aindex = k*lda;
        bindex1 = k+cindex1;
        bindex2 = k+cindex2;
        bindex3 = k+cindex3;
        bindex4 = k+cindex4;

        C[i+cindex1] += A[i+aindex] * B[bindex1];
        C[i+cindex2] += A[i+aindex] * B[bindex2];
        C[i+cindex3] += A[i+aindex] * B[bindex3];
        C[i+cindex4] += A[i+aindex] * B[bindex4];
      }
    }
  }

  for (; j < N; ++j) {
    cindex = j*lda;

    i = 0;
    for (; i < M2; i+=2) {
      c = _mm_loadu_pd(C+i+cindex);

      k = 0;
      for (; k < K_unroll; k+=4) {
        aindex = k*lda;
        bindex = k+cindex;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, b));

        aindex = (k+1)*lda;
        bindex += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, b));

        aindex = (k+2)*lda;
        bindex += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, b));

        aindex = (k+3)*lda;
        bindex += 1;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, b));
      }

      for (; k < K; k++) {
         //cij += A[i+k*n] * B[k+j*n];
         //C[i+cindex] += A[i+aindex] * B[bindex];
        aindex = k*lda;
        bindex = k+cindex;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        c = _mm_add_pd(c, _mm_mul_pd(a, b));
      }
      
      //C[i+j*n] = cij;
      _mm_storeu_pd(C+i+cindex, c);
    }

    for (; i < M; ++i) {
      k = 0;
      /*for (; k < N_unroll; k+=4) {
        C[i+cindex] += A[i+k*n] * B[k+j*n];
        C[i+cindex] += A[i+(k+1)*n] * B[k+1+j*n];
        C[i+cindex] += A[i+(k+2)*n] * B[k+2+j*n];
        C[i+cindex] += A[i+(k+3)*n] * B[k+3+j*n];
      }*/

      for (; k < K; ++k) {
        C[i+cindex] += A[i+k*lda] * B[k+j*lda];
      }
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
    for (int i = 0; i < lda; i += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	       /* Correct block dimensions if block "goes off edge of" the matrix */
	       int M = min (BLOCK_SIZE, lda-i);
	       int N = min (BLOCK_SIZE, lda-j);
	       int K = min (BLOCK_SIZE, lda-k);

	       /* Perform individual block dgemm */
	       do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

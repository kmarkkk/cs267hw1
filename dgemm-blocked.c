#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void
pr(double *arr, int len)
{
  for (int t = 0; t < len; t++) {
    printf("%.2f ", arr[t]);
  }
  printf("\n");
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  /* For each row i of A */
      for (int k = 0; k < K; ++k)
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

static void
do_bk(int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      double cij = C[row * M + col];
      for (int k = 0; k < K; ++k) {
        cij += A[k * M + row] * B[col * K + k];
      }
      C[row * M + col] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* restrict C)
{
  /*
  for (int i = 0; i < lda; ++i)
    for (int j = 0; j < lda; ++j) 
    {
      double cij = C[i+j*lda];
      for( int k = 0; k < lda; k++ )
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
    */


  int tripe_size = BLOCK_SIZE * lda;
  double a_l2[tripe_size];
  double b_l2[tripe_size];
  double c_l2[tripe_size];
  double a_l1[BLOCK_SIZE * BLOCK_SIZE]; 
  double b_l1[BLOCK_SIZE * BLOCK_SIZE]; 
  double c_l1[BLOCK_SIZE * BLOCK_SIZE]; 
  /* For each block-row of A */ 
  for (int k = 0; k < lda; k += BLOCK_SIZE) {
    int K = min (BLOCK_SIZE, lda-k);
    memcpy(a_l2, A + k * lda, K * lda);
    for (int t = 0; t < lda; t++) {
      memcpy(b_l2 + t * K, B + k + t * lda, K);
    }

    for (int i = 0; i < lda; i += BLOCK_SIZE) {
      int M = min (BLOCK_SIZE, lda-i);
      for (int col = 0; col < lda; col++) {
        memcpy(c_l2 + col * M, C + i + col * lda, M);
      }
      for (int col = 0; col < K; col++) {
        memcpy(a_l1 + col * M, a_l2 + col * lda + i, M);
      }
      
      /* For each block-column of B */
      for (int j = 0; j < lda; j += BLOCK_SIZE) {
        /* Accumulate block dgemms into block of C */
        int N = min (BLOCK_SIZE, lda-j);
        memcpy(c_l1, c_l2 + j * M, M * N);
        memcpy(a_l1, a_l1 + j * M, M * N);
        /* Correct block dimensions if block "goes off edge of" the matrix */
        /* Perform individual block dgemm */
        //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        do_bk(lda, M, N, K, a_l1, b_l1, c_l1);
        memcpy(c_l2 + j * M, c_l1, M * N);
        pr(c_l1, M * N);
      }
      for (int col = 0; col < lda; col++) {
        memcpy(C + i + col * lda, c_l2 + col * M, M);
      }
    }
  }
}



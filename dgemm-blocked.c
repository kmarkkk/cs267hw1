#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

int BLOCK_SIZE = 41;

#define min(a,b) (((a)<(b))?(a):(b))
#define DOUBLE_SIZE sizeof(double)
#define P_SIZE 10
#define PT 1

static void
pr(double *arr, int R, int C, int rm)
{
  if (!PT) {
    return;
  }
  for (int row = 0; row < R; row++) {
    for (int col = 0; col < C; col++) {
      if (rm) {
        printf("%.2f ", arr[row * C + col]);
      } else {
        printf("%.2f ", arr[col * R + row]);
      }
    }
    printf("\n");
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  /* For each row i of A */
      for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) 
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

static void
do_bk2(int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  for (int k = 0; k < K; ++k) {
    double *AA = A + k * M;
    for (int col = 0; col < N; ++col) {
      double *CC = C + M * col;
      double *BB = B + K * col;
      for (int row = 0; row < M; ++row) {
        CC[row] += AA[row] * BB[k];
      }
    }
  }
}



static void
do_bk(int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  /*
     pr(A, 30);
     pr(B, 30);
     pr(C, 10);
   */
  for (int col = 0; col < N; ++col) {
    double *BB = B + col * K;
    double *CC = C + col * M;
    for (int row = 0; row < M; ++row) {
      double *AA = A + row * K;
      double cij = CC[row];
      for (int k = 0; k < K; ++k) {
        cij += AA[k] * BB[k];
      }
      CC[row] = cij;
    }
  }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* restrict C, int b1,
    int b2, int b3)
{
  /*
  printf("Called\n");
  pr(A, 30);
  pr(B, 30);
  pr(C, 10);
  printf("-----\n");
  */
  /*
      for( int k = 0; k < lda; k++ )
    for (int j = 0; j < lda; ++j) 
  for (int i = 0; i < lda; ++i)
    {
      double cij = C[i+j*lda];
        cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
    */

  int BLOCK_SIZE1 = b1;
  int BLOCK_SIZE2 = b2;
  int BLOCK_SIZE3 = b3;


  double a_l2[BLOCK_SIZE1 * lda];
  double b_l2[BLOCK_SIZE1 * lda];
  double c_l2[BLOCK_SIZE2 * lda];
  double a_l1[BLOCK_SIZE1 * BLOCK_SIZE2]; 
  double b_l1[BLOCK_SIZE3 * BLOCK_SIZE1]; 
  double c_l1[BLOCK_SIZE2 * BLOCK_SIZE3]; 
  /* For each block-row of A */ 
  for (int k = 0; k < lda; k += BLOCK_SIZE1) {
    int K = min (BLOCK_SIZE1, lda-k);
    memcpy(a_l2, A + k * lda, K * lda * DOUBLE_SIZE);
    for (int t = 0; t < lda; t++) {
      memcpy(b_l2 + (t * K), B + (k + t * lda), 
          K * DOUBLE_SIZE);
    }

    for (int i = 0; i < lda; i += BLOCK_SIZE2) {
      int M = min (BLOCK_SIZE2, lda-i);
      for (int col = 0; col < lda; col++) {
        memcpy(c_l2 + col * M, 
            C + (i + col * lda), M * DOUBLE_SIZE);
      }

      for (int col = 0; col < K; col++) {
        memcpy(a_l1 + col * M, 
            a_l2 + (col * lda + i), M * DOUBLE_SIZE);
      }
      /* For each block-column of B */
      for (int j = 0; j < lda; j += BLOCK_SIZE3) {
        /* Accumulate block dgemms into block of C */
        int N = min (BLOCK_SIZE3, lda-j);
        memcpy(c_l1, c_l2 + j * M, M * N * DOUBLE_SIZE);
        memcpy(b_l1, b_l2 + j * K, K * N * DOUBLE_SIZE);
        /* Correct block dimensions if block "goes off edge of" the matrix */
        /* Perform individual block dgemm */
        //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        do_bk2(lda, M, N, K, a_l1, b_l1, c_l1);
        memcpy(c_l2 + j * M, c_l1, M * N * DOUBLE_SIZE);
      }
      for (int col = 0; col < lda; col++) {
        memcpy(C + (i + col * lda), 
            c_l2 + col * M, M * DOUBLE_SIZE);
      }
    }
  }

}


void square_dgemm2 (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
    /* For each block-column of B */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
    for (int j = 0; j < lda; j += BLOCK_SIZE)
  for (int i = 0; i < lda; i += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

#include <string.h>
#include <stdlib.h>
#include <emmintrin.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";
#define BLOCK_SIZE 64
#define PADDING_BASE 4
#define DOUBLE_SIZE sizeof(double)
#define min(a,b) (((a)<(b))?(a):(b))
#define C_ROW 8
#define C_COL 4

void mm(int m, int n, int k, double *A, double *B, double *C, int);

typedef union
{
  __m128d v;
  double d[2];
} vector;
  
static void
pr(double *arr, int R, int C, int rm)
{
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

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matric_00es stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int lda, double* Ar, double* Br, double* Cr)
{
  double *A = Ar;
  double *B = Br;
  double *C = Cr;
  int new_lda = lda;
  /* Padding if needed */
  if (lda % PADDING_BASE != 0) {
    new_lda = (lda / PADDING_BASE + 1) * PADDING_BASE;
    int size = new_lda * new_lda;
    A = (double *)malloc(3 * size * DOUBLE_SIZE);
    memset(A, 0, 3 * size * DOUBLE_SIZE);
    B = A + size;
    C = B + size;
    for (int t = 0; t < lda; t++) {
      memcpy(A + t * new_lda, Ar + t * lda, lda * DOUBLE_SIZE);
      memcpy(B + t * new_lda, Br + t * lda, lda * DOUBLE_SIZE);
      memcpy(C + t * new_lda, Cr + t * lda, lda * DOUBLE_SIZE);
    }
  }

  for (int k = 0; k < new_lda; k += BLOCK_SIZE) {
    int K = min (BLOCK_SIZE, new_lda-k);
    for (int j = 0; j < new_lda; j += BLOCK_SIZE) {
      int N = min (BLOCK_SIZE, new_lda-j);
      for (int i = 0; i < new_lda; i += BLOCK_SIZE) {
        int M = min (BLOCK_SIZE, new_lda-i);
	      mm(M, N, K, 
            A + i + k*new_lda, B + k + j*new_lda, C + i + j*new_lda, 
            new_lda);
      }
    }
  }

  if (lda % PADDING_BASE != 0) {
    int new_lda = (lda / PADDING_BASE + 1) * PADDING_BASE;
    for (int t = 0; t < lda; t++) {
      memcpy(Ar + t * lda, A + t * new_lda, lda * DOUBLE_SIZE);
      memcpy(Br + t * lda, B + t * new_lda, lda * DOUBLE_SIZE);
      memcpy(Cr + t * lda, C + t * new_lda, lda * DOUBLE_SIZE);
    }
    free(A);
  }


}

void mm(int m, int n, int k, double *A, double *B, double *C, int lda)
{
  /* Copy optimization for cache conflict */
  double cA[m * k];
  double cB[k * n];
  /* For each row i of A */
  for (int j = 0; j < n; j += 4)  {
    double *pb = B + j * lda;
    double *pcB = cB + j * k;
    for (int t = 0; t < k; t++) {
      *pcB++ = *pb;
      *pcB++ = *(pb + lda);
      *pcB++ = *(pb + 2 * lda);
      *pcB++ = *(pb + 3 * lda);
      pb++;
    }

    for (int i = 0; i < m; i += 4){
      if (j == 0) {
        double *pa = A + i;
        double *pcA = cA + i * k;
        for (int t = 0; t < k; t++) {
          *pcA++ = *pa;
          *pcA++ = *(pa + 1);
          *pcA++ = *(pa + 2);
          *pcA++ = *(pa + 3);
          pa += lda;
        }
      }

      vector c_00_c_10, c_01_c_11, c_02_c_12, c_03_c_13;
      vector c_20_c_30, c_21_c_31, c_22_c_32, c_23_c_33;
      vector a_0_a_1, a_2_a_3;
      vector b_0, b_1, b_2, b_3;

      c_00_c_10.v = _mm_setzero_pd();
      c_01_c_11.v = _mm_setzero_pd();
      c_02_c_12.v = _mm_setzero_pd();
      c_03_c_13.v = _mm_setzero_pd();
      c_20_c_30.v = _mm_setzero_pd();
      c_21_c_31.v = _mm_setzero_pd();
      c_22_c_32.v = _mm_setzero_pd();
      c_23_c_33.v = _mm_setzero_pd();
      
      double *la = cA + i * k;
      double *lb = cB + j * k;
      for (int p = 0; p < k; ++p) {
        a_0_a_1.v = _mm_load_pd(la);
        a_2_a_3.v = _mm_load_pd(la + 2);

        b_0.v = _mm_load1_pd(lb);
        b_1.v = _mm_load1_pd(lb + 1);
        b_2.v = _mm_load1_pd(lb + 2);
        b_3.v = _mm_load1_pd(lb + 3);

        c_00_c_10.v = _mm_add_pd(c_00_c_10.v, _mm_mul_pd(a_0_a_1.v, b_0.v));
        c_01_c_11.v = _mm_add_pd(c_01_c_11.v, _mm_mul_pd(a_0_a_1.v, b_1.v));
        c_02_c_12.v = _mm_add_pd(c_02_c_12.v, _mm_mul_pd(a_0_a_1.v, b_2.v));
        c_03_c_13.v = _mm_add_pd(c_03_c_13.v, _mm_mul_pd(a_0_a_1.v, b_3.v));

        c_20_c_30.v = _mm_add_pd(c_20_c_30.v, _mm_mul_pd(a_2_a_3.v, b_0.v));
        c_21_c_31.v = _mm_add_pd(c_21_c_31.v, _mm_mul_pd(a_2_a_3.v, b_1.v));
        c_22_c_32.v = _mm_add_pd(c_22_c_32.v, _mm_mul_pd(a_2_a_3.v, b_2.v));
        c_23_c_33.v = _mm_add_pd(c_23_c_33.v, _mm_mul_pd(a_2_a_3.v, b_3.v));
        la += 4;
        lb += 4;
      }
      C[i + j * lda] += c_00_c_10.d[0];
      C[i + 1 + j * lda] += c_00_c_10.d[1];
      C[i + 2 + j * lda] += c_20_c_30.d[0];
      C[i + 3 + j * lda] += c_20_c_30.d[1];

      C[i + (j + 1) * lda] += c_01_c_11.d[0];
      C[i + 1 + (j + 1) * lda] += c_01_c_11.d[1];
      C[i + 2 + (j + 1) * lda] += c_21_c_31.d[0];
      C[i + 3 + (j + 1) * lda] += c_21_c_31.d[1];

      C[i + (j + 2) * lda] += c_02_c_12.d[0];
      C[i + 1 + (j + 2) * lda] += c_02_c_12.d[1];
      C[i + 2 + (j + 2) * lda] += c_22_c_32.d[0];
      C[i + 3 + (j + 2) * lda] += c_22_c_32.d[1];

      C[i + (j + 3) * lda] += c_03_c_13.d[0];
      C[i + 1 + (j + 3) * lda] += c_03_c_13.d[1];
      C[i + 2 + (j + 3) * lda] += c_23_c_33.d[0];
      C[i + 3 + (j + 3) * lda] += c_23_c_33.d[1];
    }
  }
}
  


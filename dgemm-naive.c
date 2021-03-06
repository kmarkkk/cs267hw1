#include <string.h>
#include <stdlib.h>
#include <emmintrin.h>

const char* dgemm_desc = "Matrix multiplication for team 40";
/* k, j, i */
#define BLOCK_SIZE1 256
#define BLOCK_SIZE2 256
#define PADDING_BASE 4
#define DOUBLE_SIZE sizeof(double)
#define min(a,b) (((a)<(b))?(a):(b))

void mm(int m, int n, int k, double *A, double *B, double *C, int);

typedef union
{
  __m128d vec;
  double vals[2];
} vector;
  


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matric_00es stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int lda, double* Ar, double* Br, double* restrict Cr)
{
  double *restrict A = Ar;
  double *restrict B = Br;
  double *restrict C = Cr;
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
    }
  }

  for (int k = 0; k < new_lda; k += BLOCK_SIZE1) {
    int K = min (BLOCK_SIZE1, new_lda-k);
    for (int j = 0; j < new_lda; j += BLOCK_SIZE2) {
      int N = min (BLOCK_SIZE2, new_lda-j);
        //printf("k %d j %d K %d N %d lda %d\n", k, j, K, N, new_lda);
	      mm(new_lda, N, K, 
            A + k*new_lda, B + k + j*new_lda, C + j*new_lda, 
            new_lda);
    }
  }

  if (lda % PADDING_BASE != 0) {
    int new_lda = (lda / PADDING_BASE + 1) * PADDING_BASE;
    for (int t = 0; t < lda; t++) {
      memcpy(Cr + t * lda, C + t * new_lda, lda * DOUBLE_SIZE);
    }
    free(A);
  }
}

void mm(int m, int n, int k, double *restrict A, 
    double *restrict B, double *restrict C, int lda)
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

      c_00_c_10.vec = _mm_setzero_pd();
      c_01_c_11.vec = _mm_setzero_pd();
      c_02_c_12.vec = _mm_setzero_pd();
      c_03_c_13.vec = _mm_setzero_pd();
      c_20_c_30.vec = _mm_setzero_pd();
      c_21_c_31.vec = _mm_setzero_pd();
      c_22_c_32.vec = _mm_setzero_pd();
      c_23_c_33.vec = _mm_setzero_pd();
      
      double *la = cA + i * k;
      double *lb = cB + j * k;
      for (int p = 0; p < k; ++p) {
        a_0_a_1.vec = _mm_load_pd(la);
        a_2_a_3.vec = _mm_load_pd(la + 2);

        b_0.vec = _mm_load1_pd(lb);
        b_1.vec = _mm_load1_pd(lb + 1);
        b_2.vec = _mm_load1_pd(lb + 2);
        b_3.vec = _mm_load1_pd(lb + 3);

        // Store and multiply is slow....
        c_00_c_10.vec = _mm_add_pd(c_00_c_10.vec, 
            a_0_a_1.vec * b_0.vec);
        c_01_c_11.vec = _mm_add_pd(c_01_c_11.vec, 
            a_0_a_1.vec * b_1.vec);
        c_02_c_12.vec = _mm_add_pd(c_02_c_12.vec, 
            a_0_a_1.vec * b_2.vec);
        c_03_c_13.vec = _mm_add_pd(c_03_c_13.vec, 
            a_0_a_1.vec * b_3.vec);

        c_20_c_30.vec = _mm_add_pd(c_20_c_30.vec, 
            a_2_a_3.vec * b_0.vec);
        c_21_c_31.vec = _mm_add_pd(c_21_c_31.vec, 
            a_2_a_3.vec * b_1.vec);
        c_22_c_32.vec = _mm_add_pd(c_22_c_32.vec, 
            a_2_a_3.vec * b_2.vec);
        c_23_c_33.vec = _mm_add_pd(c_23_c_33.vec, 
            a_2_a_3.vec * b_3.vec);
        la += 4;
        lb += 4;
      }
      double *cp = C + i + j * lda;
      *cp += c_00_c_10.vals[0];
      *(cp + 1) += c_00_c_10.vals[1];
      *(cp + 2) += c_20_c_30.vals[0];
      *(cp + 3) += c_20_c_30.vals[1];

      cp += lda;
      *cp += c_01_c_11.vals[0];
      *(cp + 1) += c_01_c_11.vals[1];
      *(cp + 2) += c_21_c_31.vals[0];
      *(cp + 3) += c_21_c_31.vals[1];

      cp += lda;
      *cp += c_02_c_12.vals[0];
      *(cp + 1) += c_02_c_12.vals[1];
      *(cp + 2) += c_22_c_32.vals[0];
      *(cp + 3) += c_22_c_32.vals[1];

      cp += lda;
      *cp += c_03_c_13.vals[0];
      *(cp + 1) += c_03_c_13.vals[1];
      *(cp + 2) += c_23_c_33.vals[0];
      *(cp + 3) += c_23_c_33.vals[1];
    }
  }
}
  


#include <emmintrin.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  int N2, N4, aindex, bindex, bindex1, bindex2, bindex3, bindex4, cindex, cindex1, cindex2, cindex3, cindex4, i, j;
  //__m128d a, b, b1, b2, b3, b4, c, c1, c2, c3, c4;
  __m128d b, b1, b2, b3, b4;
  register __m128d a, c, c1, c2, c3, c4;

  if (n%2 != 0)
    N2 = n-1;
  else
    N2 = n;

  N4 = n - (n % 4);

  j = 0;
  for (; j < N4; j+=4) {
    cindex1 = j*n;
    cindex2 = (j+1)*n;
    cindex3 = (j+2)*n;
    cindex4 = (j+3)*n;
    //for (int i = 0; i < n; ++i) {
    i = 0;
    for (; i < N2; i+=2) {
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
      for (int k = 0; k < n; k++) {
	       //cij += A[i+k*n] * B[k+j*n];
	       //C[i+cindex] += A[i+aindex] * B[bindex];
        aindex = k*n;
        bindex1 = k+cindex1;
        bindex2 = k+cindex2;
        bindex3 = k+cindex3;
        bindex4 = k+cindex4;
        a = _mm_loadu_pd(A+i+aindex);
        b1 = _mm_load1_pd(B+bindex1);
        b2 = _mm_load1_pd(B+bindex2);
        b3 = _mm_load1_pd(B+bindex3);
        b4 = _mm_load1_pd(B+bindex4);
        //__m128d c = _mm_loadu_pd(C+i+cindex);
        //c = _mm_add_pd(c, _mm_mul_pd(a, bcol));
        c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1));
        c2 = _mm_add_pd(c2, _mm_mul_pd(a, b2));
        c3 = _mm_add_pd(c3, _mm_mul_pd(a, b3));
        c4 = _mm_add_pd(c4, _mm_mul_pd(a, b4));
        //_mm_storeu_pd(C+i+cindex, c);
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
    for (; i < n; ++i) {
      //for (int k = 0; k < n; ++k) {
        //C[i+cindex] += A[i+k*n] * B[k+j*n];
      //}

      for (int k = 0; k < n; ++k) {
        aindex = k*n;
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

  for (; j < n; ++j) {
    cindex = j*n;
    //for (int i = 0; i < n; ++i) {
    i = 0;
    for (; i < N2; i+=2) {
      //double cij = C[i+j*n];
      //int aindex = k*n;
      //int cindex = j*n;
      //int bindex = k+cindex;
      //__m128d bcol = _mm_load1_pd(B+bindex);
      c = _mm_loadu_pd(C+i+cindex);
      //int k = 0;
      //for(; k < N; k = k+2) {
      for (int k = 0; k < n; k++) {
         //cij += A[i+k*n] * B[k+j*n];
         //C[i+cindex] += A[i+aindex] * B[bindex];
        aindex = k*n;
        bindex = k+cindex;
        a = _mm_loadu_pd(A+i+aindex);
        b = _mm_load1_pd(B+bindex);
        //__m128d c = _mm_loadu_pd(C+i+cindex);
        //c = _mm_add_pd(c, _mm_mul_pd(a, bcol));
        c = _mm_add_pd(c, _mm_mul_pd(a, b));
        //_mm_storeu_pd(C+i+cindex, c);
      }
      //for (; i < n; ++i) {
        //C[i+cindex] += A[i+aindex] * B[bindex];
      //}
      //C[i+j*n] = cij;
      _mm_storeu_pd(C+i+cindex, c);
    }
    for (; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        C[i+cindex] += A[i+k*n] * B[k+j*n];
      }
    }
  }
}

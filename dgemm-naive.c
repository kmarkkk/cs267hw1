const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int j = 0; j < n; ++j)
    /* For each column j of B */
    for (int k = 0; k < n; ++k) 
    {
      /* Compute C(i,j) */
      //double cij = C[i+j*n];
      int aindex = k*n;
      int cindex = j*n;
      int bindex = k+cindex;
      for(int i = 0; i < n; ++i)
	//cij += A[i+k*n] * B[k+j*n];
	C[i+cindex] += A[i+aindex] * B[bindex];
      //C[i+j*n] = cij;
    }
}

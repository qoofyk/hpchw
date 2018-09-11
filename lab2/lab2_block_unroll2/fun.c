#include "fun.h"

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}


/*Debug: print element*/
void show_element(double* C, int n){
	int i;
	for(i=0;i<n*n;i++)
		printf("%f ", C[i]);
	printf("\n");
}

void check_malloc(void * pointer){
  if (pointer == NULL) {
    perror("Malloc error!\n");
    fprintf (stderr, "at %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }
}

/*init every element to 0*/
void init_0(double* C, int n){
	for(int i=0;i<n*n;i++)
		C[i]=0;
}

void init(double* A, int n){
	for(int i=0;i<n*n;i++)
		A[i]=(rand()%10000+1)/1000.0;
}

/*verify the calculation whether equal to verify matrix*/
void verification(double *C, double* verify, int n){
	int i;
	for(i=0;i<n*n;i++){
		if(C[i] != verify[i]){
			printf("i=%d, C[i]=%f Get Wrong result! verify[i]=%f\n",i,C[i],verify[i]);
			fflush(stdout);
		}
	}
}

void dgemm_ijk (int n, double* A, double* B, double* C){
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j){
			register double cij = C[i+j*n]; /* cij = C[i][j] */
			for(int k = 0; k < n; ++k){
				//printf("a=%f, b=%f\n", A[i+k*n],B[k+j*n]);
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
			C[i+j*n] = cij; /* C[i][j] = cij */
			//printf("%f\n", cij);
		}
}

// void block_dgemm_ijk (int N, int b, double* A, double* B, double* C){
// 	int i0, j0,k0,i,j,k;
// 	for (i0=0; i<N; i0+=b)
// 		for (j0=0; j0<N; j0+=b)
// 			for (k0=0; k0<N; k0+=b)
// 				for (i=i0; i<MIN(i0+b,N); i++)
// 					for(j=j0; j<MIN(j0+b,N); j++)
// 						for(k=k0; k<MIN(k0+b,N);k++)
// 							C(i,j) += A(i,k) * B(k,j);
// }
void block_dgemm_ijk (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for (i=i0; i<MIN(i0+b,N); ++i)
					for(j=j0; j<MIN(j0+b,N); ++j){
						register double cij = C(i,j); /* cij = C[i][j] */
						for(k=k0; k<MIN(k0+b,N); ++k){
							cij += A(i,k) * B(k,j);
						}
						C(i,j) = cij;
					}

}

// void dgemm_jik (int n, double* A, double* B, double* C){
// 	for (int j = 0; j < n; ++j)
// 		for (int i = 0; i < n; ++i){
// 			double cij = C[i+j*n]; /* cij = C[i][j] */
// 			for(int k = 0; k < n; k++ ){
// 				//printf("a=%f, b=%f\n", A[i+k*n],B[k+j*n]);
// 				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
// 			}
// 			C[i+j*n] = cij; /* C[i][j] = cij */
// 			//printf("%f\n", cij);
// 		}
// }

void block_dgemm_jik (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for(j=j0; j<MIN(j0+b,N); ++j)
					for (i=i0; i<MIN(i0+b,N); ++i){
						register double cij = C(i,j); /* cij = C[i][j] */
						for(k=k0; k<MIN(k0+b,N); ++k){
							cij += A(i,k) * B(k,j);
						}
						C(i,j) = cij;
					}

}

// void dgemm_kij (int n, double* A, double* B, double* C){
// 	double aik;
// 	for (int k = 0; k < n; ++k)
// 		for (int i = 0; i < n; ++i){
// 			aik = A[i+k*n];
// 			for(int j = 0; j < n; j++ ){
// 				C[i+j*n] += aik * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
// 			}
// 		}
// }

void block_dgemm_kij (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	register double aik;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for(k=k0; k<MIN(k0+b,N);++k)
					for (i=i0; i<MIN(i0+b,N); ++i){
						aik = A(i,k);
						for(j=j0; j<MIN(j0+b,N); ++j)
							C(i,j) += aik * B(k,j);
					}
}


// void dgemm_ikj (int n, double* A, double* B, double* C){
// 	double aik;
// 	for (int i = 0; i < n; ++i)
// 		for (int k = 0; k < n; ++k){
// 			aik = A[i+k*n];
// 			for(int j = 0; j < n; j++ ){
// 				C[i+j*n] += aik * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
// 			}
// 		}
// }

void block_dgemm_ikj (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	register double aik;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for (i=i0; i<MIN(i0+b,N); ++i)
					for(k=k0; k<MIN(k0+b,N);++k){
						aik = A(i,k);
						for(j=j0; j<MIN(j0+b,N); ++j)
							C(i,j) += aik * B(k,j);
					}
}

// void dgemm_jki (int n, double* A, double* B, double* C){
// 	double bkj;
// 	for (int j = 0; j < n; ++j)
// 		for (int k = 0; k < n; ++k){
// 			bkj = B[k+j*n];
// 			for(int i = 0; i < n; i++ ){
// 				C[i+j*n] += A[i+k*n] * bkj; /* cij += A[i][k]*B[k][j] */
// 			}
// 		}
// }

void block_dgemm_jki (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	register double bkj;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for(j=j0; j<MIN(j0+b,N); ++j)
					for(k=k0; k<MIN(k0+b,N);++k){
						bkj = B(k,j);
						for (i=i0; i<MIN(i0+b,N); i=i+2){
							C(i,j) += A(i,k) * bkj;
							C(i+1,j) += A(i+1,k) * bkj;
						}

					}
}

// void dgemm_kji (int n, double* A, double* B, double* C){
// 	double bkj;
// 	for (int k = 0; k < n; ++k)
// 		for (int j = 0; j < n; ++j){
// 			bkj = B[k+j*n];
// 			for(int i = 0; i < n; i++ ){
// 				C[i+j*n] += A[i+k*n] * bkj; /* cij += A[i][k]*B[k][j] */
// 			}
// 		}
// }

void block_dgemm_kji (int N, int b, double* A, double* B, double* C){
	int i0, j0,k0,i,j,k;
	register double bkj;
	for (i0=0; i<N; i0+=b)
		for (j0=0; j0<N; j0+=b)
			for (k0=0; k0<N; k0+=b)
				for(k=k0; k<MIN(k0+b,N);++k)
					for(j=j0; j<MIN(j0+b,N); ++j){
						bkj = B(k,j);
						for (i=i0; i<MIN(i0+b,N); ++i)
							C(i,j) += A(i,k) * bkj;
					}
}

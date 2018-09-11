#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}

void Bublesort(double a[],int n){
     int i,j,k;
     for(j=0;j<n;j++) { /* 气泡法要排序n次*/
          for(i=0;i<n-j;i++) { /* 值比较大的元素沉下去后，只把剩下的元素中的最大值再沉下去就可以啦 */
               if(a[i]>a[i+1]) { /* 把值比较大的元素沉到底 */
                    k=a[i];
                    a[i]=a[i+1];
                    a[i+1]=k;
               }
          }
     }
}

void dgemm_ijk (int n, double* A, double* B, double* C){
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(int k = 0; k < n; k++ )
				cij += A[k+j*n] * B[i+k*n]; /* cij += A[i][k]*B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
}

void dgemm_ikj (int n, double* A, double* B, double* C){
	int i=0,j=0,k=0;
	for (i = 0; i < n; ++i)
		for (k = 0; k < n; ++k){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(j = 0; j < n; j++ )
				cij += A[k+j*n] * B[i+k*n]; /* cij += A[i][k]*B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
}

void dgemm_jik (int n, double* A, double* B, double* C){
	int i=0,j=0,k=0;
	for (j = 0; j < n; ++j)
		for (i = 0; i < n; ++i){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(k = 0; k < n; k++ )
				cij += A[k+j*n] * B[i+k*n]; /* cij += A[i][k]*B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
}

void dgemm_jki (int n, double* A, double* B, double* C){
	int i=0,j=0,k=0;
	double cij;
	// for (j = 0; j < n; ++j)
	// 	for (k = 0; k < n; ++k){
	// 		double cij = C[i+j*n]; /* cij = C[i][j] */
	// 		for(i = 0; i < n; i++ )
	// 			cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
	// 		C[i+j*n] = cij; /* C[i][j] = cij */
	// 	}

	for (int j = 0; j < n; ++j){
		for (int i = 0; i < n; ++i)
			for(int k = 0; k < n; k++ ){
				cij = C[i+j*n]; /* cij = C[i][j] */
				cij += A[k+j*n] * B[i+k*n]; /* cij += A[i][k]*B[k][j] */
				C[i+j*n] = cij; /* C[i][j] = cij */
			}

	}
}


void dgemm_kij (int n, double* A, double* B, double* C){
	int i=0,j=0,k=0;
	for (k = 0; k < n; ++k)
		for (i = 0; i < n; ++i){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(j = 0; j < n; j++ )
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
}

void dgemm_kji (int n, double* A, double* B, double* C){
	int i=0,j=0,k=0;
	for (k = 0; k < n; ++k)
		for (j = 0; j < n; ++j){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(i = 0; i < n; i++ )
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
}

void show_element(double* C, int n){
	int i;
	for(i=0;i<n*n;i++)
		printf("%f ", C[i]);
	printf("\n");
}
int main(int argc, char** argv) {
	double t0, t1;
	int n; //n*n matrix
	//double* a_p, *b_p, *c_p;
	double* c_p;
	double result[6];

	// a_p = (double *)malloc(sizeof(double)*(n*n));
	// b_p = (double *)malloc(sizeof(double)*(n*n));
	double a_p[] = {1,2,3,4,5,6,7,8,9};
	double b_p[] = {9,8,7,6,5,4,3,2,1};
	c_p = (double *)malloc(sizeof(double)*(n*n));

	if(argc != 2) {
		fprintf(stderr, "Usage: %s matrix_size\n", argv[0]);
		exit(1);
	}
	n = atoi(argv[1]);

	t0 = get_cur_time();
	dgemm_ijk(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[0] = t1 - t0;
	show_element(c_p,n);
	printf("1. ijk elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	t0 = get_cur_time();
	dgemm_ikj(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[1] = t1 - t0;
	show_element(c_p,n);
	printf("2. ikj elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	t0 = get_cur_time();
	dgemm_jik(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[2] = t1 - t0;
	show_element(c_p,n);
	printf("3. jik elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	t0 = get_cur_time();
	dgemm_jki(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[3] = t1 - t0;
	show_element(c_p,n);
	printf("4. jki elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	t0 = get_cur_time();
	dgemm_kij(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[4] = t1 - t0;
	show_element(c_p,n);
	printf("5. kij elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	t0 = get_cur_time();
	dgemm_kji(n,a_p,b_p,c_p);
	t1 = get_cur_time();
	result[5] = t1 - t0;
	show_element(c_p,n);
	printf("6. kji elapsed time: %f seconds, Gflops=%f\n", t1 - t0, 2*n*n*n/(t1-t0));
	fflush(stdout);

	Bublesort(result,6);

	return 0;
}

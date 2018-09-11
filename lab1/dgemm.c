/********************************************************
Name:YUANKUN FU
Course name: HPC590
Lab number:1 Sequential Matrix Multiplication
Brief desc of the file: dgemm.c
********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}


void dgemm_ijk (int n, double* A, double* B, double* C){
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(int k = 0; k < n; k++ ){
				//printf("a=%f, b=%f\n", A[i+k*n],B[k+j*n]);
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
			C[i+j*n] = cij; /* C[i][j] = cij */
			//printf("%f\n", cij);
		}
}

void dgemm_jik (int n, double* A, double* B, double* C){
	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i){
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for(int k = 0; k < n; k++ ){
				//printf("a=%f, b=%f\n", A[i+k*n],B[k+j*n]);
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
			C[i+j*n] = cij; /* C[i][j] = cij */
			//printf("%f\n", cij);
		}
}

void dgemm_kij (int n, double* A, double* B, double* C){
	double aik;
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i){
			aik = A[i+k*n];
			for(int j = 0; j < n; j++ ){
				C[i+j*n] += aik * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
		}
}

void dgemm_ikj (int n, double* A, double* B, double* C){
	double aik;
	for (int i = 0; i < n; ++i)
		for (int k = 0; k < n; ++k){
			aik = A[i+k*n];
			for(int j = 0; j < n; j++ ){
				C[i+j*n] += aik * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
		}
}



void dgemm_jki (int n, double* A, double* B, double* C){
	double bkj;
	for (int j = 0; j < n; ++j)
		for (int k = 0; k < n; ++k){
			bkj = B[k+j*n];
			for(int i = 0; i < n; i++ ){
				C[i+j*n] += A[i+k*n] * bkj; /* cij += A[i][k]*B[k][j] */
			}
		}
}


void dgemm_kji (int n, double* A, double* B, double* C){
	double bkj;
	for (int k = 0; k < n; ++k)
		for (int j = 0; j < n; ++j){
			bkj = B[k+j*n];
			for(int i = 0; i < n; i++ ){
				C[i+j*n] += A[i+k*n] * bkj; /* cij += A[i][k]*B[k][j] */
			}
		}
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

int main(int argc, char** argv) {
	double t0, t1;
	int n; //n*n matrix
	double *a_p, *b_p;
	double *c_p,*verify;
	double result[6];
	time_t t;
	int i,loop;

	// double a_p[] = {1,4,7,2,5,8,3,6,9};
	// double b_p[] = {9,6,3,8,5,2,7,4,1};


	if(argc != 3) {
		fprintf(stderr, "Usage: %s matrix_size\n", argv[0]);
		exit(1);
	}
	n = atoi(argv[1]);
	loop = atoi(argv[2]);

	/* Intializes random number generator */
   	srand((unsigned) time(&t));

   	/*Initialise matrix A, B and verify matrix*/
	a_p = (double *)malloc(sizeof(double)*(n*n));
	b_p = (double *)malloc(sizeof(double)*(n*n));
	verify = (double *)malloc(sizeof(double)*(n*n));
	init(a_p,n);
	init(b_p,n);
	init_0(verify,n);
	//generate verification matrix
	printf("generate verification matrix\n");
	fflush(stdout);
	dgemm_ijk(n,a_p,b_p,verify);
	//show_element(verify,n);
	printf("-----------------------------\n");

	/*************compute ijk****************/
	c_p = (double *)malloc(sizeof(double)*(n*n));
	check_malloc(c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	dgemm_ijk(n,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[0] += t1 - t0;
	// }
	// result[0]=result[0]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("1. ijk average elapsed time: %f seconds, Gflops=%f\n", result[0], 2.0*pow(n,3)/(result[0]*1e9));
	// fflush(stdout);
	// //free(c_p);

	// /*************compute jik****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	dgemm_jik(n,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[1] += t1 - t0;
	// }
	// result[1]=result[1]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("2. jik average elapsed time: %f seconds, Gflops=%f\n", result[1], 2.0*pow(n,3)/(result[1]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// /*************compute kij****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	dgemm_kij(n,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[2] += t1 - t0;
	// }
	// result[2]=result[2]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("3. kij average elapsed time: %f seconds, Gflops=%f\n", result[2], 2.0*pow(n,3)/(result[2]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// /*************compute ikj****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	dgemm_ikj(n,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[3] += t1 - t0;
	// }
	// result[3] = result[3]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("4. ikj average elapsed time: %f seconds, Gflops=%f\n", result[3], 2.0*pow(n,3)/(result[3]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// ************compute jki***************
	// c_p = (double *)malloc(sizeof(double)*(n*n));
	// check_malloc(c_p);
	for (i = 0; i < loop; ++i){
		init_0(c_p,n);
		t0 = get_cur_time();
		dgemm_jki(n,a_p,b_p,c_p);
		t1 = get_cur_time();
		result[4] += t1 - t0;
	}
	result[4] = result[4]/loop;
	verification(c_p,verify,n);
	// show_element(c_p,n);
	printf("5. jki average elapsed time: %f seconds, Gflops=%f\n", result[4], 2.0*pow(n,3)/(result[4]*1e9));
	fflush(stdout);
	// free(c_p);


	// /*************compute kji****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// for(i=0;i<loop;++i){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	dgemm_kji(n,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[5] += t1 - t0;
	// }
	// result[5] = result[5]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("6. kji average elapsed time: %f seconds, Gflops=%f\n", result[5], 2.0*pow(n,3)/(result[5]*1e9));
	// fflush(stdout);
	// // free(c_p);

	free(c_p);
	free(verify);

	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define A(i,j) A[i+j*N]
#define B(i,j) B[i+j*N]
#define C(i,j) C[i+j*N]

double get_cur_time();
void show_element(double* C, int n);
void check_malloc(void * pointer);
void init_0(double* C, int n);
void init(double* A, int n);
void verification(double *C, double* verify, int n);
void dgemm_ijk (int n, double* A, double* B, double* C);
void block_dgemm_ijk (int N, int b, double* A, double* B, double* C);
void block_dgemm_jik (int N, int b, double* A, double* B, double* C);
void block_dgemm_kij (int N, int b, double* A, double* B, double* C);
void block_dgemm_ikj (int N, int b, double* A, double* B, double* C);
void block_dgemm_jki (int N, int b, double* A, double* B, double* C);
void block_dgemm_kji (int N, int b, double* A, double* B, double* C);

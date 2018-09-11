#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define A(i,j) A[i+j*gv->n]
#define B(i,j) B[i+j*gv->n]
#define C(i,j) C[i+j*gv->n]

#define a_p(i,j) a_p+(i+j*gv->n)*sizeof(double)
#define b_p(i,j) b_p+(i+j*gv->n)*sizeof(double)

#define DGEMM dgemm_

typedef struct gv_t {
	int rank[2], size[2], namelen, color;
	char processor_name[128];
	int sqrt_process_size;

	int N; //N*N matrix
	int n; //each process will get n*n matrix
	int start_position;
	int num_row;

}* GV;

double get_cur_time();
void show_element(double* C, int n);
void check_malloc(void * pointer);
void check_MPI_success(GV gv, int errorcode);
void init_1(double* C, int n);
void init_0(double* C, int n);
void init(double* A, int n);
void verification(GV gv, double *C, int n);
void dgemm_ijk (int n, double* A, double* B, double* C);

extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
// const char* dgemm_desc = "Reference dgemm.";


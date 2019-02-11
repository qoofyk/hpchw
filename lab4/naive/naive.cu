#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	double* elements;
} Matrix;

typedef struct gv_t {
	int N; //N*N matrix
	int loop;
	int block_size; // Thread block size
}* GV;

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
double MatMul(GV gv, const Matrix A, const Matrix B, Matrix C) {
	float milliseconds = 0;
  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double); cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(double);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(double);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(gv->block_size, gv->block_size);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	cudaEventRecord(start);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaEventRecord(stop);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);

	cudaEventElapsedTime(&milliseconds, start, stop);

	// printf("Execution time=%f ms, Gflops=%f\n", milliseconds, 2.0*pow(gv->N,3)/milliseconds/1e6);
	// fflush(stdout);

	return milliseconds;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	double Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]
				* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}

void init(double* A, int n){
	for(int i=0;i<n*n;i++)
		A[i]=(rand()%10000+1)/1000.0;
}

/*init every element to 0*/
void init_0(double* C, int n){
	for(int i=0;i<n*n;i++)
		C[i]=0;
}

/*init every element to 1*/
void init_1(double* C, int n){
	for(int i=0;i<n*n;i++)
		C[i]=1;
}

/*verify the calculation whether equal to verify matrix*/
void verification(GV gv, double *C, int n){
	int i;
	for(i=0;i<n*n;i++){
		if(C[i] != gv->N){
			printf("i=%d, C[i]=%f Get Wrong result!\n",i,C[i]);
			fflush(stdout);
		}
	}
}

void check_malloc(void * pointer){
  if (pointer == NULL) {
    perror("Malloc error!\n");
    fprintf (stderr, "at %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }
}

int main(int argc, char** argv) {
	Matrix A, B, C;
	double elapsed_time;
	time_t t;
	int i;
	GV gv;
	// double *B;
	// double a_p[] = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	// double b_p[] = {16,12,8,4,15,11,7,3,14,10,6,2,13,9,5,1};


	if(argc != 4) {
		fprintf(stderr, "Usage: %s real_matrix_size, thread block_size\n", argv[0]);
		exit(1);
	}

	gv    = (GV) malloc(sizeof(*gv));

	gv->N = atoi(argv[1]);
	gv->block_size = atoi(argv[2]);
	gv->loop = atoi(argv[3]);
	printf("N=%d,block_size=%d,loop=%d\n",
		gv->N,gv->block_size,gv->loop);

	/* Intializes random number generator */
   	srand((unsigned) time(&t));

   	/*Initialise matrix A, B and verify matrix*/
 	//printf("Initialise A B matrix\n");
	// printf("-----------------------------\n");
	// fflush(stdout);
	A.width = gv->N;
	A.height = gv->N;
	A.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
	check_malloc(A.elements);

	B.width = gv->N;
	B.height = gv->N;
	B.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
	check_malloc(B.elements);

	C.width = gv->N;
	C.height = gv->N;
	C.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
	check_malloc(C.elements);

	// verify 1
	init_1(A.elements,gv->N);
	init_1(B.elements,gv->N);
	init_0(C.elements,gv->N);
	MatMul(gv,A,B,C);
	verification(gv,C.elements,gv->N);
	printf("Pass VERIFY 1\n");
	fflush(stdout);

	// start calculation
	init(A.elements,gv->N);
	init(B.elements,gv->N);
	elapsed_time=0;
	for (i = 0; i < gv->loop; ++i){
		init_0(C.elements,gv->N);
		elapsed_time += MatMul(gv,A,B,C);
	}
	elapsed_time = elapsed_time/gv->loop;
	printf("CALCULATION: Naive GPU average elapsed time: %f seconds, Gflops=%f\n",
		elapsed_time/1e3, 2.0*pow(gv->N,3)/elapsed_time/1e6);

	free(A.elements);
	free(B.elements);
	free(C.elements);

	free(gv);

	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// Thread block size
#define BLOCK_SIZE 32
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    double* elements;
} Matrix;

typedef struct gv_t {
    int N; //N*N matrix
    int loop;
}* GV;

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col){
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, double value){
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
double MatMul(GV gv, const Matrix A, const Matrix B, Matrix C){
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(double);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    cudaEventRecord(start);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time=%f ms, Gflops=%f\n", milliseconds, 2.0*pow(gv->N,3)/milliseconds/1e6);
    fflush(stdout);

    return milliseconds;
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub by accumulating results into Cvalue
    double Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    #pragma unroll
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory. Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
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

void show_element(GV gv, double* C, int n){
    int i;
    for(i=0;i<n;i++){
        printf("%f ", C[i]);
        fflush(stdout);
    }
    printf("\n");
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


    if(argc != 3) {
        fprintf(stderr, "Usage: %s real_matrix_size, loop\n", argv[0]);
        exit(1);
    }

    gv    = (GV) malloc(sizeof(*gv));

    gv->N = atoi(argv[1]);
    gv->loop = atoi(argv[2]);
    printf("N=%d,loop=%d\n", gv->N,gv->loop);

    /* Intializes random number generator */
    srand((unsigned) time(&t));

    /*Initialise matrix A, B and verify matrix*/
    //printf("Initialise A B matrix\n");
    // printf("-----------------------------\n");
    // fflush(stdout);
    A.width = A.height = A.stride = gv->N;
    A.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
    check_malloc(A.elements);

    B.width = B.height = B.stride=gv->N;
    B.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
    check_malloc(B.elements);

    C.width = C.height = C.stride=gv->N;
    C.elements = (double *)malloc(sizeof(double)*(gv->N*gv->N));
    check_malloc(C.elements);

    // verify 1
    init_1(A.elements,gv->N);
    // show_element(gv,A.elements,gv->N*gv->N);
    init_1(B.elements,gv->N);
    // show_element(gv,B.elements,gv->N*gv->N);
    init_0(C.elements,gv->N);
    elapsed_time = MatMul(gv,A,B,C);
    // show_element(gv,C.elements,gv->N*gv->N);
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
    printf("CALCULATION: Naive GPU average elapsed time: %f milliseconds, Gflops=%f\n",
        elapsed_time, 2.0*pow(gv->N,3)/elapsed_time/1e6);

    free(A.elements);
    free(B.elements);
    free(C.elements);

    free(gv);

    return 0;
}

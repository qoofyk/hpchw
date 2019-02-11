#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 8192 // a - mxk matrix
#define n 8192 // b - kxn matrix
#define k 8192 // c - mxn matrix

void init(double* A, int t){
   for(int i=0;i<t*t;i++)
      A[i]=(rand()%10000+1)/1000.0;
}

int main (int argc, char** argv){
   cudaError_t cudaStat ; // cudaMalloc status
   cublasStatus_t stat ; // CUBLAS functions status
   cublasHandle_t handle ; // CUBLAS context
   time_t t;

   float milliseconds = 0;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // int i,j; // i-row index ,j- column index
   double * a; // mxk matrix a on the host
   double * b; // kxn matrix b on the host
   double * c; // mxn matrix c on the host

   /* Intializes random number generator */
   srand((unsigned) time(&t));

   a=(double *) malloc (m*k* sizeof (double)); // host memory for a
   b=(double *) malloc (k*n* sizeof (double)); // host memory for b
   c=(double *) malloc (m*n* sizeof (double)); // host memory for c

   // define an mxk matrix a column by column
   // int ind =11; // a:
   // for(j=0;j<k;j++){ // 11 ,17 ,23 ,29 ,35
   //    for(i=0;i<m;i++){ // 12 ,18 ,24 ,30 ,36
   //       a[IDX2C(i,j,m)]=(double)ind++; // 13 ,19 ,25 ,31 ,37
   //    } // 14 ,20 ,26 ,32 ,38
   // } // 15 ,21 ,27 ,33 ,39


   // 16 ,22 ,28 ,34 ,40
   // print a row by row
   // printf ("a:\n");
   // for(i=0;i<m;i++){
   //    for(j=0;j<k;j++){
   //       printf (" %f",a[ IDX2C (i,j,m )]);
   //    }
   //    printf ("\n");
   // }

   // define a kxn matrix b column by column
   // ind =11; // b:
   // for(j=0;j<n;j ++){ // 11 ,16 ,21 ,26
   //    for(i=0;i<k;i ++){ // 12 ,17 ,22 ,27
   //       b[IDX2C(i,j,k)]=(double)ind ++; // 13 ,18 ,23 ,28
   //    } // 14 ,19 ,24 ,29
   // } // 15 ,20 ,25 ,30

   // print b row by row
   // printf ("b:\n");
   // for(i=0;i<k;i++){
   //    for(j=0;j<n;j++){
   //       printf (" %f",b[IDX2C(i,j,k)]);
   //    }
   //    printf ("\n");
   // }

   // define an mxn matrix c column by column
   // ind =11; // c:
   // for(j=0;j<n;j++){ // 11 ,17 ,23 ,29
   //    for(i=0;i<m;i++){ // 12 ,18 ,24 ,30
   //       c[IDX2C(i,j,m)]=(double)ind ++; // 13 ,19 ,25 ,31
   //    } // 14 ,20 ,26 ,32
   // } // 15 ,21 ,27 ,33

   // 16 ,22 ,28 ,34
   // print c row by row
   // printf ("c:\n");
   // for(i=0;i<m;i++){
   //    for(j=0;j<n;j++){
   //       printf (" %f",c[ IDX2C (i,j,m )]);
   //    }
   //    printf ("\n");
   // }

   init(a,m);
   init(b,n);

   // on the device
   double * d_a; // d_a - a on the device
   double * d_b; // d_b - b on the device
   double * d_c; // d_c - c on the device

   cudaStat = cudaMalloc (( void **)& d_a ,m*k* sizeof (*a)); // device memory alloc for a
   cudaStat = cudaMalloc (( void **)& d_b ,k*n* sizeof (*b)); // device memory alloc for b
   cudaStat = cudaMalloc (( void **)& d_c ,m*n* sizeof (*c)); // device memory alloc for c

   stat = cublasCreate (& handle ); // initialize CUBLAS context
   // copy matrices from the host to the device
   stat = cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); //a -> d_a
   stat = cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); //b -> d_b
   stat = cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); //c -> d_c

   double al =1.0; // al =1
   double bet =1.0; // bet =1

   // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
   // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
   // al ,bet -scalars
   cudaEventRecord(start);
   stat=cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m);
   cudaEventRecord(stop);

   stat = cublasGetMatrix (m,n,sizeof(*c),d_c,m,c,m); // cp d_c - >c
   cudaEventSynchronize(stop);
   // printf ("c after Dgemm :\n");
   // for(i=0;i<m;i ++){
   //    for(j=0;j<n;j ++){
   //       printf (" %f",c[IDX2C(i,j,m)]); // print c after Sgemm
   //    }
   //    printf ("\n");
   // }

   cudaFree(d_a); // free device memory
   cudaFree(d_b); // free device memory
   cudaFree(d_c); // free device memory

   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Execution time=%f ms, Gflops=%f\n", milliseconds, 2.0*pow(m,3)/milliseconds/1e6);
   fflush(stdout);

   cublasDestroy ( handle ); // destroy CUBLAS context

   free (a); // free host memory
   free (b); // free host memory
   free (c); // free host memory

   return EXIT_SUCCESS ;
}

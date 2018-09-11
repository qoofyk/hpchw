/********************************************************
Name:YUANKUN FU
Course name: HPC590
Lab number:1 Sequential Matrix Multiplication
Brief desc of the file: dgemm.c
********************************************************/
#include "fun.h"

int main(int argc, char** argv) {
	double t0, t1;
	int n; //n*n matrix
	double *a_p, *b_p;
	double *c_p,*verify;
	double result[6];
	time_t t;
	int i,loop;
	int block_size;

	// double a_p[] = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	// double b_p[] = {16,12,8,4,15,11,7,3,14,10,6,2,13,9,5,1};


	if(argc != 4) {
		fprintf(stderr, "Usage: %s matrix_size\n", argv[0]);
		exit(1);
	}
	n = atoi(argv[1]);
	loop = atoi(argv[2]);
	block_size = atoi(argv[3]);

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
	// show_element(verify,n);
	printf("-----------------------------\n");

	/**************compute block ijk****************/
	c_p = (double *)malloc(sizeof(double)*(n*n));
	check_malloc(c_p);
	// // block_dgemm_ijk(n,block_size,a_p,b_p,c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	block_dgemm_ijk(n,block_size,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[0] += t1 - t0;
	// }
	// result[0]=result[0]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("1. block_ijk average elapsed time: %f seconds, Gflops=%f\n", result[0], 2.0*pow(n,3)/(result[0]*1e9));
	// fflush(stdout);
	// //free(c_p);*/

	// /*************compute jik****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// // init_0(c_p,n);
	// // block_dgemm_jik(n,block_size,a_p,b_p,c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	block_dgemm_jik(n,block_size,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[1] += t1 - t0;
	// }
	// result[1]=result[1]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("2. block_jik average elapsed time: %f seconds, Gflops=%f\n", result[1], 2.0*pow(n,3)/(result[1]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// /*************compute kij****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// // init_0(c_p,n);
	// // block_dgemm_kij(n,block_size,a_p,b_p,c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	block_dgemm_kij(n,block_size,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[2] += t1 - t0;
	// }
	// result[2]=result[2]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("3. block_kij average elapsed time: %f seconds, Gflops=%f\n", result[2], 2.0*pow(n,3)/(result[2]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// /*************compute ikj****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// // init_0(c_p,n);
	// // block_dgemm_ikj(n,block_size,a_p,b_p,c_p);
	// for(i=0;i<loop;i++){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	block_dgemm_ikj(n,block_size,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[3] += t1 - t0;
	// }
	// result[3] = result[3]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("4. block_ikj average elapsed time: %f seconds, Gflops=%f\n", result[3], 2.0*pow(n,3)/(result[3]*1e9));
	// fflush(stdout);
	// // free(c_p);


	// ************compute jki***************
	// c_p = (double *)malloc(sizeof(double)*(n*n));
	// check_malloc(c_p);
	// init_0(c_p,n);
	// block_dgemm_jki(n,block_size,a_p,b_p,c_p);
	for (i = 0; i < loop; ++i){
		init_0(c_p,n);
		t0 = get_cur_time();
		block_dgemm_jki(n,block_size,a_p,b_p,c_p);
		t1 = get_cur_time();
		result[4] += t1 - t0;
	}
	result[4] = result[4]/loop;
	verification(c_p,verify,n);
	// show_element(c_p,n);
	printf("5. block_jki average elapsed time: %f seconds, Gflops=%f\n", result[4], 2.0*pow(n,3)/(result[4]*1e9));
	fflush(stdout);
	// free(c_p);


	// /*************compute kji****************/
	// // c_p = (double *)malloc(sizeof(double)*(n*n));
	// // check_malloc(c_p);
	// // init_0(c_p,n);
	// // block_dgemm_kji(n,block_size,a_p,b_p,c_p);
	// for(i=0;i<loop;++i){
	// 	init_0(c_p,n);
	// 	t0 = get_cur_time();
	// 	block_dgemm_kji(n,block_size,a_p,b_p,c_p);
	// 	t1 = get_cur_time();
	// 	result[5] += t1 - t0;
	// }
	// result[5] = result[5]/loop;
	// verification(c_p,verify,n);
	// // show_element(c_p,n);
	// printf("6. block_kji average elapsed time: %f seconds, Gflops=%f\n", result[5], 2.0*pow(n,3)/(result[5]*1e9));
	// fflush(stdout);
	// // free(c_p);

	free(c_p);
	free(verify);

	return 0;
}

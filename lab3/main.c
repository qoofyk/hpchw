/********************************************************
Name:YUANKUN FU
Course name: HPC590
Lab number:1 Sequential Matrix Multiplication
Brief desc of the file: dgemm.c
********************************************************/
#include "fun.h"

int main(int argc, char** argv) {
	double t0, t1;
	double *a_p, *b_p;
	double *c_p;
	double result;
	time_t t;
	int i;
	GV gv;
	// double *B;
	// double a_p[] = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	// double b_p[] = {16,12,8,4,15,11,7,3,14,10,6,2,13,9,5,1};


	if(argc != 5) {
		fprintf(stderr, "Usage: %s real_matrix_size, each process_matrix_size\n", argv[0]);
		exit(1);
	}

	gv    = (GV) malloc(sizeof(*gv));

	gv->N = atoi(argv[1]);
	gv->n = atoi(argv[2]);
	gv->block_size = atoi(argv[3]);
	gv->loop = atoi(argv[4]);
	printf("N=%d,n=%d,block_size=%d,loop=%d\n",
		gv->N,gv->n,gv->block_size,gv->loop);

	/* Intializes random number generator */
   	srand((unsigned) time(&t));

   	/*Initialise matrix A, B and verify matrix*/
 //   	printf("Initialise A B matrix\n");
	// printf("-----------------------------\n");
	// fflush(stdout);
	a_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));
	b_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));

	// init_1(a_p,gv->n);
	// // show_element(gv,a_p,gv->n*gv->n,0);
	// init_1(b_p,gv->n);
	// // show_element(gv,b_p,gv->n*gv->n,0);
	// B=b_p;
	// // show_row_element(gv,B(1,0),gv->n,0);



	/**************compute block ijk****************/
	c_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));
	check_malloc(c_p);

	MPI_Init(&argc, &argv);
	// MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &gv->rank[0]);
	MPI_Comm_size(MPI_COMM_WORLD, &gv->size[0]);
	MPI_Get_processor_name(gv->processor_name, &gv->namelen);
	gv->sqrt_process_size=sqrt(gv->size[0]);
	printf("Hello world! I’m rank %d of %d on %s, row=column=%d\n",
		gv->rank[0], gv->size[0], gv->processor_name,gv->sqrt_process_size);
	fflush(stdout);

	// verify 1
	init_1(a_p,gv->n);
	init_1(b_p,gv->n);
	result=0;
	init_0(c_p,gv->n);
	t0 = get_cur_time();
	summa(gv,a_p,b_p,c_p);
	t1 = get_cur_time();
	result += t1 - t0;
	verification(gv,c_p,gv->n);
	printf("Pass VERIFY 1\n");
	fflush(stdout);

	// verify 2
	init(a_p,gv->n);
	// show_element(gv,a_p,gv->n*gv->n,0);
	init_0(b_p,gv->n);
	for(i=0;i<gv->sqrt_process_size;i++){
		if(gv->rank[0]==(i+gv->sqrt_process_size*i)){
			init_angle(gv,b_p,gv->n);
			break;
		}
	}
	// show_element(gv,b_p,gv->n*gv->n,0);
	init_0(c_p,gv->n);
	t0 = get_cur_time();
	summa(gv,a_p,b_p,c_p);
	t1 = get_cur_time();
	result += t1 - t0;
	verification_angle(gv,a_p,c_p,gv->n);
	printf("Pass VERIFY ANGLE\n");
	fflush(stdout);

	// start calculation
	init(a_p,gv->n);
	init(b_p,gv->n);
	result=0;
	for (i = 0; i < gv->loop; ++i){
		init_0(c_p,gv->n);
		t0 = get_cur_time();
		summa(gv,a_p,b_p,c_p);
		t1 = get_cur_time();
		result += t1 - t0;
	}
	result = result/gv->loop;
	printf("CALCULATION: SUMMA average elapsed time: %f seconds, Gflops=%f\n",
		result, 2.0*pow(gv->N,3)/(gv->size[0]*result*1e9));

	free(a_p);
	free(b_p);
	free(c_p);

	MPI_Finalize();
	free(gv);

	return 0;
}

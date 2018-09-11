#include "fun.h"

/*Debug: print element*/
void show_element(double* C, int n){
	int i;
	for(i=0;i<n;i++)
		printf("%f ", C[i]);
	printf("\n");
	fflush(stdout);
}

/*init every element to 1*/
void init_a(GV gv, double* C, int n){
	int id=1;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			C(i,j)=id++;
}
void init_b(GV gv, double* C, int n){
	int id=n*n;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			C(i,j)=id--;
}

void check_malloc(void * pointer){
  if (pointer == NULL) {
    perror("Malloc error!\n");
    fprintf (stderr, "at %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are N-by-N matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * This function wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */
void square_dgemm (int N, double* A, double* B, double* C){
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  double ALPHA = 1.;
  double BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void rectangle_dgemm (int m, int k, double* A, double* B, double* C){
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = m;
  int K = k;
  int N = m;
  double ALPHA = 1.;
  double BETA = 1.;
  int LDA = m;
  int LDB = m;
  int LDC = m;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

int main(int argc, char** argv) {
	double t0, t1;

	double *c_p;
	double result;
	time_t t;
	int i;
	GV gv;
	// double a_p[] = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	// double b_p[] = {16,12,8,4,15,11,7,3,14,10,6,2,13,9,5,1};
	double *a_p, *b_p;

	if(argc != 5) {
		fprintf(stderr, "Usage: %s real_matrix_size, each process_matrix_size\n", argv[0]);
		exit(1);
	}

	gv    = (GV) malloc(sizeof(*gv));

	gv->N = atoi(argv[1]);
	gv->n = atoi(argv[2]);
	gv->num_row = atoi(argv[3]);
	gv->start_position = atoi(argv[4]);

	/* Intializes random number generator */
   	// srand((unsigned) time(&t));

   	/*Initialise matrix A, B and verify matrix*/
	a_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));
	b_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));
	// d_p = (double *)malloc(sizeof(double)*(gv->num_row*gv->n));

	//generate verification matrix
	printf("generate verification matrix\n");
	fflush(stdout);
	init_a(gv,a_p,gv->n);
	show_element(a_p,gv->n*gv->n);
	init_b(gv,b_p,gv->n);
	show_element(b_p,gv->n*gv->n);
	printf("-----------------------------\n");
	fflush(stdout);

	/**************use dgemm****************/
	c_p = (double *)malloc(sizeof(double)*(gv->n*gv->n));
	check_malloc(c_p);

	show_element(a_p(0,gv->start_position),gv->n*gv->num_row);
	show_element(b_p(gv->start_position,0),gv->n*gv->num_row);
	rectangle_dgemm(gv->n,gv->num_row,a_p(0,gv->start_position),b_p(gv->start_position,0),c_p);

	// verification(gv,c_p,gv->n);
	show_element(c_p,gv->n*gv->n);


	// free(a_p);
	// free(b_p);
	free(c_p);

	free(gv);

	return 0;
}

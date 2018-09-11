#include "fun.h"

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}


/*Debug: print element*/
void show_element(GV gv, double* C, int n, int global_k){
	int i;
	for(i=0;i<n;i++)
		printf("%f ", C[i]);
	printf("-- Rank%d,global_k=%d send column\n", gv->rank[0],global_k);
}

void show_row_element(GV gv, double* C, int n, int global_k){
	int i;
	for(i=0;i<n;i++)
		printf("%f ", *(C+i*gv->n));
	printf("-- Rank%d,global_k=%d send row\n", gv->rank[0],global_k);
}

void check_malloc(void * pointer){
  if (pointer == NULL) {
    perror("Malloc error!\n");
    fprintf (stderr, "at %s, line %d.\n", __FILE__, __LINE__);
    exit(1);
  }
}

void check_MPI_success(GV gv, int errorcode){
  if(errorcode!= MPI_SUCCESS){
    perror("MPI_SEND not MPI_SUCCESS!\n");
    fprintf (stderr, "Node %d at  %s, line %d.\n", gv->rank[0], __FILE__, __LINE__);
    exit(1);
  }
}

/*init every element to 0*/
void init_0(double* C, int n){
	for(int i=0;i<n*n;i++)
		C[i]=0;
}


/*init angle to 1*/
void init_angle(GV gv, double* C, int n){
	int i,j;
	for(int i=0;i<n;i++)
		C(i,i)=1;
}

/*init every element to 1*/
void init_1(double* C, int n){
	for(int i=0;i<n*n;i++)
		C[i]=1;
}

void init(double* A, int n){
	for(int i=0;i<n*n;i++)
		A[i]=(rand()%10000+1)/1000.0;
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

/*verify the calculation whether equal to verify matrix*/
void verification_angle(GV gv, double *A, double *C, int n){
	int i;
	for(i=0;i<n*n;i++){
		if(A[i] != C[i]){
			printf("VERIFY2 i=%d, C[i]=%f Get Wrong result!\n",i,C[i]);
			fflush(stdout);
		}
	}
}

void dgemm_ijk (int n, double* A, double* B, double* C){
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j){
			register double cij = C[i+j*n]; /* cij = C[i][j] */
			for(int k = 0; k < n; ++k){
				//printf("a=%f, b=%f\n", A[i+k*n],B[k+j*n]);
				cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
			}
			C[i+j*n] = cij; /* C[i][j] = cij */
			//printf("%f\n", cij);
		}
}

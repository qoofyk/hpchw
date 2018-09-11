#include "fun.h"

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

void rectangle_dgemm (int m, int k, double* A, double* B, double* C, int LDA, int LDB){
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = m;
  int K = k;
  int N = m;
  double ALPHA = 1.;
  double BETA = 1.;
  // int LDA = m;
  // int LDB = m;
  int LDC = m;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void summa(GV gv, double* A, double* B, double* C){
	int errorcode;
	int dest,src;
	int buffer_length;
	MPI_Status status;
	int root;
	int my_row_idx,my_column_idx;
	int i,j,k;
	int global_k;

	MPI_Datatype block_row;
	MPI_Type_vector(gv->n,gv->block_size,gv->n,MPI_DOUBLE, &block_row);
	MPI_Type_commit(&block_row);

	double* recv_A, *recv_B;

	recv_A = (double *)malloc(sizeof(double)*(gv->n*gv->block_size));
	recv_B = (double *)malloc(sizeof(double)*(gv->block_size*gv->n));

	my_row_idx = gv->rank[0]/gv->sqrt_process_size;
	my_column_idx = gv->rank[0]%gv->sqrt_process_size;

	buffer_length = gv->n*gv->block_size;

	// printf("*****Rank=%d Begin summa,my_row_idx=%d,my_column_idx=%d******\n",
	// 	gv->rank[0],my_row_idx,my_column_idx);
	// fflush(stdout);

	// global_k:global row_index, column_index
	for(global_k=0;global_k<gv->N;global_k=global_k+gv->block_size){
		root = global_k/gv->n;

		// if(my_row_idx==root && my_column_idx==root){
		// 	printf("-----Rank=%d global_k=%d root=%d-----\n",
		// 		gv->rank[0],global_k,root);
		// 	fflush(stdout);
		// }

		k=global_k%gv->n;

		// case 1
		if(my_row_idx==root && my_column_idx==root){
			// broadcast  A[i,k]
			// printf("Case 1 Broadcast: Rank%d Send A column, global_k=%d\n", gv->rank[0], global_k);
			// fflush(stdout);
			// show_element(gv,A(0,k),gv->n*gv->block_size,global_k);
			for(i=0;i<gv->sqrt_process_size;i++){
				dest=my_row_idx*gv->sqrt_process_size+i;
				if(dest==gv->rank[0]) continue;
				errorcode = MPI_Send(A(0,k),buffer_length,MPI_DOUBLE,dest,TAG_A,MPI_COMM_WORLD);
		    	check_MPI_success(gv, errorcode);
			}

			// broadcast B[k,j]
			// printf("Case 1 Broadcast: Rank%d Send B row, global_k=%d\n", gv->rank[0], global_k);
			// fflush(stdout);
			// for(i=0;i<gv->block_size;i++)
			// 	show_row_element(gv,B(k+i,0),gv->n,global_k);
			for(j=0;j<gv->sqrt_process_size;j++){
				dest=my_column_idx+j*gv->sqrt_process_size;
				if(dest==gv->rank[0]) continue;
				errorcode = MPI_Send(B(k,0),1,block_row,dest,TAG_B,MPI_COMM_WORLD);
		    	check_MPI_success(gv, errorcode);
			}

			//compute C
			rectangle_dgemm(gv->n,gv->block_size,A(0,k),B(k,0),C,gv->n,gv->n);
		}
		// case 2
		else if(my_row_idx==root && my_column_idx!=root){
			// broadcast B[k,j]
			// printf("Case 2 Broadcast: Rank%d Send B row, global_k=%d\n", gv->rank[0], global_k);
			// fflush(stdout);
			// for(i=0;i<gv->block_size;i++)
			// 	show_row_element(gv,B(k+i,0),gv->n,global_k);
			for(j=0;j<gv->sqrt_process_size;j++){
				dest=my_column_idx+j*gv->sqrt_process_size;
				if(dest==gv->rank[0]) continue;
				errorcode = MPI_Send(B(k,0),1,block_row,dest,TAG_B,MPI_COMM_WORLD);
		    	check_MPI_success(gv, errorcode);
			}

			// receive A[i,k]
			src=my_row_idx*gv->sqrt_process_size+root;
			// printf("Case 2 Recv: Rank%d Recv A from %d, global_k=%d\n", gv->rank[0], src, global_k);
			// fflush(stdout);
			errorcode = MPI_Recv(recv_A, buffer_length, MPI_DOUBLE, src, TAG_A, MPI_COMM_WORLD, &status);

			//compute C
			rectangle_dgemm(gv->n,gv->block_size,recv_A,B(k,0),C,gv->n,gv->n);
		}
		// case 3
		else if(my_row_idx!=root && my_column_idx==root){
			// broadcast  A[i,k]
			// printf("Case 3 Broadcast: Rank%d Send A column, global_k=%d\n", gv->rank[0], global_k);
			// fflush(stdout);
			// show_element(gv,A(0,k),gv->n*gv->block_size,global_k);
			for(i=0;i<gv->sqrt_process_size;i++){
				dest=my_row_idx*gv->sqrt_process_size+i;
				if(dest==gv->rank[0]) continue;
				errorcode = MPI_Send(A(0,k),buffer_length,MPI_DOUBLE,dest,TAG_A,MPI_COMM_WORLD);
		    	check_MPI_success(gv, errorcode);
			}

			//receive B[k,j]
			src=root*gv->sqrt_process_size+my_column_idx;
			// printf("Case 3 Recv: Rank%d Recv B row from %d, global_k=%d\n", gv->rank[0], src, global_k);
			// fflush(stdout);
			errorcode = MPI_Recv(recv_B, gv->block_size*gv->n, MPI_DOUBLE, src, TAG_B, MPI_COMM_WORLD, &status);

			//compute C
			rectangle_dgemm(gv->n,gv->block_size,A(0,k),recv_B,C,gv->n,gv->block_size);
		}
		//Case 4
		else if(my_row_idx!=root && my_column_idx!=root){
			//receive A[i,k]
			src=my_row_idx*gv->sqrt_process_size+root;
			// printf("Case 4 Recv: Rank%d Recv A from %d, global_k=%d\n", gv->rank[0], src, global_k);
			// fflush(stdout);
			errorcode = MPI_Recv(recv_A, buffer_length, MPI_DOUBLE, src, TAG_A, MPI_COMM_WORLD, &status);

			//receive B[k,j]
			src=root*gv->sqrt_process_size+my_column_idx;
			// printf("Case 4 Recv: Rank%d Recv B from %d, global_k=%d\n", gv->rank[0], src, global_k);
			// fflush(stdout);
			errorcode = MPI_Recv(recv_B, buffer_length, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_B, MPI_COMM_WORLD, &status);

			//compute C
			rectangle_dgemm(gv->n,gv->block_size,recv_A,recv_B,C,gv->n,gv->block_size);
		}
		else{
			printf("Error!\n");
			fflush(stdout);
		}

	}

}

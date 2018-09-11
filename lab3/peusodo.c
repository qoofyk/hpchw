/********************************************************
Name:YUANKUN FU
Course name: HPC590
Lab number:1 Sequential Matrix Multiplication
Brief desc of the file: dgemm.c
********************************************************/

int main(int argc, char** argv) {
	N ,n, block_size, loop = atoi();
	...
	geneart Matrix A,B,C;
	...
	MPI_Initialise;
	...
	All_1_matrix_verification(gv,c_p,gv->n);
	...
	diangonal_matrix_verification(gv,a_p,c_p,gv->n);
	...
	Generate random_number in A B;
	for (loop)
		summa(gv,a_p,b_p,c_p);
	Get average elapsed_time;
	Calculate GFlops;
}


void summa(GV gv, double* A, double* B, double* C){
	Generate MPI_Type_vector block_row;
	Generate receive buffer recv_A, recv_B;
	Get my_row_idx and my_column_idx in the process matrix;
	// global_k:global row_index, column_index
	for(global_k=0;global_k<gv->N;global_k=global_k+gv->block_size){
		root = global_k/gv->n;
		if(my_row_idx==root && my_column_idx==root){// case 1
			broadcast  A[i,k]
			broadcast B[k,j]
			Use my own A and B to compute C with dgemm;
		}
		else if(my_row_idx==root && my_column_idx!=root){// case 2
			broadcast B[k,j]
			receive A[i,k]
			Use recv_A and my own B to compute C with dgemm;
		}
		else if(my_row_idx!=root && my_column_idx==root){// case 3
			broadcast  A[i,k];
			receive B[k,j];
			Use my own A and recv_B to compute C with dgemm;
		}
		else(my_row_idx!=root && my_column_idx!=root){//Case 4
			receive A[i,k];
			receive B[k,j];
			Use recv_A and recv_B to compute C with dgemm;
		}
	}
}

/********************************************************
Name:YUANKUN FU
Course name: HPC590
Lab number:1 Sequential Matrix Multiplication
Brief desc of the file: README
********************************************************/

How to run?
1.compile
fuyuan@login2:~/hpchw/hw1> make clean;make
rm -f dgemm.o dgemm
rm -f *~ core
cc -c -O3 dgemm.c
cc -o dgemm dgemm.o

2. Run code
$> ./dgemm 1000 10
The first argument is input matrix size, e.g. here means 1000*1000 matrix multiplication. C=A*B.
The second argument means that each function dgemm_ijk,	ikj, jik, jki, kij, kji
will compute 10 times and then get the average elapsed time and GFLOPS.

3. Run batch job.
Here I write a qsub jobs_2000.pbs
Just qsub jobs_2000.pbs to get results from matrix size range from 10 to 2000.
From range 10~600, I run each function with 20 times.
From range 700~1500, I run each function with 8 times.
From range 1600~2000, I run each function with 3 times.

total_process=16
sqrt_process_size=4
each_process_matrixsize_n=1024
real_matrixsize_N=`expr $each_process_matrixsize_n \* $sqrt_process_size`
# inner_loop=4
# block_size=64
my_run_exp1="aprun -n $total_process -N 16 -d 2 /N/u/fuyuan/BigRed2/hpchw/lab3/summa"

inner_loop=2
# block_size=(16 32 64 128 256 512 1024)
# block_size=(512 1024 2048 4096)
block_size=(128)
# echo "=============SUMMA:Real_Matrix_size=$real_matrixsize_N==============="
# $my_run_exp1 $real_matrixsize_N $each_process_matrixsize_n $block_size $inner_loop

echo "=============SUMMA:Real_Matrix_size=$real_matrixsize_N==============="
for ((j=0;j<${#block_size[@]};j++)); do
		echo
		echo "=============SUMMA:Matrix_size=$real_matrixsize_N, each_process_matrixsize_n=$each_process_matrixsize_n, block_size=${block_size[j]}==============="
		$my_run_exp1 $real_matrixsize_N $each_process_matrixsize_n ${block_size[j]} $inner_loop
		echo "-----------End SUMMA-------------"
		echo
done


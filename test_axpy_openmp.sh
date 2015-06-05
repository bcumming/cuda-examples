for threads in 1 8
do
    echo ===== $threads threads =====
    echo
    for n in 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
    do
        OMP_NUM_THREADS=$threads aprun -cc none ./axpy_omp $n &> tmp
        t=`grep ^axpy tmp | awk '{print $3}'`
        printf "%5d%12.8f\n" $n $t
    done
done

#for n in 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
for n in 14 18 20 22 24 26
do
    aprun ./axpy_kernel $n &> tmp
    t=`grep ^axpy tmp | awk '{print $3}'`
    printf "%5d%12.8f\n" $n $t
done

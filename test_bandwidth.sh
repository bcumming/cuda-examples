for i in 8 10 12 14 16 18 20 22
do
    echo ====== length = 2^$i
    aprun ./memcopy     $i | grep "H2D BW"
    aprun ./axpy_kernel $i | grep "H2D BW"
done

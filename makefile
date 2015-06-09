FLAGS=-arch=sm_35
FLAGS+=-std=c++11
FLAGS+=-O3

LINK=-lcublas -lcuda

all : axpy_cublas axpy_kernel memcopy memcopy2 memcopy3 blur diffusion2d diffusion2d_mpi blur_twice blur_twice.omp dot

axpy_cublas : axpy_cublas.cu util.h
	nvcc $(FLAGS) axpy_cublas.cu -o axpy_cublas $(LINK)

axpy_kernel : axpy_kernel.cu util.h
	nvcc $(FLAGS) axpy_kernel.cu -o axpy_kernel $(LINK)

axpy_omp : axpy_omp.cpp
	g++ -std=c++11 -fopenmp -O3 axpy_omp.cpp -o axpy_omp

memcopy : memcopy.cu util.h
	nvcc $(FLAGS) memcopy.cu -o memcopy $(LINK)

memcopy2 : memcopy2.cu util.h
	nvcc $(FLAGS) memcopy2.cu -o memcopy2 $(LINK)

memcopy3 : memcopy3.cu util.h
	nvcc $(FLAGS) memcopy3.cu -o memcopy3 $(LINK)

blur : blur.cu util.h
	nvcc $(FLAGS) blur.cu -o blur $(LINK)

diffusion2d : diffusion2d.cu util.h diffusion_kernels.o
	nvcc -c $(FLAGS) diffusion2d.cu $(LINK) -I$(CRAY_MPICH2_DIR)/include
	CC diffusion2d.o diffusion_kernels.o -o diffusion2d

diffusion_kernels.o : diffusion_kernels.cu
	nvcc -c $(FLAGS) diffusion_kernels.cu $(LINK) -I$(CRAY_MPICH2_DIR)/include

diffusion2d_mpi : diffusion2d_mpi.cpp util.h diffusion_kernels.o
	CC -std=c++11 -O3 diffusion2d_mpi.cpp diffusion_kernels.o -o diffusion2d_mpi

blur_twice : blur_twice.cu util.h
	nvcc $(FLAGS) blur_twice.cu -o blur_twice $(LINK)

blur_twice.omp : blur_twice.cpp util.h
	CC -std=c++11 -O3 -fopenmp blur_twice.cpp -o blur_twice.omp $(LINK)

dot : dot.cu util.h
	nvcc $(FLAGS) dot.cu -o dot $(LINK)

test : axpy_cublas axpy_kernel
	aprun axpy_cublas 20
	aprun axpy_kernel 20

clean :
	rm -f axpy_cublas
	rm -f axpy_kernel
	rm -f axpy_omp
	rm -f memcopy
	rm -f memcopy2
	rm -f memcopy3
	rm -f blur
	rm -f diffusion2d
	rm -f diffusion2d_mpi
	rm -f blur_twice

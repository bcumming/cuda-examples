FLAGS=-arch=sm_35
FLAGS+=-std=c++11
FLAGS+=-O3

LINK=-lcublas -lcuda

CUDA_BENCHMARKS= axpy_cublas.cuda axpy_kernel.cuda memcopy.cuda memcopy2.cuda memcopy3.cuda blur.cuda diffusion2d.cuda diffusion2d_mpi.cuda blur_twice.cuda dot.cuda
DIFFUSION_BENCHMARKS= diffusion2d.cuda diffusion2d_mpi.cuda
OPENMP_BENCHMARKS= blur_twice.omp axpy.omp

BENCHMARKS=$(CUDA_BENCHMARKS)

all : $(BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
axpy_cublas.cuda : axpy_cublas.cu util.h
	nvcc $(FLAGS) axpy_cublas.cu -o axpy_cublas.cuda $(LINK)

axpy_kernel.cuda : axpy_kernel.cu util.h
	nvcc $(FLAGS) axpy_kernel.cu -o axpy_kernel.cuda $(LINK)

memcopy.cuda : memcopy.cu util.h
	nvcc $(FLAGS) memcopy.cu -o memcopy.cuda $(LINK)

memcopy2.cuda : memcopy2.cu util.h
	nvcc $(FLAGS) memcopy2.cu -o memcopy2.cuda $(LINK)

memcopy3.cuda : memcopy3.cu util.h
	nvcc $(FLAGS) memcopy3.cu -o memcopy3.cuda $(LINK)

blur.cuda : blur.cu util.h
	nvcc $(FLAGS) blur.cu -o blur.cuda $(LINK)

blur_twice.cuda : blur_twice.cu util.h
	nvcc $(FLAGS) blur_twice.cu -o blur_twice.cuda $(LINK)

dot.cuda : dot.cu util.h
	nvcc $(FLAGS) dot.cu -o dot.cuda $(LINK)

#----------------------------------------
# diffusion2D benchmarks
#----------------------------------------
diffusion2d.cuda : diffusion2d.cu util.h diffusion_kernels.o
	nvcc -c $(FLAGS) diffusion2d.cu $(LINK) -I$(CRAY_MPICH2_DIR)/include
	CC diffusion2d.o diffusion_kernels.o -o diffusion2d.cuda

diffusion_kernels.o : diffusion_kernels.cu
	nvcc -c $(FLAGS) diffusion_kernels.cu $(LINK) -I$(CRAY_MPICH2_DIR)/include

diffusion2d_mpi.cuda : diffusion2d_mpi.cpp util.h diffusion_kernels.o
	CC -std=c++11 -O3 diffusion2d_mpi.cpp diffusion_kernels.o -o diffusion2d_mpi.cuda

#----------------------------------------
# openmp benchmarks
#----------------------------------------
blur_twice.omp : blur_twice.cpp util.h
	CC -std=c++11 -O3 -fopenmp blur_twice.cpp -o blur_twice.omp $(LINK)

axpy.omp : axpy_omp.cpp
	g++ -std=c++11 -fopenmp -O3 axpy_omp.cpp -o axpy.omp

clean :
	rm -f *.cuda
	rm -f *.omp

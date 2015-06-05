#include <iostream>

#include <cuda.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

#define USE_PINNED
// CUDA kernel implementing newton solve for
//      f(x) = 0
// where
//      f(x) = exp(cos(x)) - 2
__global__
void newton(int n, double *x) {
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    auto grid_step = blockDim.x * gridDim.x;

    auto f  = [] (double x) {
        return exp(cos(x))-2;
    };
    auto fp = [] (double x) {
        return -sin(x) * exp(cos(x));
    };

    while(tid<n) {
        auto x0 = x[tid];
        for(int i=0; i<5; ++i) {
            x0 -= f(x0)/fp(x0);
        }
        x[tid] = x0;
        tid += grid_step;
    }
}

int main(int argc, char** argv) {
    size_t pow        = read_arg(argc, argv, 1, 20);
    size_t num_chunks = read_arg(argc, argv, 2, 1);

    size_t N = 1 << pow;
    auto size_in_bytes = N * sizeof(double);

    std::cout << "memory copy overlap test of length N = " << N
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << " with " << num_chunks << " chunks"
              << std::endl;

    cuInit(0);

    double* xd = malloc_device<double>(N);
    double* xh = malloc_host_pinned<double>(N, 1.5);
    double* x  = malloc_host_pinned<double>(N);

    int chunk_size = N/num_chunks; // assume N % num_chunks == 0

    // precompute kernel launch configuration
    auto block_dim = 128ul;
    auto grid_dim = chunk_size/block_dim + (chunk_size%block_dim ? 1 : 0);
    grid_dim = std::min(1024ul, grid_dim);

    CudaStream D2H_stream(true);
    CudaStream H2D_stream(true);
    CudaStream kernel_stream(true);

    auto start_event = D2H_stream.get_event();
    for(int i=0; i<num_chunks; ++i) {
        auto offset = i*chunk_size;

        // copy chunk to device
        copy_to_device_async<double>(xh+offset, xd+offset, chunk_size, H2D_stream.stream());

        // force the kernel stream to wait for the memcpy
        auto H2D_event = H2D_stream.get_event();
        kernel_stream.wait_on_event(H2D_event);

        // y += 2 * x
        newton<<<grid_dim, block_dim, 0, kernel_stream.stream()>>>(chunk_size, xd+offset);
        cuda_check_last_kernel("newton kernel");

        // copy chunk of result back to host
        auto kernel_event = kernel_stream.get_event();
        D2H_stream.wait_on_event(kernel_event);
        copy_to_host_async<double>(xd+offset, x+offset, chunk_size, D2H_stream.stream());
    }
    auto end_event = D2H_stream.get_event();
    end_event.wait();

    auto time_total = end_event.time_since(start_event);

    std::cout << "-------\ntimings\n-------" << std::endl;
    std::cout << "total : " << time_total << std::endl;

    // check for errors
    auto f  = [] (double x) { return exp(cos(x))-2.; };
    auto errors = 0;
    for(auto i=0; i<N; ++i) {
        if(std::fabs(f(x[i]))>1e-10) {
            errors++;
        }
    }
    if(errors>0) std::cout << "\n============ FAILED with " << errors << " errors" << std::endl;
    else         std::cout << "\n============ PASSED" << std::endl;

    cudaFree(xd);
    cudaFreeHost(xh);
    cudaFreeHost(x);

    return 0;
}


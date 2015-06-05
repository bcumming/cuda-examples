#include <iostream>

#include <cuda.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

__global__
void dispersion(int n, double alpha, const double *in, double* out) {
    extern __shared__ double buffer[];

    int block_start = blockDim.x * blockIdx.x;
    int block_end   = block_start + blockDim.x;
    int lid = threadIdx.x;
    int gid = lid + block_start;

    auto laplace = [&alpha] (int pos, double const* field) {
        return alpha*(field[pos-1] -2.0*field[pos] + field[pos+1]);
    };

    if(gid<n) {
        int li = lid+2;
        int gi = gid+2;

        buffer[li] = laplace(gi, in);
        if(blockDim.x==0) {
            buffer[1] = laplace(block_start+1, in);
            buffer[blockDim.x+2] = laplace(block_end+2, in);
        }

        __syncthreads();

        out[gi] = in[gi] + laplace(li, buffer);
    }
}

int main(int argc, char** argv) {
    size_t pow    = read_arg(argc, argv, 1, 20);
    size_t nsteps = read_arg(argc, argv, 2, 100);
    size_t N = 1 << pow;

    auto size_in_bytes = N * sizeof(double);

    std::cout << "dispersion 1D test of length N = " << N
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    // TODO : find an alpha that guarentees convergence, then work backwards from there to get dt
    double dt = 0.01;
    double dx = 0.1;
    double alpha = dt/(dx*dx);

    cuInit(0);

    auto x_host = malloc_host<double>(N+4, 0.);
    // set boundary conditions to 1
    x_host[0]   = 1.0;
    x_host[1]   = 1.0;
    x_host[N+2] = 1.0;
    x_host[N+3] = 1.0;

    auto x0 = malloc_device<double>(N+4);
    auto x1 = malloc_device<double>(N+4);

    // copy initial conditions to device
    copy_to_device<double>(x_host, x0, N+4);

    // find the launch grid configuration
    auto block_dim = 128ul;
    auto grid_dim = N/block_dim + (N%block_dim ? 1 : 0);

    std::cout << "threads per block " << block_dim << ", in " << grid_dim << " blocks" << std::endl;

    CudaStream stream;
    auto start_event = stream.get_event();
    for(auto step=0; step<nsteps; ++step) {
        auto shared_size = sizeof(double)*(block_dim+4);

        dispersion<<<grid_dim, block_dim, shared_size>>>(N, alpha, x0, x1);

        std::swap(x0, x1);
    }
    auto stop_event = stream.get_event();

    // copy result back to host
    copy_to_host<double>(x0, x_host, N+4);

    for(auto i=0; i<10; ++i) std::cout << x_host[i] << " "; std::cout << std::endl;

    stop_event.wait();
    auto time = stop_event.time_since(start_event);
    std::cout << "==== that took " << time << " seconds"
              << " (" << time/nsteps << "s/step)" << std::endl;

    return 0;
}


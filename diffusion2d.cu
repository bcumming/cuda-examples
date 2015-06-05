#include <iostream>

#include <cuda.h>

#include <mpi.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

void fill_gpu(double *v, double value, int n);

__global__
void diffusion_interior(double *x0, double *x1, int nx, int ny, double dt) {
    auto i = threadIdx.x + blockDim.x*blockIdx.x;
    auto j = threadIdx.y + blockDim.y*blockIdx.y;

    if(i<nx && j<ny) {
        auto pos = (j+1)*(nx+2) + i + 1;
        x1[pos] = x0[pos] + dt* (-4.*x0[pos]
                   + x0[pos-nx-2] + x0[pos+nx+2]
                   + x0[pos-1] + x0[pos+1]);
    }
}

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);

    // set domain size
    size_t nx = 64;
    size_t ny = 1 << pow;
    double dt = 0.01;

    // initialize MPI
    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // calculate global domain sizes
    if(ny%mpi_size) {
        std::cout << "error : global domain dimension " << ny
                  << "must be divisible by number of MPI ranks " << mpi_size
                  << std::endl;
        exit(1);
    }
    else if(mpi_rank==0) {
        printf("=== 2D diffusion : %dx%d for %d steps\n", nx, ny, nsteps);
        printf("=== %d MPI ranks : %dx%d\n", mpi_size, nx, ny/mpi_size, nsteps);
    }
    ny /= mpi_size;

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = (nx+2)*(ny+2);
    double *x_host = malloc_host_pinned<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);

    // set initial conditions of 0 everywhere
    fill_gpu(x0, 0., buffer_size);
    fill_gpu(x1, 0., buffer_size);

    // set boundary conditions of 1 on south border
    if(mpi_rank==0) {
        fill_gpu(x0, 1., nx+2);
        fill_gpu(x1, 1., nx+2);
    }

    CudaStream stream;
    CudaStream copy_stream(true);
    auto start_event = stream.get_event();

    // time stepping loop
    for(auto step=0; step<nsteps; ++step) {
        dim3 b_dim(8, 8);
        dim3 g_dim(nx/b_dim.x, ny/b_dim.y);

        diffusion_interior <<<g_dim, b_dim>>> (x0, x1, nx, ny, dt);

        std::swap(x0, x1);
    }
    auto stop_event = stream.get_event();
    stop_event.wait();

    copy_to_host<double>(x0, x_host, buffer_size);

    double time = stop_event.time_since(start_event);

    std::cout << "time to solution : "
              << time << "s, "
              << nsteps*nx*ny / time << " points/second"
              << std::endl;

    MPI_Finalize();

    return 0;
}

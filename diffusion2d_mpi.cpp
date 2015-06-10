#include <cassert>
#include <iostream>
#include <future>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <mpi.h>

#include "util.h"
#include "CudaStream.h"
#include "CudaEvent.h"

void diffusion_interior(double *x0, double *x1, int nx, int ny, double dt, cudaStream_t s=NULL);
void diffusion_boundary(double *x0, double *x1, int nx, double dt, cudaStream_t s=NULL);
void fill_gpu(double *v, double value, int n);

enum class Neighbour {north, south};

bool exchange_boundary( double *x0,
                        double *x1,
                        double *send_buff,
                        double *recv_buff,
                        int nx,
                        int rank,
                        Neighbour neighbour,
                        double dt,
                        bool exchange=false) //, CudaStream const& s)
{
    CudaStream s(true);
    MPI_Request request[2];
    MPI_Status  status[2];

    if(exchange) {
        // determine the MPI rank of the neighbour
        int neigh_rank;
        if(neighbour == Neighbour::south) {
            neigh_rank = rank-1;
        }
        else {
            neigh_rank = rank+1;
        }

        // post receives first
        MPI_Irecv(recv_buff, nx, MPI_DOUBLE, neigh_rank, 1, MPI_COMM_WORLD, &request[0]);

        // copy from device to host
        copy_to_host_async<double>(x0, send_buff, nx, s.stream());
        auto sende = s.get_event();

        // wait buffer to finish copying from device before sending
        sende.wait();
        MPI_Isend(send_buff, nx, MPI_DOUBLE, neigh_rank, 1, MPI_COMM_WORLD, &request[1]);

        // wait for receive to finish
        MPI_Wait(request, status);

        // copy halo received from neighbour to device
        copy_to_device_async<double>(recv_buff, x0, nx, s.stream());
    }

    // enqueue kernel
    int offset = nx + 2;
    if(neighbour == Neighbour::north) {
        offset = -offset;
    }
    diffusion_boundary(x0+offset, x1+offset, nx, dt, s.stream());

    if(exchange) {
        // wait for send to finish
        MPI_Wait(request+1, status+1);
    }

    // the thread waits until the kernel has completed
    s.get_event().wait();

    return true;
}

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    int pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    int nsteps = read_arg(argc, argv, 2, 100);

    // set domain size
    int nx = 128;
    int ny = 1 << pow;
    double dt = 0.01;

    // initialize MPI
    int mpi_rank, mpi_size;

    // intialize MPI with MPI_THREAD_MULTIPLE because the boundary threads
    // call MPI asynchroously
    int level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &level);
    switch(level) {
        case MPI_THREAD_SINGLE     : std::cout << "thread single"   << std::endl; break;
        case MPI_THREAD_FUNNELED   : std::cout << "thread funneled" << std::endl; break;
        case MPI_THREAD_SERIALIZED : std::cout << "thread serialized" << std::endl; break;
        case MPI_THREAD_MULTIPLE   : std::cout << "thread multiple" << std::endl; break;
    }
    assert(level == MPI_THREAD_MULTIPLE);
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
    bool has_south = mpi_rank>0;
    bool has_north = mpi_rank<(mpi_size-1);
    std::cout << "rank " << mpi_rank << " south " << has_south << " north " << has_north << std::endl;

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = (nx+2)*(ny+2);
    double *x_host = malloc_host_pinned<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);

    double *send_north = malloc_host_pinned<double>(nx+2);
    double *send_south = malloc_host_pinned<double>(nx+2);
    double *recv_north = malloc_host_pinned<double>(nx+2);
    double *recv_south = malloc_host_pinned<double>(nx+2);

    // set initial conditions of 0 everywhere
    fill_gpu(x0, 0., buffer_size);
    fill_gpu(x1, 0., buffer_size);

    // set boundary conditions of 1 on south border
    if(mpi_rank==0) {
        fill_gpu(x0, 1., nx+2);
        fill_gpu(x1, 1., nx+2);
    }

    CudaStream south_stream(true);
    CudaStream north_stream(true);
    CudaStream interior_stream(true);
    CudaStream null_stream(false);

    auto start_event = null_stream.get_event();

    // turn on the cuda profiler here
    cuda_check_status( cudaProfilerStart() );

    // time stepping loop
    auto async_policy = std::launch::async;
    //auto async_policy = std::launch::deferred;
    for(auto step=0; step<nsteps; ++step) {
        std::vector<std::future<bool>> results;
        {
            auto neighbour = Neighbour::south;
            auto pos = 2;
            auto future = std::async(
                async_policy,
                exchange_boundary,
                x0+pos, x1+pos, send_south, recv_south, nx, mpi_rank, neighbour, dt, has_south
            );
            results.emplace_back(std::move(future));
        }
        {
            auto neighbour = Neighbour::north;
            auto pos = (ny+1)*(nx+2) + 2;
            auto future = std::async(
                async_policy,
                exchange_boundary,
                x0+pos, x1+pos, send_north, recv_north, nx, mpi_rank, neighbour, dt, has_north
            );
            results.emplace_back(std::move(future));
        }

        diffusion_interior(x0, x1, nx, ny, dt, interior_stream.stream());
        interior_stream.get_event().wait();

        for(auto &r : results) {
            r.get();
        }

        std::swap(x0, x1);
    }
    auto stop_event = interior_stream.get_event();
    stop_event.wait();

    copy_to_host<double>(x0, x_host, buffer_size);
    cudaProfilerStop();

    double time = stop_event.time_since(start_event);

    std::cout << "time to solution : "
              << time << "s, "
              << nsteps*nx*ny / time << " points/second"
              << std::endl;

    MPI_Finalize();

    cudaFree(x0);
    cudaFree(x1);
    cudaFreeHost(x_host);

    cudaFreeHost(send_north);
    cudaFreeHost(send_south);
    cudaFreeHost(send_east);
    cudaFreeHost(send_west);

    return 0;
}

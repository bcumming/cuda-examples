#include <algorithm>

namespace kernels {
    __global__
    void diffusion_interior(double *x0, double *x1, int nx, int ny, double dt) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;
        auto j = threadIdx.y + blockDim.y*blockIdx.y+1;

        if(i<nx && j<ny-1) {
            auto pos = (j+1)*(nx+2) + i + 1;
            x1[pos] = x0[pos] + dt* (-4.*x0[pos]
                       + x0[pos-nx-2] + x0[pos+nx+2]
                       + x0[pos-1] + x0[pos+1]);
        }
    }

    __global__
    void diffusion_boundary(double *x0, double *x1, int nx, double dt) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        if(i<nx) {
            auto pos = 0;
            x1[pos] = x0[pos] + dt* (-4.*x0[pos]
                       + x0[pos-nx-2] + x0[pos+nx+2]
                       + x0[pos-1] + x0[pos+1]);
        }
    }


    template <typename T>
    __global__
    void fill(T *v, T value, int n) {
        int tid  = threadIdx.x + blockDim.x*blockIdx.x;
        int grid_step = blockDim.x * gridDim.x;

        while(tid<n) {
            v[tid] = value;
            tid += grid_step;
        }
    }
}

void diffusion_interior(double *x0, double *x1, int nx, int ny, double dt, cudaStream_t s=NULL) {
    dim3 b_dim(16, 16);
    dim3 g_dim(nx/b_dim.x, (ny-2)/b_dim.y);

    kernels::diffusion_interior <<<g_dim, b_dim, 0, s>>> (x0, x1, nx, ny, dt);
}

void diffusion_boundary(double *x0, double *x1, int nx, double dt, cudaStream_t s=NULL) {
    kernels::diffusion_boundary <<<128, nx/128 + 1, 0, s>>> (x0, x1, nx, dt);
}


template <typename T>
void fill_gpu(T *v, T value, int n) {
    auto block_dim = 128ul;
    auto grid_dim = n/block_dim + (n%block_dim ? 1 : 0);
    grid_dim = std::min(1024ul, grid_dim);

    kernels::fill<T><<<grid_dim, block_dim>>>(v, value, n);
}

void fill_gpu(double *v, double value, int n) {
    fill_gpu<double>(v, value, n);
}


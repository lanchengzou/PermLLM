
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstddef>
#include "ATen/core/Tensor.h"


#define BLOCK_SIZE 64

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



template<typename scalar_t>
__global__ void permutation_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> m_ptr,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> p_ptr,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> result
) {
    // __shared__ scalar_t shared_m[BLOCK_SIZE][BLOCK_SIZE];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m_ptr.size(1)) {
        for(size_t i = 0; i < m_ptr.size(0); ++i) {
            result[i][idx] = m_ptr[i][p_ptr[idx]];
        }
    }
}

template<typename scalar_t>
__global__ void permutation_kernel_2d(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> m_ptr,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> p_ptr,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> result
) {
    // __shared__ scalar_t shared_m[BLOCK_SIZE][BLOCK_SIZE];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < m_ptr.size(0) && idy < m_ptr.size(1)) {
        result[idx][idy] = m_ptr[idx][p_ptr[idy]];
    }
}

torch::Tensor permutation_cuda(
    torch::Tensor m,
    torch::Tensor p
) {
    cudaSetDevice(m.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = BLOCK_SIZE;
    const int blocks = (m.size(1) + threads - 1) / threads;

    auto result = torch::empty({m.size(0), m.size(1)}, 
    torch::dtype(m.dtype()).device(m.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(m.type(), "permutation_cuda", ([&]{
        permutation_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            m.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            p.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        ); 
    }));

    return result;
}


torch::Tensor permutation_cuda_2d(
    torch::Tensor m,
    torch::Tensor p
) {
    cudaSetDevice(m.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block_size_x = 16;
    const int block_size_y = 64;
    const int block_x = (m.size(0) + block_size_x - 1) / block_size_x;
    const int block_y = (m.size(1) + block_size_y - 1) / block_size_y;

    dim3 grids(block_x, block_y);
    dim3 threads(block_size_x, block_size_y);

    auto result = torch::empty({m.size(0), m.size(1)}, 
        torch::dtype(m.dtype()).device(m.device()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(m.type(), "permutation_cuda", ([&]{
        permutation_kernel_2d<scalar_t><<<grids, threads, 0, stream>>>(
            m.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            p.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        ); 
    }));

    return result;
}

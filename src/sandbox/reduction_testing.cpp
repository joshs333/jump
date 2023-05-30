/**
 * @brief Me testing out some reduction kernels to maximize performance.
 * @note There is an amazing slide deck on reductions here:
 *  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *  Absolutely fantastic!!
 */

#include <jump/array.hpp>

template<typename T>
__global__
void warpReduce(jump::array<T> arr, jump::array<T> scratch) {
    extern __shared__ T shared_buffer[];

    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // T mySum = (i < arr.size()) ? g_idata[i] : 0;

    // if (i + blockSize < n)
    //     mySum += g_idata[i+blockSize];

    // sdata[tid] = mySum;
    // __syncthreads();

    // // do reduction in shared mem
    // for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    // {
    //     if (tid < s)
    //     {
    //         sdata[tid] = mySum = mySum + sdata[tid + s];
    //     }

    //     __syncthreads();
    // }
}


int main(int argc, char** argv) {
    jump::array<int> k(64, 5);
    jump::array<int> scratch(k.size(), 5);
    k.to_device();

    const std::size_t threads_per_block = 32;
    std::size_t threads = k.size() / threads_per_block();

    // int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    warpReduce<<<threads, threads_per_block>>>(k, scratch);

    return 0;
}




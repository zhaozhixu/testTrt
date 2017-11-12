#include "tensorCuda.h"
#include "common.h"

__global__ void sliceTensorKernel(float *src, float *dst, int start, int len, int s_len, int block_size, int d_block_num)
{
     int i = threadIdx.x;
     int index = i / len * s_len + i % len + start;
     CHECK(cudaMemcpy(dst + i * block_size, src + index * block_size, block_size*sizeof(float), cudaMemcpyDeviceToDevice));
}

void sliceTensorHost(float *src, float *dst, int start, int len, int s_len, int block_size, int d_block_num)
{
     int cuda_block_num = d_block_num / MAX_THREADS_PER_BLOCK + 1;
     dim3 threads_per_block(MAX_THREADS_PER_BLOCK);
     sliceTensorKernel<<<cuda_block_num, threads_per_block>>>(src, dst, start, len, s_len, block_size, d_block_num);
}

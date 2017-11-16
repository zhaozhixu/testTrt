#include "tensorCuda.h"

__global__ void sliceTensorKernel(float *src, float *dst, int sdim, int ddim, int start, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x;
     dst[di] = src[si];
}

__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = di * dim_size;
     float now = src[si], max = now;
     int maxi = 0;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i];
          if (now > max) {
               max = now;
               maxi = i;
          }
     }
     dst[di] = max;
     arg[di] = maxi;
}

__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     dst[di] = src1[di] * src2[di];
}

#include <cuda_runtime.h>
#include <iostream>
/* #include <helper_functions.h> */
/* #include <helper_cuda.h> */

#define MAX_THREADS_PER_BLOCK 1024
#define CHECK(status)									\
     {                                                  \
          if (status != 0)                              \
          {                                             \
               std::cout << "Cuda failure: " << status; \
               abort();                                 \
          }                                             \
     }

__global__ void sliceTensorKernel(float *src, float *dst, int start, int len, int s_len, int block_size, int d_block_num);
void sliceTensorHost(float *src, float *dst, int start, int len, int s_len, int block_size, int d_block_num);

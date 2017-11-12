#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
/* #include <cuda_runtime.h> */
/* #include <helper_functions.h> */
/* #include <helper_cuda.h> */

#define MAXDIM 8

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

int computeLength(int *dims, int ndim);
Tensor createTensor(float *data, int ndim, int *dims);
int printTensor(Tensor *tensor, const char *fmt);
void sliceTensor(Tensor *src, Tensor *dst, int dim, int start, int len);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensorCuda.h"

#define MAXDIM 8

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

int computeLength(int *dims, int ndim);
Tensor *createTensor(float *data, int ndim, int *dims);
int printTensor(Tensor *tensor, const char *fmt);
Tensor *sliceTensor(Tensor *src, int dim, int start, int len);
Tensor *sliceTensorCuda(Tensor *src, int dim, int start, int len);

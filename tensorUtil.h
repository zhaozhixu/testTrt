#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensorCuda.h"

#define MAXDIM 8

typedef enum CloneKind {
     H2H, H2D, D2D, D2H
} CloneKind;

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

void *cloneMem(void *src, size_t size, const CloneKind kind);
int computeLength(int ndim, int *dims);
Tensor *createTensor(float *data, int ndim, int *dims);
void printTensor(Tensor *tensor, const char *fmt);
Tensor *sliceTensor(Tensor *src, int dim, int start, int len);
void *sliceTensorCuda(Tensor *src, Tensor *dst, int dim, int start, int len);
Tensor *reshapeTensor(Tensor *src, int newNdim, int *newDims);
void reduceArgMax(Tensor *src, Tensor **dst, Tensor **arg, int dim);

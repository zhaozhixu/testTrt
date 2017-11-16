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

int isShapeEqual(int ndim1, const int *dims1, int ndim2, const int *dims2);
void *cloneMem(const void *src, size_t size, CloneKind kind);
int computeLength(int ndim, const int *dims);
Tensor *createTensor(float *data, int ndim, const int *dims);
void printTensor(const Tensor *tensor, const char *fmt);
Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len);
Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len);
Tensor *creatSlicedTensorCuda(const Tensor *src, int dim, int start, int len);
void *sliceTensorCuda(const Tensor *src, Tensor *dst, int dim, int start, int len);
Tensor *reshapeTensor(const Tensor *src, int newNdim, const int *newDims);
Tensor *createReducedTensor(const Tensor *src, int dim);
void *reduceArgMax(const Tensor *src, Tensor *dst, Tensor *arg, int dim);
Tensor *multiplyElement(const Tensor *src1, const Tensor *src2, Tensor *dst);

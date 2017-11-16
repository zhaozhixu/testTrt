#include <time.h>
#include "tensorUtil.h"
#include "tensorCuda.h"

clock_t start, end;
Tensor *t, *tcuda;

int ndim = 3;
int dims[] = {3, 2, 3};
float data[] = {0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0,
                16.0, 17.0};
/* float data[] = {0.0, 2.0, 1.0, */
/*                 5.0, 4.0, 3.0, */
/*                 6.0, 7.0, 6.0, */
/*                 9.0, 10.0, 11.0, */
/*                 12.0, 12.0, 14.0, */
/*                 15.0, 16.0, 16.0}; */


void init()
{
     /* int ndim = 4; */
     /* int dims[] = {20, 24, 78, 72}; */
     /* int dims[] = {20, 24, 1872, 3}; */
     /* size_t data_size = 2695680; */
     /* float *data = (float *)malloc(sizeof(float) * data_size); */
     /* int i; */
     /* for (i = 0; i < data_size; i++) */
     /*      data[i] = 1.0; */

     t = createTensor(data, ndim, dims);
     float *tcuda_data = (float *)cloneMem(t->data, sizeof(float) * t->len, H2D);
     tcuda = createTensor(tcuda_data, t->ndim, t->dims);
     printTensor(t, "%.2f");
}

void testSliceTensor()
{
     /* Tensor *st = createSlicedTensor(t, 2, 2, 1800); */
     Tensor *st = createSlicedTensor(t, 2, 1, 2);
     start = clock();
     /* sliceTensor(t, st, 2, 2, 1800); */
     sliceTensor(t, st, 2, 1, 2);
     end = clock();
     printf("sliceTensor in %ld\n", end - start);
     printTensor(st, "%.2f");

     /* Tensor *stcuda = creatSlicedTensorCuda(tcuda, 2, 2, 1800); */
     Tensor *stcuda = creatSlicedTensorCuda(tcuda, 2, 1, 2);
     start = clock();
     /* sliceTensorCuda(tcuda, stcuda, 2, 2, 1800); */
     sliceTensorCuda(tcuda, stcuda, 2, 1, 2);
     end = clock();
     printf("sliceTensorCuda in %ld\n", end - start);
     float *sthost_data = (float *)cloneMem(stcuda->data, stcuda->len * sizeof(float), D2H);
     Tensor *sthost = createTensor(sthost_data, stcuda->ndim, stcuda->dims);
     printTensor(sthost, "%.2f");
}

void testReshapeTensor()
{
     /* printTensor(t, "%.2f"); */

     /* int newNdim = 3; */
     /* int newDims[] = {3, 3, 2}; */
     int newNdim = 2;
     int newDims[] = {3, 6};
     start = clock();
     Tensor *rt = reshapeTensor(t, newNdim, newDims);
     end = clock();
     printf("reshapeTensor in %ld\n", end - start);
     printTensor(rt, "%.2f");
}

void testReduceArgMax()
{
     /* printTensor(t, "%.2f"); */
     Tensor *dst = createReducedTensor(tcuda, tcuda->ndim-1);
     Tensor *arg = createReducedTensor(tcuda, tcuda->ndim-1);
     start = clock();
     reduceArgMax(tcuda, dst, arg, tcuda->ndim-1);
     end = clock();
     printf("reduceArgMax in %ld\n", end - start);

     float *dst_host_data = (float *)cloneMem(dst->data, sizeof(float) * dst->len, D2H);
     Tensor *dst_host = createTensor(dst_host_data, dst->ndim, dst->dims);
     printTensor(dst_host, "%.2f");
     float *arg_host_data = (float *)cloneMem(arg->data, sizeof(float) * arg->len, D2H);
     Tensor *arg_host = createTensor(arg_host_data, arg->ndim, arg->dims);
     printTensor(arg_host, "%.2f");
}

void testMultiplyElement()
{
     float *dst_cuda_data;
     cudaMalloc(&dst_cuda_data, sizeof(float) * tcuda->len);
     Tensor *dst = createTensor(dst_cuda_data, tcuda->ndim, tcuda->dims);
     Tensor * src1 = createTensor(tcuda->data, tcuda->ndim, tcuda->dims);
     Tensor * src2 = createTensor(tcuda->data, tcuda->ndim, tcuda->dims);

     start = clock();
     multiplyElement(src1, src2, dst);
     end = clock();
     printf("multiplyElement in %ld\n", end - start);

     float *dst_host_data = (float *)cloneMem(dst->data, sizeof(float) * dst->len, D2H);
     Tensor *dst_host = createTensor(dst_host_data, dst->ndim, dst->dims);
     printTensor(dst_host, "%.2f");
}

int main(int argc, char *argv[])
{
     init();
     testSliceTensor();
     testReshapeTensor();
     testReduceArgMax();
     testMultiplyElement();
}

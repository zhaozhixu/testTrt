#include <time.h>
#include "tensorUtil.h"
#include "tensorCuda.h"

clock_t start, end;
Tensor *t;

void init()
{
/* int ndim = 3; */
/* int dims[] = {3, 2, 3}; */
/* float data[] = {0.0, 1.0, 2.0, 3.0, */
/*                 4.0, 5.0, 6.0, 7.0, */
/*                 8.0, 9.0, 10.0, 11.0, */
/*                 12.0, 13.0, 14.0, 15.0, */
/*                 16.0, 17.0}; */
/* float data[] = {0.0, 2.0, 1.0, */
/*                 5.0, 4.0, 3.0, */
/*                 6.0, 7.0, 6.0, */
/*                 9.0, 10.0, 11.0, */
/*                 12.0, 12.0, 14.0, */
/*                 15.0, 16.0, 16.0}; */

     int ndim = 4;
     /* int dims[] = {20, 24, 78, 72}; */
     int dims[] = {20, 24, 1872, 3};
     size_t data_size = 2695680;
     float *data = (float *)malloc(sizeof(float) * data_size);
     int i;
     for (i = 0; i < data_size; i++)
          data[i] = 1.0;
     t = createTensor(data, ndim, dims);
}

void testSliceTensor()
{
     start = clock();
     Tensor *st = sliceTensor(t, 2, 2, 1800);
     end = clock();
     /* printTensor(st, "%.2f"); */
     printf("sliceTensor in %ld\n", end - start);
     float *tcuda_data = (float *)cloneMem(t->data, sizeof(float) * t->len, H2D);
     Tensor *tcuda = createTensor(tcuda_data, t->ndim, t->dims);

     float *stcuda_data;
     cudaMalloc(&stcuda_data, sizeof(float) * t->len);
     Tensor *stcuda = createTensor(stcuda_data, t->ndim, t->dims);
     start = clock();
     sliceTensorCuda(tcuda, stcuda, 2, 2, 1800);
     end = clock();
     float *sthost_data = (float *)cloneMem(stcuda->data, stcuda->len * sizeof(float), D2H);
     Tensor *sthost = createTensor(sthost_data, stcuda->ndim, stcuda->dims);
     /* printTensor(sthost, "%.2f"); */
     printf("sliceTensorCuda in %ld\n", end - start);
}

void testReshapeTensor()
{
     /* printTensor(t, "%.2f"); */

     int newNdim = 3;
     int newDims[] = {3, 3, 2};
     start = clock();
     Tensor *rt = reshapeTensor(t, newNdim, newDims);
     end = clock();
     printf("reshapeTensor in %ld\n", end - start);
     /* printTensor(rt, "%.2f"); */
}

void testReduceArgMax()
{
     /* printTensor(t, "%.2f"); */
     float *tcuda_data;
     cudaMalloc(&tcuda_data, sizeof(float) * t->len);
     cudaMemcpy(tcuda_data, t->data, sizeof(float) * t->len, cudaMemcpyHostToDevice);
     Tensor *tcuda = createTensor(tcuda_data, t->ndim, t->dims);

     Tensor *dst = NULL, *arg = NULL;
     start = clock();
     reduceArgMax(tcuda, &dst, &arg, tcuda->ndim-1);
     end = clock();
     printf("reduceArgMax in %ld\n", end - start);

     float *dst_host_data = (float *)malloc(sizeof(float) * dst->len);
     cudaMemcpy(dst_host_data, dst->data, sizeof(float) * dst->len, cudaMemcpyDeviceToHost);
     Tensor *dst_host = createTensor(dst_host_data, dst->ndim, dst->dims);
     /* printTensor(dst_host, "%.2f"); */
     float *arg_host_data = (float *)malloc(sizeof(float) * arg->len);
     cudaMemcpy(arg_host_data, arg->data, sizeof(float) * arg->len, cudaMemcpyDeviceToHost);
     Tensor *arg_host = createTensor(arg_host_data, arg->ndim, arg->dims);
     /* printTensor(arg_host, "%.2f"); */
}

int main(int argc, char *argv[])
{
     init();
     testSliceTensor();
     /* testReduceArgMax(); */
}

#include <time.h>
#include "tensorUtil.h"
#include "tensorCuda.h"

int main(int argc, char *argv[])
{
     clock_t start, end;
     size_t data_size = 2695680;
     int i, ndim = 4;
     float *data = (float *)malloc(sizeof(float) * data_size);
     for (i = 0; i < data_size; i++)
          data[i] = 1.0;

     int dims[] = {20, 24, 78, 72};
     Tensor *t = createTensor(data, ndim, dims);
     /* printTensor(t, "%.2f"); */

     start = clock();
     Tensor *st = sliceTensor(t, 3, 36, 36);
     end = clock();
     /* printTensor(st, "%.2f"); */
     printf("sliceTensor in %ld\n", end - start);

     float *tcuda_data;
     cudaMalloc(&tcuda_data, sizeof(float) * t->len);
     cudaMemcpy(tcuda_data, t->data, sizeof(float) * t->len, cudaMemcpyHostToDevice);
     Tensor *tcuda = createTensor(tcuda_data, ndim, dims);

     start = clock();
     Tensor *stcuda = sliceTensorCuda(tcuda, 3, 36, 36);
     end = clock();
     float *sthost_data = (float *)malloc(stcuda->len * sizeof(float));
     cudaMemcpy(sthost_data, stcuda->data, sizeof(float) * stcuda->len, cudaMemcpyDeviceToHost);
     Tensor *sthost = createTensor(sthost_data, stcuda->ndim, stcuda->dims);
     /* printTensor(sthost, "%.2f"); */
     printf("sliceTensorCuda in %ld\n", end - start);

     /* start = clock(); */
     /* Tensor *stcuda2 = sliceTensorCuda2(tcuda, 3, 0, 27); */
     /* end = clock(); */
     /* float *sthost_data2 = (float *)malloc(stcuda2->len * sizeof(float)); */
     /* cudaMemcpy(sthost_data2, stcuda2->data, sizeof(float) * stcuda2->len, cudaMemcpyDeviceToHost); */
     /* Tensor *sthost2 = createTensor(sthost_data2, stcuda2->ndim, stcuda2->dims); */
     /* /\* printTensor(sthost2, "%.2f"); *\/ */
     /* printf("sliceTensorCuda2 in %ld\n", end - start); */
}

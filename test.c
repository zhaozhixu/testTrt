#include "tensorUtil.h"

int main(int argc, char *argv[])
{
     float data[] = {0.0, 1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0,
                     8.0, 9.0, 10.0, 11.0,
                     12.0, 13.0, 14.0, 15.0,
                     16.0, 17.0};
     int dims[] = {2, 3, 3};
     Tensor t = createTensor(data, 3, dims);
     printTensor(&t, "%.2f");

     int sdims[] = {2, 2, 3};
     float *sdata = (float *)malloc(sizeof(float) * 12);
     Tensor st = createTensor(sdata, 3, sdims);
     sliceTensor(&t, &st, 1, 1, 2);
     printTensor(&st, "%.2f");
}

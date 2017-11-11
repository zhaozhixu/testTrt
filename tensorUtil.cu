#include "tensorUtil.h"

static int assertTensor(Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->dims, tensor->ndim));
}

int computeLength(int *dims, int ndim)
{
     assert(dims);
     int i, len = 1;
     for (i = 0; i < ndim; i++)
          len *= dims[i];
     return len;
}

Tensor createTensor(float *data, int ndim, int *dims)
{
     Tensor t;
     t.data = data;
     t.ndim = ndim;
     t.dims = dims;
     t.len = computeLength(dims, ndim);
     return t;
}

int printTensor(Tensor *tensor)
{
     assertTensor(tensor);
     int i, j, dim_sizes[MAXDIM];
     int ndim = tensor->ndim, len = tensor->len, *dims = tensor->dims;
     float *data = tensor->data;

     dim_sizes[ndim-1] = 1;
     if (ndim > 1)
          for (i = ndim-2; i >= 0; i--)
               dim_sizes[i] = dims[i+1] * dim_sizes[i+1];
     for (i = 0; i < len; i++) {
          for (j = 0; j < ndim; j++) {
               if (data[i] % dim_sizes[j] == 0)
          }
          printf("%e ", data[i]);
     }
}

void sliceTensor(Tensor *src, Tensor *dst, int dim, int start, int len)
{
     assertTensor(src);
     assert(dim <= MAXDIM);
     assert(len+start <= src->dims[dim]);

     dst = (Tensor *)malloc(sizeof(tensor));
     dst->len = src->len / src->dims[dim] * len;
     dst->data = (float *)malloc(dst->len * sizeof(float));

     int i, block_size = 1, block_num = 1;
     for (i = dim+1; i < src->ndim-1; i++)
          block_size *= src->dims[i];
     for (i = 0; i <= dim; i++)
          block_num *= src->dims[i];

     int skip_len = src->dims[dim] - len;
     float *dstp = dst->data, *srcp = src->data;
     for (i = 0; i < block_num; i++, srcp += block_size) {
          if (i % src->dims[dim] < skip_len)
               continue;
          memmove(dstp, srcp, block_size);
          dstp += block_size;
     }
}

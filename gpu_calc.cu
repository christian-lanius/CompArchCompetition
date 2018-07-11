#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

__global__ void device_matmul()
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!
}

__host__ void launch_kernel(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst)
{

  //This function launches the gpu-kernel (a kind of function).
  //Please modify this function for convolutional calculation.
  //You need to allocate the device memory and so on in this function.


  ////////// initialization //////////
  
  double **tmpmat = (double**) malloc(sizeof(double*) * (num+2));
  for (int i=0; i<num+2; i++)  {
    tmpmat[i] = (double*)malloc(sizeof(double) * (num+2));
  }
  for (int i=0; i<num+2; i++)  {
    tmpmat[0][i] = 0.0f;
    tmpmat[num+1][i] = 0.0f;
  }



  for (int i=1; i<=num; i++)  {
    tmpmat[i][0] = 0.0f;
    for (int j=1; j<=num; j++) {
      tmpmat[i][j] = gpu_mat[(i-1)*num + (j-1)];
    }
    tmpmat[i][num+1] = 0.0f;
  }

  double *d_mat, *d_kernel, *d_res;
  cudaMalloc((void **)&d_mat, sizeof(double) * (num+2)* (num+2));
  cudaMalloc((void **)&d_kernel, sizeof(double) * 3);
  cudaMalloc((void **)&d_res, sizeof(double) * (num+2)* (num+2));

  int width = num+2;
  for (int i=0; i<num+2; i++)  {
    d_mat[0*width + i] = 0.0f;
    d_mat[(num+1)*width + i] = 0.0f;
  }

  for (int i=1; i<=num; i++)  {
    // tmpmat[i][0] = 0.0f;
    d_mat[(i-1)*width + 0] = 0.0f;
    for (int j=1; j<=num; j++) {
      d_mat[(i-1)*width + j]  = gpu_mat[(i-1)*num + (j-1)];
    }
    d_mat[(i-1)*width + num+1] = 0.0f;
  }
  for (int i=0; i<num+2; i++)  {
    tmpmat[i] = (double*)malloc(sizeof(double) * (num+2));
  }
  for (int i=0; i<num+2; i++)  {
    tmpmat[0][i] = 0.0f;
    tmpmat[num+1][i] = 0.0f;
  }

  ////////////////////////////////////

  for (int i=1; i<=num; i++) {
    for (int j=1; j<=num; j++) {
      double tmpsum = 0.0f;
      for (int ky=0; ky<3; ky++) 
      for (int kx=0; kx<3; kx++)
        tmpsum += gpu_convkernel[ ky*3 + kx] * tmpmat[i-1 + ky][j-1 + kx];
        
      gpu_matDst[ (i-1)*num + j-1 ] = tmpsum;
    }
  }


  // ------free------ // 
  for (int i=0; i<num+2; i++)  {
    free(tmpmat[i]);
  }
  free(tmpmat);

  

}




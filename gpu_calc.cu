#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"
#include <unistd.h>

void initKernel(double *gpu_kernel, double *gpu_convkernel) __attribute__ ((always_inline));


__global__ void device_matmul( int num, double *gpu_in, double *gpu_kernel, double *gpu_out)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!

  int x;
  int y;
  x = threadIdx.x;
  y = blockIdx.x;

  double tmpsum = 0.0f;
  for (int ky=0; ky<3; ky++) 
    for (int kx=0; kx<3; kx++)
      tmpsum += gpu_kernel[ ky*3 + kx] * gpu_in[(y + ky)*num + (x + kx)];
      
      gpu_out[ y*num + x ] = tmpsum;

}

__host__ void launch_kernel(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst)
{

  //This function launches the gpu-kernel (a kind of function).
  //Please modify this function for convolutional calculation.
  //You need to allocate the device memory and so on in this function.


  ////////// initialization //////////

  double *gpu_in;
  double *gpu_out;
  double *gpu_kernel;
  
  initKernel(gpu_kernel, gpu_convkernel);
  cudaMalloc((void **) &gpu_in, sizeof(double) * (num+2) * (num+2));
  cudaMalloc((void **) &gpu_out, sizeof(double) * num * num);
  cudaMemset(gpu_in, 0, sizeof(double) * (num+2)* (num+2));
  //cudaMemsetAsync(&gpu_in[(num+1)*(num+2)], 0, sizeof(double) * (num+2));
  


  double **tmpmat = (double**) malloc(sizeof(double*) * (num+2));
  double *tmpArray = (double *) malloc(sizeof(double) * (num+2) * (num+2));
  for (int i=0; i<num+2; i++)  {
    tmpmat[i] = &tmpArray[i*(num+2)];
  }
  
  memset(tmpmat[0], 0, sizeof(double) * (num+2));
  memset(tmpmat[num+1], 0, sizeof(double) * (num+2));
  for (int i=1; i<=num; i++)  {
    tmpmat[i][0] = 0.0f;
    memcpy( &(tmpmat[i][1]), &gpu_mat[(i-1)*num], sizeof(double)*num);
    tmpmat[i][num+1] = 0.0f;
    cudaMemcpyAsync(&gpu_in[i*(num+2)+1], &gpu_mat[(i-1)*num], sizeof(double)*(num), cudaMemcpyHostToDevice);
  }
  
  ////////////////////////////////////
  return;
  for (int i=1; i<=num; i++) {
    for (int j=1; j<=num; j++) {
      double tmpsum = 0.0f;
      for (int ky=0; ky<3; ky++) 
      for (int kx=0; kx<3; kx++)
        tmpsum += gpu_convkernel[ ky*3 + kx] * tmpmat[i-1 + ky][j-1 + kx];
        
      gpu_matDst[ (i-1)*num + j-1 ] = tmpsum;
    }
  }
  /*
  device_matmul<<<num,num>>>(num, gpu_in, gpu_kernel, gpu_out);
  cudaMemcpy(gpu_matDst, gpu_out, sizeof(double) * num * num, cudaMemcpyDeviceToHost);
  */
  
  // ------free------ // 
  free(tmpArray);
  free(tmpmat);
  
  

}

void initKernel(double *gpu_kernel, double *gpu_convkernel){
  cudaMalloc((void **) &gpu_kernel, sizeof(double) * 9);
  cudaMemcpyAsync(gpu_kernel, gpu_convkernel, sizeof(double) * 9, cudaMemcpyHostToDevice);
}




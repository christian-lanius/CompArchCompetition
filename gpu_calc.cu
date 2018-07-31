#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"
#include <unistd.h>
#include <cuda_profiler_api.h>


__global__ void device_matmul( int num, double *gpu_in, double *gpu_kernel, double *gpu_out)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!
  int x;
  int y;
  x = threadIdx.x;
  y = 2*blockIdx.x;


  extern __shared__ double s[];
  s[0*(num+2) + x] = gpu_in[(y + 0)*(num+2) + x];
  s[1*(num+2) + x] = gpu_in[(y + 1)*(num+2) + x];
  s[2*(num+2) + x] = gpu_in[(y + 2)*(num+2) + x];
  s[3*(num+2) + x] = gpu_in[(y + 3)*(num+2) + x];
  
  if(x >= num - 2){
    s[0*(num+2) + x+2] = gpu_in[(y + 0)*(num+2) + x+2];
    s[1*(num+2) + x+2] = gpu_in[(y + 1)*(num+2) + x+2];
    s[2*(num+2) + x+2] = gpu_in[(y + 2)*(num+2) + x+2];
    s[3*(num+2) + x+2] = gpu_in[(y + 3)*(num+2) + x+2];
    
  }
  __syncthreads();
  
  for(int offset=0;offset<2;offset++){
    double tmpsum = 0.0f;
    #pragma unroll
    for (int ky=0; ky<3; ky++){ 
      #pragma unroll
      for (int kx=0; kx<3; kx++){
        tmpsum += gpu_kernel[ ky*3 + kx] * s[(ky+offset)*(num+2) + (x + kx)];
      }
    }
    //printf("(%d|%d)\n", x,y+offset);
    gpu_out[ (y+offset)*num + x ] = tmpsum;
  }

}

__host__ void launch_kernel(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst)
{

  //This function launches the gpu-kernel (a kind of function).
  //Please modify this function for convolutional calculation.
  //You need to allocate the device memory and so on in this function.

  ////////// initialization //////////
  cudaProfilerStart();
  double *gpu_in;
  double *gpu_out;
  double *gpu_kernel;
  
  //Kernel initalization
  cudaMalloc((void **) &gpu_kernel, sizeof(double) * 3*3);
  cudaMemcpyAsync(gpu_kernel, gpu_convkernel, sizeof(double) * 3*3, cudaMemcpyHostToDevice);
  //Input and Output Initalization
  cudaMalloc((void **) &gpu_in, sizeof(double) * (num+2) * (num+2));
  cudaMalloc((void **) &gpu_out, sizeof(double) * num * num);
  cudaMemset(gpu_in, 0, sizeof(double) * (num+2)* (num+2));
  //double *tmpmat;
  //tmpmat = (double *) malloc(sizeof(double)*(num+2)*num);
  for (int i=1; i<=num; i++)  {
    //tmpmat[(i-1)*(num+2) + 0] = 0.0f;
    //tmpmat[(i-1)*(num+2) + num+1] = 0.0f;
    //memcpy(&tmpmat[(i-1)*(num+2) + 1], &gpu_mat[(i-1)*num], sizeof(double)*(num));
    cudaMemcpyAsync(&gpu_in[i*(num+2)+1], &gpu_mat[(i-1)*num], sizeof(double)*(num), cudaMemcpyHostToDevice);
  }
  //cudaMemcpy(&gpu_in[(num+2)], tmpmat, sizeof(double)*num*(num+2), cudaMemcpyHostToDevice);
  
  ////////////////////////////////////
  device_matmul<<<512,1024, 4*(num+2)*sizeof(double)>>>(num, gpu_in, gpu_kernel, gpu_out);
  cudaMemcpy(gpu_matDst, gpu_out, sizeof(double) * num * num, cudaMemcpyDeviceToHost);
  cudaProfilerStop();
  
  // ------free------ // 
  //free(tmpArray);
  //free(tmpmat);
  
  

}






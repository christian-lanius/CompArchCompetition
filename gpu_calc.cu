#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"
#include <unistd.h>
#include <cuda_profiler_api.h>

#define NUM_ROWS 4
__global__ void device_matmul( int num, double *gpu_in, double *gpu_kernel, double *gpu_out)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!

  int x;
  int y;
  x = threadIdx.x;
  y = NUM_ROWS*blockIdx.x;


  extern __shared__ double s[];

  #pragma unroll
  for(int offset=0; offset<(2+NUM_ROWS);offset++){
    s[offset*(num+2) + x] = gpu_in[(y + offset)*(num+2) + x];  
  }

  if(x >= num - 2){
    #pragma unroll
    for(int offset=0; offset<(2+NUM_ROWS);offset++){
      s[offset*(num+2) + x+2] = gpu_in[(y + offset)*(num+2) + x+2];  
    }
    
  }
  __syncthreads();
  
  for(int offset=0;offset<NUM_ROWS;offset++){
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
  cudaMalloc((void **) &gpu_in, sizeof(double) * (num+2) * (num+2));
  cudaMemset(gpu_in, 0, sizeof(double) * (num+2)* (num+2));
  for (int i=1; i<=num; i++)  {
    cudaMemcpyAsync(&gpu_in[i*(num+2)+1], &gpu_mat[(i-1)*num], sizeof(double)*(num), cudaMemcpyHostToDevice);
  }
  
  //Kernel initalization
  cudaMalloc((void **) &gpu_kernel, sizeof(double) * 3*3);
  cudaMemcpyAsync(gpu_kernel, gpu_convkernel, sizeof(double) * 3*3, cudaMemcpyHostToDevice);
  //Input and Output Initalization
  
  cudaMalloc((void **) &gpu_out, sizeof(double) * num * num);
  
  
  ////////////////////////////////////
  device_matmul<<<1024/NUM_ROWS,1024, (2+NUM_ROWS)*(num+2)*sizeof(double)>>>(num, gpu_in, gpu_kernel, gpu_out);
  cudaMemcpy(gpu_matDst, gpu_out, sizeof(double) * num * num, cudaMemcpyDeviceToHost);
  
  
  // ------free------ // 
  return;
  cudaFree(gpu_in);
  cudaFree(gpu_kernel);
  cudaFree(gpu_out);
  //free(tmpmat);
  cudaProfilerStop();
  

}






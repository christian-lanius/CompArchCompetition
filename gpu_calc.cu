#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"
#include <unistd.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>

#define NUM_ROWS 2
#define NUM_STREAMS 8

#define SHARED_MEM_SIZE 48*1024 //48 kByte


__global__ void device_matmul( int num, int stream_offset, double *gpu_in, double *gpu_kernel, double *gpu_out)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!

  int x;
  int y;
  
  x = threadIdx.x;
  y = NUM_ROWS*blockIdx.x+stream_offset;


  extern __shared__ double s[];
  double *gpu_kernel_shared;
  gpu_kernel_shared = &s[(2+NUM_ROWS)*num];
  reinterpret_cast<double4*>(s)[x] = reinterpret_cast<double4*>(gpu_in)[y*num/4 + x];

  if(x<9){
    gpu_kernel_shared[x] = gpu_kernel[x];
  }
  
  __syncthreads();
  
  #pragma unroll
  for(int offset=0;offset<NUM_ROWS;++offset){
    double tmpsum = 0.0f;
    #pragma unroll
    for (int ky=0; ky<3; ++ky){
      int in_y = (ky+offset)*(num);
      int ker_y = ky*3;
      #pragma unroll
      for (int kx=0; kx<3; ++kx){
        int in_x = x+kx;
        if( in_x != 0 && in_x != num+1)
          tmpsum += gpu_kernel_shared[ ker_y + kx] * s[in_y+ (in_x-1)];
      }
    }
    gpu_out[ (y+offset)*num + x ] = tmpsum;
  }

}

__host__ void launch_kernel(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst)
{

  //This function launches the gpu-kernel (a kind of function).
  //Please modify this function for convolutional calculation.
  //You need to allocate the device memory and so on in this function.

  ////////// initialization //////////
  //cudaProfilerStart();
  //cudaHostRegister(gpu_matDst, sizeof(double)*num*num, cudaHostRegisterMapped);
  //cudaHostRegister(gpu_mat, sizeof(double)*num*num, cudaHostRegisterMapped);
  double *gpu_in;
  double *gpu_out;
  double *gpu_kernel;
  cudaStream_t streams[NUM_STREAMS];
  for(int stream_idx=0;stream_idx<NUM_STREAMS;stream_idx++){
    cudaStreamCreate(&streams[stream_idx]);
  }
  cudaMalloc((void **) &gpu_in, sizeof(double) * (num+2) * (num));
  cudaMemset(gpu_in, 0, sizeof(double) * (num+2)* (num));
  
  
  //Kernel initalization
  cudaMalloc((void **) &gpu_kernel, sizeof(double) * 3*3);
  cudaMemcpyAsync(gpu_kernel, gpu_convkernel, sizeof(double) * 3*3, cudaMemcpyHostToDevice, streams[1]);
  //Input and Output Initalization
  
  cudaMalloc((void **) &gpu_out, sizeof(double) * num * num);
  
  
  ////////////////////////////////////
  
  
  
  for(int stream_idx=0;stream_idx<NUM_STREAMS;stream_idx++){
    //if(stream_idx != 1) continue;
    int offset = stream_idx*(num)*num/NUM_STREAMS;
    if(stream_idx == 0){//First line copy is offset by 1 (because of zero padding), thus copy one line less
      cudaMemcpyAsync(&gpu_in[num+offset], &gpu_mat[offset], sizeof(double)*num*(num/NUM_STREAMS+1), cudaMemcpyHostToDevice, streams[stream_idx]);
    }else if(stream_idx == NUM_STREAMS-1){ //Last line is one line less copy because of zero padding
      cudaMemcpyAsync(&gpu_in[offset], &gpu_mat[offset-num], sizeof(double)*num*(num/NUM_STREAMS+1), cudaMemcpyHostToDevice, streams[stream_idx]);
    }else{ //copy one line before and one after as well as actual data
      cudaMemcpyAsync(&gpu_in[offset], &gpu_mat[offset-num], sizeof(double)*num*(num/NUM_STREAMS+2), cudaMemcpyHostToDevice, streams[stream_idx]);
    }
    device_matmul<<<num/NUM_ROWS/NUM_STREAMS,num, sizeof(double)*((3+NUM_ROWS)*num+9), streams[stream_idx]>>>(num, stream_idx*num/NUM_STREAMS, gpu_in, gpu_kernel, gpu_out);
    cudaMemcpyAsync(&gpu_matDst[offset], &gpu_out[offset], sizeof(double) * num * num/NUM_STREAMS, cudaMemcpyDeviceToHost, streams[stream_idx]);
    
  }

  return;
  // ------free------ //
  //Dont have to free memory as this is the last cuda call and the memory will be free'd automatically at the end of the program 
  //Dont do this in real life, its a memory leak
  cudaFree(gpu_in);
  cudaFree(gpu_kernel);
  cudaFree(gpu_out);


  

}






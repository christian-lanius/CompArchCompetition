#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"
#include <unistd.h>


__global__ void device_matmul( int num, double *gpu_in, double *gpu_kernel, double *gpu_out)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!

  int x;
  int y;
  x = threadIdx.x;
  y = blockIdx.x;

  extern __shared__ double s[];
  s[0*(num+2) + x] = gpu_in[(y + 0)*(num+2) + x];
  s[1*(num+2) + x] = gpu_in[(y + 1)*(num+2) + x];
  s[2*(num+2) + x] = gpu_in[(y + 2)*(num+2) + x];
  
  if(x >= num - 2){
    //s[0*(num+2) + x+1] = gpu_in[(y + 0)*(num+2) + x+1]; 
    //s[1*(num+2) + x+1] = gpu_in[(y + 1)*(num+2) + x+1]; 
    //s[2*(num+2) + x+1] = gpu_in[(y + 2)*(num+2) + x+1];  
    
    s[0*(num+2) + x+2] = gpu_in[(y + 0)*(num+2) + x+2];
    s[1*(num+2) + x+2] = gpu_in[(y + 1)*(num+2) + x+2];
    s[2*(num+2) + x+2] = gpu_in[(y + 2)*(num+2) + x+2];
    
  }
  __syncthreads();

  double tmpsum = 0.0f;
  #pragma unroll
  for (int ky=0; ky<3; ky++){ 
    #pragma unroll
    for (int kx=0; kx<3; kx++){ 
      //tmpsum += gpu_kernel[ ky*3 + kx] * gpu_in[(y + ky)*(num+2) + (x + kx)];
      tmpsum += gpu_kernel[ ky*3 + kx] * s[ky*(num+2) + (x + kx)];
    }
  }
  //printf("(%d|%d)\n", x,y);
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
  
  //Kernel initalization
  cudaMalloc((void **) &gpu_kernel, sizeof(double) * 9);
  cudaMemcpyAsync(gpu_kernel, gpu_convkernel, sizeof(double) * 9, cudaMemcpyHostToDevice);
  //Input and Output Initalization
  cudaMalloc((void **) &gpu_in, sizeof(double) * (num+2) * (num+2));
  cudaMalloc((void **) &gpu_out, sizeof(double) * num * num);
  cudaMemset(gpu_in, 0, sizeof(double) * (num+2)* (num+2));
  /*
  double **tmpmat = (double**) malloc(sizeof(double*) * (num+2));
  double *tmpArray = (double *) malloc(sizeof(double) * (num+2) * (num+2));
  double *cpu_matDst = (double *) malloc(sizeof(double) * num * num);
  for (int i=0; i<num+2; i++)  {
    tmpmat[i] = &tmpArray[i*(num+2)];
  }
  
  memset(tmpmat[0], 0, sizeof(double) * (num+2));
  memset(tmpmat[num+1], 0, sizeof(double) * (num+2));
  */
  for (int i=1; i<=num; i++)  {
    //tmpmat[i][0] = 0.0f;
    //memcpy( &(tmpmat[i][1]), &gpu_mat[(i-1)*num], sizeof(double)*num);
    //tmpmat[i][num+1] = 0.0f;
    cudaMemcpyAsync(&gpu_in[i*(num+2)+1], &gpu_mat[(i-1)*num], sizeof(double)*(num), cudaMemcpyHostToDevice);
  }
  
  ////////////////////////////////////
  /*
  for (int i=1; i<=num; i++) {
    for (int j=1; j<=num; j++) {
      double tmpsum = 0.0f;
      for (int ky=0; ky<3; ky++) 
      for (int kx=0; kx<3; kx++)
        tmpsum += gpu_convkernel[ ky*3 + kx] * tmpmat[i-1 + ky][j-1 + kx];
        
      cpu_matDst[ (i-1)*num + j-1 ] = tmpsum;
    }
  }*/
  
  device_matmul<<<num,num, 3*(num+2)*sizeof(double)>>>(num, gpu_in, gpu_kernel, gpu_out);
  cudaMemcpy(gpu_matDst, gpu_out, sizeof(double) * num * num, cudaMemcpyDeviceToHost);
  /*
  for (int x=0; x<num; x++) {
    for (int y=0; y<num; y++) {
      double eps = 10e-9;
      if(abs(cpu_matDst[ y*num + x ] - gpu_matDst[ y*num + x ])> eps){
        printf("(%d|%d): %f|%f\n",x,y,cpu_matDst[ y*num + x ],gpu_matDst[ y*num + x ]);
        
      }
    }
  }*/
  
  
  // ------free------ // 
  //free(tmpArray);
  //free(tmpmat);
  
  

}






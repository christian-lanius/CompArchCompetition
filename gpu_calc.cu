#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

__global__ void device_matmul(double *d_mat, double *d_kernel, double *d_res)
{
  //This kernel calculates convolution GPU.
  //Please modify this kernel!!
  int x = threadIdx.x;
  int y = blockIdx.x;
  int num = 1024;
  int width = num+2;
  double tmp_sum = 0.0f;
  for (int ky=-1; ky<2; ky++) 
      for (int kx=-1; kx<2; kx++)
        tmp_sum += d_kernel[ (ky+1)*3 + kx+1] * d_mat[(y+ky)*width + x + kx];
      
  d_res[y*width + x] = tmp_sum;
      
}

__host__ void launch_kernel(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst)
{

  //This function launches the gpu-kernel (a kind of function).
  //Please modify this function for convolutional calculation.
  //You need to allocate the device memory and so on in this function.


  ////////// initialization //////////
  int width = num+2;
  double **tmpmat_2 = (double**) malloc(sizeof(double*) * (num+2));
  for (int i=0; i<num+2; i++)  {
    tmpmat_2[i] = (double*)malloc(sizeof(double) * (num+2));
  }
  for (int i=0; i<num+2; i++)  {
    tmpmat_2[0][i] = 0.0f;
    tmpmat_2[num+1][i] = 0.0f;
  }
  
  for (int i=1; i<=num; i++)  {
    tmpmat_2[i][0] = 0.0f;
    for (int j=1; j<=num; j++) {
      tmpmat_2[i][j] = gpu_mat[(i-1)*num + (j-1)];
    }
    tmpmat_2[i][num+1] = 0.0f;
  }
  
  
  
  
  
  double *tmpmat = (double *)malloc(sizeof(double) * width*width);
  for (int i=1; i<=num; i++)  {
    tmpmat[i*width + 0] = 0.0f;
    for (int j=1; j<=num; j++) {
      tmpmat[i*width + j]  = gpu_mat[(i-1)*num + (j-1)];
    }
    tmpmat[i*width + num+1] = 0.0f;
  }
  for (int i=0; i<num+2; i++){
    //printf("%d | %d | MAX: %d\n", i, (num+1)*width + i, (num+2)*(num+2));
    tmpmat[0*width + i] = 0.0f;
    tmpmat[(num+1)*width + i] = 0.0f;
  }
  
  
  double *d_mat, *d_kernel, *d_res;
  cudaMalloc((void **)&d_mat, sizeof(double) * (num+2)* (num+2));
  cudaMalloc((void **)&d_kernel, sizeof(double) * 9);
  cudaMalloc((void **)&d_res, sizeof(double) * num*num);
  
  cudaMemcpy(d_mat, tmpmat, sizeof(double) * width*width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, gpu_convkernel, sizeof(double) * 9, cudaMemcpyHostToDevice);

  for( int x=0; x<num; x++)
  for( int y=0; y<num; y++){
    double tmp_sum = 0.0f;
    double tmp_sum_2 = 0.0f;
    for (int ky=0; ky<3; ky++) 
        for (int kx=0; kx<3; kx++){
            tmp_sum += gpu_convkernel[ ky*3 + kx] * tmpmat[(y+ky)*width + x+kx];
            //tmp_sum_2 += gpu_convkernel[ ky*3 + kx] * tmpmat_2[y + ky][x + kx];
        }
    //gpu_matDst[y*width + x] = tmp_sum;
    gpu_matDst[ y*num + x ] = tmp_sum;
  }

  //device_matmul<<<1024, 1024>>>(d_mat, d_kernel, d_res);
  //cudaMemcpy(gpu_matDst, d_res, sizeof(double) * width*width, cudaMemcpyDeviceToHost);

  
  
  ////////////////////////////////////
  /*
  for (int i=1; i<=num; i++) {
    for (int j=1; j<=num; j++) {
      double tmpsum = 0.0f;
      for (int ky=0; ky<3; ky++) 
      for (int kx=0; kx<3; kx++)
        tmpsum += gpu_convkernel[ ky*3 + kx] * tmpmat_2[i-1 + ky][j-1 + kx];
        
      gpu_matDst[ (i-1)*num + j-1 ] = tmpsum;
    }
  }
 */

  // ------free------ // 
  for (int i=0; i<num+2; i++)  {
    free(tmpmat_2[i]);
  }
  free(tmpmat);
  free(tmpmat_2);
  cudaFree(d_mat);
  cudaFree(d_kernel);
  cudaFree(d_res);
  

}




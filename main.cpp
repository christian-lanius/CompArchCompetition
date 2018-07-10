/**************************************/
/* You don't have to change this code */
/* Please modify the "gpu_calc.cu"    */
/**************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <malloc.h>
#include <time.h>
#include "toolkit.h"
#include "calculation.h"
#include "timer.h"


void initialize_mat(int num, double* gpu_mat, double* gpu_matDst, double* cpu_mat, double* cpu_matDst)
{
  for (int i=0; i<num; i++) {
    for (int j=0; j<num; j++) {
      cpu_mat[i*num + j] = gpu_mat[i*num + j] =  ((double)rand() / ((double)RAND_MAX + 1));
      cpu_matDst[i*num + j] = gpu_matDst[i*num + j] = 0.0;
    }
  }
  printf("finish matrix initialize\n");
}


void initialize_kernel(double* cpu_convkernel, double* gpu_convkernel)
{
  for(int i=0; i < 3; i++)
    for (int j=0; j<3; j++) {
      cpu_convkernel[i*3 + j] = gpu_convkernel[i*3 + j]  = ((double)rand() / ((double)RAND_MAX + 1));
    }
  
  printf("finish kernel initialize\n");
}


void cpu_function(int num, double *cpu_mat, double *cpu_convkernel, double *cpu_matDst, wtime_t &time)
{
  start_timer();
  launch_cpu(num, cpu_mat, cpu_convkernel, cpu_matDst);  // launch_cpu in cpu_calc.cpp
  stop_timer();
  time = elapsed_millis();
}

void gpu_function(int num, double *gpu_mat, double *gpu_convkernel, double *gpu_matDst, wtime_t &time)
{
  start_timer();
  launch_kernel(num, gpu_mat, gpu_convkernel,  gpu_matDst); //launch_kernel in gpu_calc.cu 
  stop_timer();
  time = elapsed_millis();
}

int main(int argc, char **argv)
{
  double *gpu_mat, *gpu_matDst;
  double *cpu_mat, *cpu_matDst;
  double *cpu_convkernel, *gpu_convkernel;

  double mat_result;
  wtime_t gpu_mult_time;
  wtime_t cpu_mult_time;

  int num = 1024;

  gpu_mat = (double*)malloc(sizeof(double) * num * num);
  gpu_matDst = (double*)malloc(sizeof(double) * num * num);
  gpu_convkernel = (double*)malloc(sizeof(double) * 3*3);
  cpu_mat = (double*)malloc(sizeof(double) * num * num);
  cpu_matDst = (double*)malloc(sizeof(double) * num * num);
  cpu_convkernel = (double*)malloc(sizeof(double) * 3*3);

  srand(time(NULL));
  initialize_mat(num, gpu_mat, gpu_matDst, cpu_mat, cpu_matDst);
  initialize_kernel(cpu_convkernel, gpu_convkernel);


  create_timer();
  gpu_function(num, gpu_mat, gpu_convkernel, gpu_matDst, gpu_mult_time);
  printf("gpu calculation finished \n");

  cpu_function(num, cpu_mat, cpu_convkernel, cpu_matDst, cpu_mult_time);
  printf("cpu calculation finished \n\n");

  mat_result = check(cpu_matDst, gpu_matDst, num);

  print_time(cpu_mult_time, gpu_mult_time);
  print_result(mat_result);

  destroy_timer();

  free(cpu_mat);
  free(cpu_convkernel);
  free(cpu_matDst);
  free(gpu_mat);
  free(gpu_convkernel);
  free(gpu_matDst);

  return 0;
}


#ifndef __CALCULATION_H_INCLUDED__
#define __CALCULATION_H_INCLUDED__



void launch_cpu(int num, double *cpu_mat, double *cpu_matB, double *cpu_matC);
void launch_kernel(int num, double *gpu_mat, double *gpu_matB, double *gpu_matC);


#endif /* __CALCULATION_H_INCLUDED__ */

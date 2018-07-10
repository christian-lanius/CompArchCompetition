#ifndef __TOOLKIT_H_INCLUDED__
#define __TOOLKIT_H_INCLUDED__

#include "timer.h"

void print_time(wtime_t cpu_mult_time, wtime_t gpu_mult_time);
void print_result(double mat_result);
double check(double *cpu_data, double *gpu_data, int num);

#endif /* __TOOLKIT_H_INCLUDED__ */

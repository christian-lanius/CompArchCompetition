/**************************************/
/* You don't have to change this code */
/* Please modify the "gpu_calc.cu"    */
/**************************************/

#include <stdio.h>
#include <stdlib.h>


void launch_cpu(int num, double *cpu_mat, double *cpu_convkernel, double *cpu_matDst)
{
  
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
      tmpmat[i][j] = cpu_mat[(i-1)*num + (j-1)];
    }
    tmpmat[i][num+1] = 0.0f;
  }

  ////////////////////////////////////

  for (int i=1; i<=num; i++) {
    for (int j=1; j<=num; j++) {
      double tmpsum = 0.0f;
      for (int ky=0; ky<3; ky++) 
      for (int kx=0; kx<3; kx++)
        tmpsum += cpu_convkernel[ ky*3 + kx] * tmpmat[i-1 + ky][j-1 + kx];
        
      cpu_matDst[ (i-1)*num + j-1 ] = tmpsum;
    }
  }


  // ------free------ // 
  for (int i=0; i<num+2; i++)  {
    free(tmpmat[i]);
  }
  free(tmpmat);
}

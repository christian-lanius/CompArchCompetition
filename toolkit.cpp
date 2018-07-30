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
#include "calculation.h"
#include "timer.h"
#include <complex>
#include <math.h>

static __inline void usage_line(const char *s1, const char *s2, const char *s3)
{
  printf(" %-10s, %-20s: %s\n", s1, s2, s3);
}

static void usage()
{
  printf("\nUsage: main [OPTION] ...\n\n");
  printf("Options:\n");
  usage_line("-n [1-8]", "--particles=[1-8]", 
             "number of particles (N=[1-8]x128)    [default: 1]");
  usage_line("-s [1-3]", "--select=[1-3]", 
             "select problem data set");
  usage_line("-h", "--help", 
             "show this Usage");
  printf("\n");
}

void initialize(int &num, char *filename, int argc, char **argv)
{
  int opt;
  int N = 128 * 1;
  int data_num = 1;

  srand((unsigned)time(NULL));

  while(1) {
	int option_index = 0;
	static struct option long_options[] = {
	  {"the number of elements", 1, NULL, 'n'},
	  {"select", 1, NULL, 's'},
	  {"help", 0, NULL, 'h'},
	  {0, 0, 0, 0}
	};
	opt = getopt_long(argc, argv, "n:s:h?",
					  long_options, &option_index);
	if(opt == -1){
	  break;
	}else if(opt == 0){
	  continue;
	}else{
	  switch(opt){
	  case 'n':
		if(1 <= atoi(optarg) && atoi(optarg) <= 8)
		  N = atoi(optarg) * 128;
		else if(atoi(optarg) == 0)
		  N = rand() % 1024;
		break;
	  case 's':
		data_num = atoi(optarg);
		break;
	  case 'h':
	  case '?':
	  default:
		usage();
	  exit(0);
	  break;
	  }
	}
  }
  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc) {
      printf("%s ", argv[optind++]);
    }
    printf("\n");
  }

  num = N;

  if(data_num == 1){
	strcpy(filename, "data_1.txt");
  }else if(data_num == 2){
	strcpy(filename, "data_2.txt");
  }else if(data_num == 3){
	strcpy(filename, "data_3.txt");
  }
  return;
}

void print_time(wtime_t cpu_mult_time, wtime_t gpu_mult_time)
{
  printf("Elapsed time on CPU               :   %.6f [msec]\n", cpu_mult_time);

  printf("Elapsed time on GPU               :   %.6f [msec]\n", gpu_mult_time);
  puts("");

  printf("%f times faster !\n", cpu_mult_time/gpu_mult_time);
}

void print_result(double mat_result)
{
  printf("Accumulated error is               %.9f\n", mat_result);

  if(mat_result >= 10e-9)
  	printf("\tincorrect...\n");
  else if (isnan(mat_result)) 
  	printf("\tincorrect...\n");
  else
  	printf("\tcorrect !!\n");
}
    


double check(double *cpu_resultMat, double *gpu_resultMat, int num)
{
  double sum = 0.0;

  for (int i=0; i<num; i++) {
    for (int j=0; j<num; j++) {
      double abs_diff = std::abs( gpu_resultMat[i*num + j] - cpu_resultMat[i*num + j]);


      sum += abs_diff;
    }
  }

  return sum;
}



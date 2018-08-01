################################################################################
#
CUDA_PATH = /usr/local/cuda-8.0
#CUDA_PATH = /usr/local/cuda
#
################################################################################
#
CC       = gcc
CXX      = g++
NVCC     = ${CUDA_PATH}/bin/nvcc
RM       = /bin/rm
AR       = ar
#
################################################################################
#
CFLAGS   += -O3 -Wall
CFLAGS   += -g
#CFLAGS   += -v
#CFLAGS   += -Q
LDFLAGS  += -L -lm
LDFLAGS  += -L${CUDA_PATH}/lib64 -lcudart -lcudadevrt
INCFLAGS  += -I${CUDA_PATH}/include
NFLAGS   += -arch sm_35
NFLAGS   += -Xptxas
NFLAGS   += -O3
NFLAGS   += -dc
#NFLAGS   += -ptxas
#
################################################################################
#
PROG      = conv
OBJS	  = timer.o toolkit.o cpu_calc.o gpu_calc.o main.o
#
################################################################################

${PROG}: ${OBJS}
	$(NVCC) -o $(PROG) $(LDFLAGS) -arch sm_35 $^

.SUFFIXES: .cpp .c .cu

.cpp.o:
	${CXX} -c ${LDFLAGS} ${CFLAGS} ${INCFLAGS} $<
.c.o:
	${CXX} -c ${LDFLAGS} ${CFLAGS} ${INCFLAGS} $<
.cu.o:
	${NVCC} -c -lm ${NFLAGS} $<

data: data.c
	${CC} -o $@ $^ -lm

clean :
	${RM} -f ${PROG} *.o *.out *~ data

update :
	make clean; make

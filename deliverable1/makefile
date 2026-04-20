CC=gcc
NVCC=nvcc
GPU_ARCH = --gpu-architecture=sm_80

CPU_FLAGS=-Wall
GPU_FLAGS=--compiler-options -Wall
LIBS=
INCLUDES=-I./include
LIB_FLAGS=-lm

OBJ_FOLDER := obj
SRC_FOLDER := src
BIN_FOLDER := bin

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o  $(OBJ_FOLDER)/spmv_utils.o


#############
#### COO ####
############# 
CPU_NAME_COO=cpu_coo
GPU_NAME_COO=gpu_coo_sequential

CPU_COO_BIN=$(BIN_FOLDER)/$(CPU_NAME_COO)
GPU_COO_BIN=$(BIN_FOLDER)/$(GPU_NAME_COO)

CPU_COO_SRC=$(SRC_FOLDER)/$(CPU_NAME_COO).c
GPU_COO_SRC=$(SRC_FOLDER)/$(GPU_NAME_COO).cu

$(CPU_COO_BIN): $(CPU_COO_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(CC) $(CPU_FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

$(GPU_COO_BIN): $(GPU_COO_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(NVCC) $(GPU_FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS) $(GPU_ARCH)


#############
#### CSR ####
############# 

CPU_NAME_CSR=cpu_csr
GPU_NAME_CSR=gpu_csr_sequential

CPU_CSR_BIN=$(BIN_FOLDER)/$(CPU_NAME_CSR)
GPU_CSR_BIN=$(BIN_FOLDER)/$(GPU_NAME_CSR)

CPU_CSR_SRC=$(SRC_FOLDER)/$(CPU_NAME_CSR).c
GPU_CSR_SRC=$(SRC_FOLDER)/$(GPU_NAME_CSR).cu

$(CPU_CSR_BIN): $(CPU_CSR_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(CC) $(CPU_FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

$(GPU_CSR_BIN): $(GPU_CSR_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(NVCC) $(GPU_FLAGS) $^ -o $@ $(INCLUDES) $(LIB_FLAGS) $(GPU_ARCH)





#################
#### GENERAL ####
#################
all: $(CPU_COO_BIN) $(GPU_COO_BIN) $(CPU_CSR_BIN) $(GPU_CSR_BIN)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/../lib/my_time_lib.c $(OBJ_FOLDER)/spmv_utils.o
	@mkdir -p $(OBJ_FOLDER)
	@$(CC) -c $(SRC_FOLDER)/../lib/my_time_lib.c -o $@ $(INCLUDES) $(LIB_FLAGS)
	
$(OBJ_FOLDER)/spmv_utils.o: ./lib/spmv_utils.c ./include/spmv_utils.h
	@mkdir -p $(OBJ_FOLDER)
	@$(CC) -c ./lib/spmv_utils.c -o $@ $(INCLUDES) $(LIB_FLAGS)
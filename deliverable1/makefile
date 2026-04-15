CC=gcc
NVCC=nvcc
GPU_ARCH = --gpu-architecture=sm_80

LIBS=
INCLUDES=-I./src/TIMER_LIB
LIB_FLAGS=-lm

OBJ_FOLDER := obj
SRC_FOLDER := src
BIN_FOLDER := bin

CPU_NAME=cpu_coo
GPU_NAME=gpu_coo_sequential

MAIN_BIN=$(BIN_FOLDER)/$(CPU_NAME)
GPU_BIN=$(BIN_FOLDER)/$(GPU_NAME)

MAIN_SRC=$(SRC_FOLDER)/$(CPU_NAME).c
GPU_SRC=$(SRC_FOLDER)/$(GPU_NAME).cu

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(MAIN_BIN) $(GPU_BIN)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/TIMER_LIB/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)
	@$(CC) -c $(SRC_FOLDER)/TIMER_LIB/my_time_lib.c -o $@ $(LIB_FLAGS)

$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

$(GPU_BIN): $(GPU_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(NVCC) $^ -o $@ $(INCLUDES) $(LIB_FLAGS) $(GPU_ARCH)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
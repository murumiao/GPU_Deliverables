###############################################################################
# Compiler configuration
###############################################################################
CC := gcc
MPICC := mpicc
NVCC := nvcc
GPU_ARCH := --gpu-architecture=sm_80

###############################################################################
# Compilation flags
###############################################################################

CFLAGS := -Wall -O3

NVCC_FLAGS := --compiler-options -Wall,-O3

# CUDA optimization flags
# CUDA_OPT_FLAGS := \
# 	-O3 \
# 	--use_fast_math \
# 	-Xptxas -O3 \
# 	--prec-div=false \
# 	--prec-sqrt=false \
# 	--ftz=true \
# 	--fmad=true \
# 	-lineinfo
CUDA_OPT_FLAGS := 

###############################################################################
# Include directories and libraries
###############################################################################

INCLUDES := -I./include
LIBS := -lm -lstdc++

###############################################################################
# Project directories
###############################################################################

SRC_DIR := src
LIB_DIR := lib
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin

###############################################################################
# Library object files
#
# Every executable will automatically link against these objects.
#
# If you add another library source file, simply add another object here.
###############################################################################

COMMON_OBJECTS := \
	$(OBJ_DIR)/my_time_lib.o \
	$(OBJ_DIR)/spmv_utils.o \
	$(OBJ_DIR)/deliverable1.o

###############################################################################
# Source discovery
#
# Automatically finds every .c and .cu file inside src/.
###############################################################################

C_SOURCES := $(wildcard $(SRC_DIR)/*.c)
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)

C_TARGETS := $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%,$(C_SOURCES))
CUDA_TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(CUDA_SOURCES))

###############################################################################
# Default target
###############################################################################

all: $(C_TARGETS) $(CUDA_TARGETS)

###############################################################################
# Build C executables
###############################################################################

$(BIN_DIR)/%: $(SRC_DIR)/%.c $(COMMON_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS)

###############################################################################
# CUDA executables
###############################################################################

$(BIN_DIR)/%: $(SRC_DIR)/%.cu $(COMMON_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) -ccbin mpicc -lnvidia-ml -lmpi \
		$(NVCC_FLAGS) $(CUDA_OPT_FLAGS) $(GPU_ARCH) \
		$^ -o $@ $(INCLUDES) $(LIBS)

###############################################################################
# Library compilation
###############################################################################

$(OBJ_DIR)/my_time_lib.o: $(LIB_DIR)/my_time_lib.c $(INC_DIR)/my_time_lib.h
	@mkdir -p $(OBJ_DIR)
	$(CC) -c $(CFLAGS) $< -o $@ $(INCLUDES)

$(OBJ_DIR)/spmv_utils.o: $(LIB_DIR)/spmv_utils.c $(INC_DIR)/spmv_utils.h
	@mkdir -p $(OBJ_DIR)
	$(CC) -c $(CFLAGS) $< -o $@ $(INCLUDES)

$(OBJ_DIR)/deliverable1.o: $(LIB_DIR)/deliverable1.cu $(INC_DIR)/deliverable1.cuh
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $(NVCC_FLAGS) $(CUDA_OPT_FLAGS) $(GPU_ARCH) $< -o $@ $(INCLUDES)

###############################################################################
# Utility targets
###############################################################################

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

###############################################################################
# Phony targets
###############################################################################

.PHONY: all clean
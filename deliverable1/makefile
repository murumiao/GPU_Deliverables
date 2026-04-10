CC=gcc

LIBS=
INCLUDES=-I./src/TIMER_LIB
LIB_FLAGS=-lm

OBJ_FOLDER := obj
SRC_FOLDER := src
BIN_FOLDER := bin

MAIN_NAME=cpu
MAIN_BIN=$(BIN_FOLDER)/$(MAIN_NAME)
MAIN_SRC=$(SRC_FOLDER)/$(MAIN_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(MAIN_BIN)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/TIMER_LIB/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)
	@$(CC) -c $(SRC_FOLDER)/TIMER_LIB/my_time_lib.c -o $@ $(LIB_FLAGS)

$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	@$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
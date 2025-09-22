CC = gcc
CFLAGS = -O3 -march=native -ffast-math -mfma -funroll-loops -pthread

# Macro defaults (override on the command line if you want)
KC ?= 256
MC ?= 6
NC ?= 64

# Turn macros into -D flags
DFLAGS := -DKC=$(KC) -DMC=$(MC) -DNC=$(NC)

SRC := dgemm.c
BIN_DIR := bin
OBJ := $(patsubst %.c,$(BIN_DIR)/%.o,$(SRC))
BIN := $(BIN_DIR)/dgemm

.PHONY: all clean

all: $(BIN)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/%.o: %.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN): $(OBJ) Makefile
	$(CC) $(OBJ) -o $@

clean:
	rm -rf $(BIN_DIR)

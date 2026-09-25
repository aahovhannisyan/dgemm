CC ?= cc
ARCH := $(shell uname -m)

ifeq ($(ARCH),$(filter $(ARCH),arm64 aarch64))
ARCH_FLAGS = -mcpu=native
# Tuned on Apple M5 Pro (5+10 cores). 16 threads split the 16 NC-wide column blocks evenly.
THREADS ?= 16
KC ?= 384
NC ?= 256
else
ARCH_FLAGS = -march=native -mfma
THREADS ?= 12
endif

CFLAGS = -O3 $(ARCH_FLAGS) -ffast-math -funroll-loops -pthread

# Macro defaults (override on the command line if you want)
KC ?= 256
MC ?= 6
NC ?= 64

# Turn macros into -D flags
DFLAGS := -DKC=$(KC) -DMC=$(MC) -DNC=$(NC) -DNUM_THREADS=$(THREADS)
ifdef VERIFY
DFLAGS += -DVERIFY
endif

BIN_DIR := bin
BIN := $(BIN_DIR)/dgemm
TEST_BIN := $(BIN_DIR)/test_gemm_local
LIB_OBJ := $(BIN_DIR)/gemm_local.o
HDR := src/gemm_local.h

.PHONY: all test clean

all: $(BIN)

$(BIN_DIR):
	mkdir -p $@

$(LIB_OBJ): src/gemm_local.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN_DIR)/dgemm.o: dgemm.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN_DIR)/test_gemm_local.o: tests/test_gemm_local.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(BIN_DIR)/dgemm.o $(LIB_OBJ)
	$(CC) -pthread $^ -o $@

$(TEST_BIN): $(BIN_DIR)/test_gemm_local.o $(LIB_OBJ)
	$(CC) -pthread $^ -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BIN_DIR)

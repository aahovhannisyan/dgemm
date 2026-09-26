CC ?= cc
MPICC ?= mpicc
ARCH := $(shell uname -m)
OS := $(shell uname -s)

ifeq ($(OS)-$(ARCH),Darwin-arm64)
ARCH_FLAGS = -mcpu=native
# Tuned on Apple M5 Pro (5+10 cores). 16 threads split the 16 NC-wide column blocks evenly.
THREADS ?= 16
KC ?= 384
NC ?= 256
else ifeq ($(ARCH),aarch64)
ARCH_FLAGS = -mcpu=native
# Graviton3E (hpc7g.16xlarge, 64 Neoverse-V1 cores). Block sizes not yet tuned.
THREADS ?= 64
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
SUMMA_BIN := $(BIN_DIR)/summa
LIB_OBJ := $(BIN_DIR)/gemm_local.o
HDR := src/gemm_local.h

.PHONY: all summa test test-summa clean

all: $(BIN)

$(BIN_DIR):
	mkdir -p $@

$(LIB_OBJ): src/gemm_local.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN_DIR)/dgemm.o: dgemm.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN_DIR)/test_gemm_local.o: tests/test_gemm_local.c $(HDR) Makefile | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/summa.o: src/summa.c $(HDR) Makefile | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $(DFLAGS) -c $< -o $@

$(BIN): $(BIN_DIR)/dgemm.o $(LIB_OBJ)
	$(CC) -pthread $^ -o $@

$(TEST_BIN): $(BIN_DIR)/test_gemm_local.o $(LIB_OBJ)
	$(CC) -pthread $^ -o $@

$(SUMMA_BIN): $(BIN_DIR)/summa.o $(LIB_OBJ)
	$(MPICC) -pthread $^ -o $@

summa: $(SUMMA_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

test-summa: $(SUMMA_BIN)
	./tests/test_summa.sh

clean:
	rm -rf $(BIN_DIR)

# KFCompare Test Makefile
# Compiles kernels twice (baseline vs KernelFaRer) and links both into test

# System compiler for test harness (needs C++ stdlib)
CXX ?= clang++

# LLVM compiler for kernels (needs KernelFaRer plugin)
LLVM_CXX ?= clang++

# Paths
KFCOMPARE_INC = ../include
KFCOMPARE_SRC = ../src
KERNEL_DIR = ../kernels

# KernelFaRer plugin
PLUGIN ?= ../../build/passes/KernelFaRer.so
PASS = kernel-replacer-pass
REPLACEMENT_MODE ?= cblas-interface

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -ffast-math -ffp-contract=fast -I$(KFCOMPARE_INC)
KERNEL_FLAGS = -O3 -ffp-contract=off
KFARER_FLAGS = -fplugin=$(PLUGIN) -fpass-plugin=$(PLUGIN) \
               -mllvm --kernelfarer-replacement-mode=$(REPLACEMENT_MODE) \
               -Rpass-analysis=$(PASS) -Rpass-missed=$(PASS)

# BLAS library for KernelFaRer-transformed code
LDFLAGS_BLAS ?= -lopenblas

# ============================================================================
# Test definitions: (name, kernel_source, func_name)
# ============================================================================
TESTS = test_naive_gemm \
        test_skip_gemm_i2 test_skip_gemm_j2 test_skip_gemm_i64 \
        test_skip_gemm_block_i2 test_skip_gemm_block_j2 \
        test_skip_gemm_block_i64 test_skip_gemm_block_j64 \
        test_sm_gemm_0 test_sm_gemm_1 test_sm_gemm_2 test_trmm_0

# test_naive_gemm
test_naive_gemm_SRC  = naive_gemm.cc
test_naive_gemm_FUNC = basicDgemm

# test_skip_gemm_* (simple striped)
test_skip_gemm_i2_SRC  = skip_gemm_i2.cc
test_skip_gemm_i2_FUNC = striped_gemm

test_skip_gemm_j2_SRC  = skip_gemm_j2.cc
test_skip_gemm_j2_FUNC = striped_gemm

test_skip_gemm_i64_SRC  = skip_gemm_i64.cc
test_skip_gemm_i64_FUNC = striped_gemm

# test_skip_gemm_block_* (blocked striped)
test_skip_gemm_block_i2_SRC  = skip_gemm_block_i2.cc
test_skip_gemm_block_i2_FUNC = striped_gemm

test_skip_gemm_block_j2_SRC  = skip_gemm_block_j2.cc
test_skip_gemm_block_j2_FUNC = striped_gemm

test_skip_gemm_block_i64_SRC  = skip_gemm_block_i64.cc
test_skip_gemm_block_i64_FUNC = striped_gemm

test_skip_gemm_block_j64_SRC  = skip_gemm_block_j64.cc
test_skip_gemm_block_j64_FUNC = striped_gemm

# test_sm_gemm_0 through test_sm_gemm_2
test_sm_gemm_0_SRC  = sm_gemm_0.cc
test_sm_gemm_0_FUNC = submatrix_gemm

test_sm_gemm_1_SRC  = sm_gemm_1.cc
test_sm_gemm_1_FUNC = submatrix_gemm

test_sm_gemm_2_SRC  = sm_gemm_2.cc
test_sm_gemm_2_FUNC = submatrix_gemm

# test_trmm_0
test_trmm_0_SRC  = trmm_0.cc
test_trmm_0_FUNC = basicStrmmLower

# ============================================================================
# Generic infrastructure
# ============================================================================
.PHONY: all clean help check-plugin run

all: $(TESTS)

# Build KFCompare library object
kfcompare.o: $(KFCOMPARE_SRC)/KFCompare.cpp $(KFCOMPARE_INC)/KFCompare.h $(KFCOMPARE_INC)/KFCompare.inl
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Check that plugin loads correctly
check-plugin:
	@echo "Checking LLVM/Clang version compatibility..."
	@$(LLVM_CXX) --version | head -1
	@echo "Testing plugin load..."
	@$(LLVM_CXX) -fplugin=$(PLUGIN) -fpass-plugin=$(PLUGIN) -x c++ -c /dev/null -o /dev/null 2>&1 || \
		(echo "ERROR: Plugin failed to load. Rebuild KernelFaRer with your LLVM clang version." && exit 1)
	@echo "Plugin OK!"

# Macro: compile kernel with renamed symbol via preprocessor
# $(1)=output.o, $(2)=source, $(3)=original func name, $(4)=new func name, $(5)=extra flags
define compile_kernel
	$(LLVM_CXX) $(KERNEL_FLAGS) $(5) -D$(3)=$(4) -c $(2) -o $(1)
endef

# Macro: generate rules for a test
# $(1)=test name (e.g., test_gemm_0)
define make_test_rules
$(1)_baseline.o: $$(KERNEL_DIR)/$$($(1)_SRC)
	$$(call compile_kernel,$$@,$$<,$$($(1)_FUNC),$$($(1)_FUNC)_baseline,)

$(1)_kfarer.o: $$(KERNEL_DIR)/$$($(1)_SRC)
	$$(call compile_kernel,$$@,$$<,$$($(1)_FUNC),$$($(1)_FUNC)_kfarer,$$(KFARER_FLAGS))

$(1): $(1).cpp $(1)_baseline.o $(1)_kfarer.o kfcompare.o
	$$(CXX) $$(CXXFLAGS) $$^ -o $$@ $$(LDFLAGS_BLAS)
endef

# Generate rules for all tests
$(foreach t,$(TESTS),$(eval $(call make_test_rules,$(t))))

# ============================================================================
# Run and clean
# ============================================================================

# Optional SIZE parameter (e.g., SIZE=512 or SIZE=128x256x64)
ifdef SIZE
  SIZE_ARG = --size $(SIZE)
else
  SIZE_ARG =
endif

# Optional BENCH=1 for benchmark mode (5 runs, average middle 3)
ifdef BENCH
  BENCH_ARG = --bench
else
  BENCH_ARG =
endif

TEST_ARGS = $(SIZE_ARG) $(BENCH_ARG)

run: $(TESTS)
	@for t in $(TESTS); do echo "" && ./$$t $(TEST_ARGS); done

# Stress test with many random iterations (default 100, override with STRESS_ITERS)
STRESS_ITERS ?= 100
stress: $(TESTS)
	@for t in $(TESTS); do echo "" && ./$$t $(TEST_ARGS) --stress $(STRESS_ITERS); done

clean:
	rm -f *.o $(TESTS) *.log

help:
	@echo "KFCompare Test Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make                     - Build all tests"
	@echo "  make test_xxx            - Build specific test"
	@echo "  make run                 - Run all tests (default size 256)"
	@echo "  make run SIZE=512        - Run with custom size (M=N=K=512)"
	@echo "  make run SIZE=128x256x64 - Run with M=128, N=256, K=64"
	@echo "  make run BENCH=1         - Benchmark mode (5 runs, avg middle 3)"
	@echo "  make run SIZE=512 BENCH=1"
	@echo "  make stress              - Stress test (100 iterations)"
	@echo "  make stress SIZE=512 STRESS_ITERS=1000"
	@echo "  make check-plugin        - Verify KernelFaRer plugin loads"
	@echo "  make clean               - Remove built files"
	@echo ""
	@echo "To add a new test, define:"
	@echo "  test_xxx_SRC  = kernel-file.cc"
	@echo "  test_xxx_FUNC = function_name"
	@echo "  TESTS += test_xxx"

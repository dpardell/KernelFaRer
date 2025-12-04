# KFCompare Test Makefile
# Compiles kernels twice (baseline vs KernelFaRer) and links both into test

# System compiler for test harness (needs C++ stdlib)
CXX ?= clang++

# LLVM compiler for kernels (needs KernelFaRer plugin)
LLVM_CXX ?= clang++

OBJCOPY ?= llvm-objcopy

# Paths
KFCOMPARE_INC = ../include
KFCOMPARE_SRC = ../src
KERNEL_DIR = ../../cplus-tests

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
# Test definitions: (name, kernel_source, grep_pattern, func_name)
# ============================================================================
TESTS = test_gemm_0 test_skip_gemm_0 test_skip_gemm_1 test_skip_gemm_2 \
        test_skip_gemm_3 test_skip_gemm_4 test_skip_gemm_5 \
        test_sm_gemm_0 test_sm_gemm_1 test_sm_gemm_2 test_trmm_0

# test_gemm_0
test_gemm_0_SRC    = gemm-0.cc
test_gemm_0_GREP   = sgemm
test_gemm_0_FUNC   = basicSgemm

# test_skip_gemm_0 through test_skip_gemm_5
test_skip_gemm_0_SRC    = skip-gemm-0.cc
test_skip_gemm_0_GREP   = striped_gemm
test_skip_gemm_0_FUNC   = striped_gemm

test_skip_gemm_1_SRC    = skip-gemm-1.cc
test_skip_gemm_1_GREP   = striped_gemm
test_skip_gemm_1_FUNC   = striped_gemm

test_skip_gemm_2_SRC    = skip-gemm-2.cc
test_skip_gemm_2_GREP   = striped_gemm
test_skip_gemm_2_FUNC   = striped_gemm

test_skip_gemm_3_SRC    = skip-gemm-3.cc
test_skip_gemm_3_GREP   = striped_gemm
test_skip_gemm_3_FUNC   = striped_gemm

test_skip_gemm_4_SRC    = skip-gemm-4.cc
test_skip_gemm_4_GREP   = striped_gemm
test_skip_gemm_4_FUNC   = striped_gemm

test_skip_gemm_5_SRC    = skip-gemm-5.cc
test_skip_gemm_5_GREP   = striped_gemm
test_skip_gemm_5_FUNC   = striped_gemm

# test_sm_gemm_0 through test_sm_gemm_2
test_sm_gemm_0_SRC    = sm-gemm-0.cc
test_sm_gemm_0_GREP   = submatrix_gemm
test_sm_gemm_0_FUNC   = submatrix_gemm

test_sm_gemm_1_SRC    = sm-gemm-1.cc
test_sm_gemm_1_GREP   = submatrix_gemm
test_sm_gemm_1_FUNC   = submatrix_gemm

test_sm_gemm_2_SRC    = sm-gemm-2.cc
test_sm_gemm_2_GREP   = submatrix_gemm
test_sm_gemm_2_FUNC   = submatrix_gemm

# test_trmm_0
test_trmm_0_SRC    = trmm-0.cc
test_trmm_0_GREP   = strmm
test_trmm_0_FUNC   = basicStrmm

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

# Macro: compile kernel and rename symbol
# $(1)=output.o, $(2)=source, $(3)=grep pattern, $(4)=new func name, $(5)=extra flags
define compile_kernel
	$(LLVM_CXX) $(KERNEL_FLAGS) $(5) -c $(2) -o $(1)
	SYM=$$(nm $(1) | grep -i '$(3)' | grep ' T ' | awk '{print $$3}') && \
	$(OBJCOPY) --redefine-sym $$SYM=_$(4) $(1)
endef

# Macro: generate rules for a test
# $(1)=test name (e.g., test_gemm_0)
define make_test_rules
$(1)_baseline.o: $$(KERNEL_DIR)/$$($(1)_SRC)
	$$(call compile_kernel,$$@,$$<,$$($(1)_GREP),$$($(1)_FUNC)_baseline,)

$(1)_kfarer.o: $$(KERNEL_DIR)/$$($(1)_SRC) | check-plugin
	$$(call compile_kernel,$$@,$$<,$$($(1)_GREP),$$($(1)_FUNC)_kfarer,$$(KFARER_FLAGS))

$(1): $(1).cpp $(1)_baseline.o $(1)_kfarer.o kfcompare.o
	$$(CXX) $$(CXXFLAGS) $$^ -o $$@ $$(LDFLAGS_BLAS)
endef

# Generate rules for all tests
$(foreach t,$(TESTS),$(eval $(call make_test_rules,$(t))))

# ============================================================================
# Run and clean
# ============================================================================
run: $(TESTS)
	@for t in $(TESTS); do echo "" && ./$$t; done

# Stress test with many random iterations (default 100, override with STRESS_ITERS)
STRESS_ITERS ?= 100
stress: $(TESTS)
	@for t in $(TESTS); do echo "" && ./$$t --stress $(STRESS_ITERS); done

clean:
	rm -f *.o $(TESTS) *.log

help:
	@echo "KFCompare Test Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all tests"
	@echo "  make test_xxx     - Build specific test"
	@echo "  make run          - Build and run all tests (single comparison)"
	@echo "  make stress       - Run stress tests (100 random iterations)"
	@echo "  make stress STRESS_ITERS=1000  - Custom iteration count"
	@echo "  make check-plugin - Verify KernelFaRer plugin loads"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "To add a new test, define:"
	@echo "  test_xxx_SRC  = kernel-file.cc"
	@echo "  test_xxx_GREP = symbol_pattern"
	@echo "  test_xxx_FUNC = function_name"
	@echo "  TESTS += test_xxx"

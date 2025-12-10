# KFCompare Test CMakeLists.txt
# Compiles kernels twice (baseline vs KernelFaRer) and links both into test

cmake_minimum_required(VERSION 3.22)

project(KFCompare)

# Paths
set(KFCOMPARE_INC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(KFCOMPARE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../kernels")

# ============================================================================
# Configuration Options
# ============================================================================
# These can be set via:
#   cmake -B build -DPLUGIN=/path/to/plugin.so -DLDFLAGS_BLAS=mkl ...
#   Or use ccmake or cmake-gui for interactive configuration
# ============================================================================

# KernelFaRer plugin
# Determine project root (go up 3 levels from kfcompare/tests/)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# Auto-detect plugin if not specified
if(NOT DEFINED PLUGIN)
    # Try .so first, then .dylib (for macOS)
    if(EXISTS "${PROJECT_ROOT}/build/passes/KernelFaRer.so")
        set(PLUGIN_DEFAULT "${PROJECT_ROOT}/build/passes/KernelFaRer.so")
    elseif(EXISTS "${PROJECT_ROOT}/build/passes/KernelFaRer.dylib")
        set(PLUGIN_DEFAULT "${PROJECT_ROOT}/build/passes/KernelFaRer.dylib")
    else()
        set(PLUGIN_DEFAULT "${PROJECT_ROOT}/build/passes/KernelFaRer.so")
    endif()
else()
    set(PLUGIN_DEFAULT "${PLUGIN}")
endif()

set(PLUGIN "${PLUGIN_DEFAULT}" CACHE FILEPATH 
    "Path to KernelFaRer plugin (e.g., /path/to/KernelFaRer.so or KernelFaRer.dylib)")

set(PASS "kernel-replacer-pass" CACHE STRING "KernelFaRer pass name")
set(REPLACEMENT_MODE "cblas-interface" CACHE STRING 
    "KernelFaRer replacement mode (e.g., cblas-interface)")

# Compiler flags
set(CXXFLAGS "-std=c++17" "-O3" "-ffast-math" "-ffp-contract=fast")
set(KERNEL_FLAGS "-O3" "-ffp-contract=off")
set(KFARER_FLAGS 
    "-fplugin=${PLUGIN}"
    "-fpass-plugin=${PLUGIN}"
    "-mllvm" "--kernelfarer-replacement-mode=${REPLACEMENT_MODE}"
    "-Rpass-analysis=${PASS}"
    "-Rpass-missed=${PASS}"
)

# BLAS library for KernelFaRer-transformed code
# Examples: "openblas", "mkl", "blas", "accelerate" (macOS)
if(NOT DEFINED LDFLAGS_BLAS)
    set(LDFLAGS_BLAS "openblas")
endif()
set(LDFLAGS_BLAS "${LDFLAGS_BLAS}" CACHE STRING 
    "BLAS library name for linking (e.g., openblas, mkl, blas, accelerate)")


# ============================================================================
# Validate configuration
# ============================================================================
if(NOT EXISTS "${PLUGIN}")
    message(WARNING "KernelFaRer plugin not found at: ${PLUGIN}")
    message(WARNING "  Build the plugin first or specify -DPLUGIN=/path/to/plugin")
    message(WARNING "  Continuing anyway, but kernel compilation may fail...")
else()
    message(STATUS "Plugin found: ${PLUGIN}")
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "KFCompare Test Configuration:")
message(STATUS "  Plugin: ${PLUGIN}")
message(STATUS "  Replacement Mode: ${REPLACEMENT_MODE}")
message(STATUS "  BLAS Library: ${LDFLAGS_BLAS}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "")

# Include directories
include_directories(${KFCOMPARE_INC})

# ============================================================================
# Enable CTest
# ============================================================================
enable_testing()

# ============================================================================
# Auto-discover tests
# ============================================================================
# Find all test files (test_*.cpp)
file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

# Function to extract function name from kernel file
function(extract_function_name kernel_file func_name_var)
    # Read the kernel file and extract the first function name after "extern \"C\" void"
    file(READ "${kernel_file}" kernel_content)
    # Match: extern "C" void followed by optional whitespace, then function name
    string(REGEX MATCH "extern \"C\" void[ \t\n]*([a-zA-Z_][a-zA-Z0-9_]*)" func_match "${kernel_content}")
    if(func_match)
        string(REGEX REPLACE "extern \"C\" void[ \t\n]*([a-zA-Z_][a-zA-Z0-9_]*).*" "\\1" extracted_func "${func_match}")
        set(${func_name_var} "${extracted_func}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Could not extract function name from ${kernel_file}")
    endif()
endfunction()

# Process each test file
set(TESTS "")
foreach(TEST_FILE ${TEST_FILES})
    # Extract test name from filename (test_foo.cpp -> foo)
    get_filename_component(TEST_FILE_BASE ${TEST_FILE} NAME_WE)
    string(REGEX REPLACE "^test_" "" TEST_NAME ${TEST_FILE_BASE})
    
    # Determine kernel file path
    set(KERNEL_FILE "${KERNEL_DIR}/${TEST_NAME}.cc")
    
    # Check kernel file exists
    if(EXISTS "${KERNEL_FILE}")
        # Extract function name from kernel file
        extract_function_name("${KERNEL_FILE}" FUNC_NAME)
        
        # Store test information
        set(TEST_${TEST_NAME}_SRC "${TEST_NAME}.cc")
        set(TEST_${TEST_NAME}_FUNC "${FUNC_NAME}")
        
        list(APPEND TESTS ${TEST_NAME})
    else()
        message(WARNING "Kernel file not found for test ${TEST_NAME}: ${KERNEL_FILE}")
    endif()
endforeach()

list(LENGTH TESTS TESTS_COUNT)
message(STATUS "Discovered ${TESTS_COUNT} tests: ${TESTS}")

# ============================================================================
# Build KFCompare library object
# ============================================================================
add_library(kfcompare_obj OBJECT
    ${KFCOMPARE_SRC}/KFCompare.cpp
)

target_compile_options(kfcompare_obj PRIVATE ${CXXFLAGS})
target_include_directories(kfcompare_obj PRIVATE ${KFCOMPARE_INC})

# ============================================================================
# Function to add a test
# ============================================================================
function(add_kfcompare_test test_name)
    # Get test-specific variables
    set(kernel_src "${TEST_${test_name}_SRC}")
    set(func_name "${TEST_${test_name}_FUNC}")
    
    if(NOT kernel_src OR NOT func_name)
        message(FATAL_ERROR "Test ${test_name} missing SRC or FUNC definition")
    endif()
    
    set(kernel_path "${KERNEL_DIR}/${kernel_src}")
    set(test_cpp "${CMAKE_CURRENT_SOURCE_DIR}/test_${test_name}.cpp")
    
    # Check files exist
    if(NOT EXISTS "${kernel_path}")
        message(FATAL_ERROR "Kernel source not found: ${kernel_path}")
    endif()
    if(NOT EXISTS "${test_cpp}")
        message(FATAL_ERROR "Test source not found: ${test_cpp}")
    endif()
    
    # Compile baseline version (no plugin, rename function)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_baseline.o
        COMMAND ${CMAKE_CXX_COMPILER} ${KERNEL_FLAGS}
            -D${func_name}=${func_name}_baseline
            -c ${kernel_path}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_baseline.o
        DEPENDS ${kernel_path}
        COMMENT "Compiling baseline kernel for ${test_name}"
        VERBATIM
    )
    
    # Compile KernelFaRer version (with plugin, rename function)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_kfarer.o
        COMMAND ${CMAKE_CXX_COMPILER} ${KERNEL_FLAGS} ${KFARER_FLAGS}
            -D${func_name}=${func_name}_kfarer
            -c ${kernel_path}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_kfarer.o
        DEPENDS ${kernel_path}
        COMMENT "Compiling KernelFaRer kernel for ${test_name}"
        VERBATIM
    )
    
    # Create custom targets for object files
    add_custom_target(${test_name}_baseline_obj
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_baseline.o
    )
    
    add_custom_target(${test_name}_kfarer_obj
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_kfarer.o
    )
    
    # Link test executable
    add_executable(test_${test_name}
        ${test_cpp}
        $<TARGET_OBJECTS:kfcompare_obj>
        ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_baseline.o
        ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_kfarer.o
    )
    
    target_compile_options(test_${test_name} PRIVATE ${CXXFLAGS})
    target_include_directories(test_${test_name} PRIVATE ${KFCOMPARE_INC})
    target_link_libraries(test_${test_name} PRIVATE ${LDFLAGS_BLAS})
    
    # Ensure object files are built before linking
    add_dependencies(test_${test_name} ${test_name}_baseline_obj ${test_name}_kfarer_obj)
    
    # Register test with CTest
    add_test(NAME ${test_name} COMMAND test_${test_name})
endfunction()

# ============================================================================
# Add all tests
# ============================================================================
foreach(test ${TESTS})
    add_kfcompare_test(${test})
endforeach()

# ============================================================================
# Custom target for running tests (uses CTest)
# ============================================================================
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_EXECUTABLES}
    COMMENT "Running all KFCompare tests with CTest"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Check that plugin loads correctly
add_custom_target(check-plugin
    COMMAND ${CMAKE_COMMAND} -E echo "Checking LLVM/Clang version compatibility..."
    COMMAND ${CMAKE_CXX_COMPILER} --version
    COMMAND ${CMAKE_COMMAND} -E echo "Testing plugin load..."
    COMMAND ${CMAKE_CXX_COMPILER} -fplugin=${PLUGIN} -fpass-plugin=${PLUGIN} -x c++ -c /dev/null -o /dev/null
    COMMENT "Verifying KernelFaRer plugin loads correctly"
    VERBATIM
)


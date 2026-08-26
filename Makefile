# ================================================================
# Makefile — STR R-Tree GPU bulk-loader
# ================================================================
#
# Directory layout:
#   src/   source files
#   bin/   compiled executables        (run from project root)
#   data/  input/output .bin datasets
#   tmp/   intermediate sort files     (auto-cleaned)
#
# Quick start:
#   make                                   # build everything
#   ./bin/gen_points 1000000000            # generate 1B points -> data/points.bin
#   ./bin/external_str data/points.bin data/rtree.bin
# ================================================================

SRC_DIR  := src
BIN_DIR  := bin
DATA_DIR := data
TMP_DIR  := tmp

NVCC      := nvcc
CXX       := g++
NVCCFLAGS := -O3 -std=c++17 -arch=sm_86
CXXFLAGS  := -O3 -std=c++17

# ----------------------------------------------------------------
# Default target
# ----------------------------------------------------------------
.PHONY: all
all: $(BIN_DIR)/external_str $(BIN_DIR)/external_str_cpu $(BIN_DIR)/rtree_query \
     $(BIN_DIR)/gen_points $(BIN_DIR)/gpu_info \
     $(BIN_DIR)/str_rtree $(BIN_DIR)/rect_rtree_query $(BIN_DIR)/gen_rects

# ----------------------------------------------------------------
# Binaries
# ----------------------------------------------------------------
$(BIN_DIR)/external_str: $(SRC_DIR)/external_str_cuda.cu $(SRC_DIR)/constants.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ 

$(BIN_DIR)/external_str_cpu: $(SRC_DIR)/external_str_cpu.cpp $(SRC_DIR)/constants.h
	$(CXX) $(CXXFLAGS) $< -o $@ -lpthread

$(BIN_DIR)/rtree_query: $(SRC_DIR)/rtree_query.cpp $(SRC_DIR)/constants.h
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/gen_points: $(SRC_DIR)/gen_points.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/gpu_info: $(SRC_DIR)/gpu_info.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# ---- textbook-layout STR R-Tree over rectangles (refactor/the-return-of-the-programer)
RECT_HDRS := $(SRC_DIR)/rect_constants.h $(SRC_DIR)/rect_rtree_format.h

$(BIN_DIR)/str_rtree: $(SRC_DIR)/str_rtree_cuda.cu $(RECT_HDRS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/rect_rtree_query: $(SRC_DIR)/rect_rtree_query.cpp $(RECT_HDRS)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/gen_rects: $(SRC_DIR)/gen_rects.cpp $(RECT_HDRS)
	$(CXX) $(CXXFLAGS) $< -o $@

# ----------------------------------------------------------------
# Benchmarks (see bench/ and the methodology section of context.md)
# ----------------------------------------------------------------
BENCH_DIR := bench

.PHONY: bench
bench: $(BIN_DIR)/bench_sort_crossover $(BIN_DIR)/bench_pinned_memory $(BIN_DIR)/bench_context_init

$(BIN_DIR)/bench_sort_crossover: $(BENCH_DIR)/bench_sort_crossover.cu $(RECT_HDRS)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $< -o $@

$(BIN_DIR)/bench_pinned_memory: $(BENCH_DIR)/bench_pinned_memory.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/bench_context_init: $(BENCH_DIR)/bench_context_init.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# ----------------------------------------------------------------
# Utility targets
# ----------------------------------------------------------------
.PHONY: clean clean-tmp clean-data help

# Remove compiled binaries (keeps directories)
clean:
	rm -f $(BIN_DIR)/external_str $(BIN_DIR)/external_str_cpu $(BIN_DIR)/rtree_query \
	      $(BIN_DIR)/gen_points $(BIN_DIR)/gpu_info \
	      $(BIN_DIR)/str_rtree $(BIN_DIR)/rect_rtree_query $(BIN_DIR)/gen_rects \
	      $(BIN_DIR)/bench_sort_crossover $(BIN_DIR)/bench_pinned_memory \
	      $(BIN_DIR)/bench_context_init

# Remove intermediate sort files left in tmp/
clean-tmp:
	find $(TMP_DIR) -name "*.bin" -o -name "*.tmp" | xargs rm -f

# Remove dataset files in data/
clean-data:
	find $(DATA_DIR) -name "*.bin" | xargs rm -f

view:
	python3 src/rtree_viewer.py data/rtree.bin

# ----------------------------------------------------------------
# Help
# ----------------------------------------------------------------
help:
	@echo ""
	@echo "  make                 Build all executables -> bin/"
	@echo "  make external_str    Build the STR R-Tree bulk-loader"
	@echo "  make gen_points      Build the point generator"
	@echo "  make gpu_info        Build the GPU info tool"
	@echo ""
	@echo "  make clean           Remove compiled binaries"
	@echo "  make clean-tmp       Remove intermediate files in tmp/"
	@echo "  make clean-data      Remove dataset files in data/"
	@echo "  make view            Visualize the R-Tree structure (requires matplotlib)"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/gpu_info"
	@echo "  ./bin/gen_points <N>                        # writes data/points.bin"
	@echo "  ./bin/gen_points <N> <output>               # writes to custom path"
	@echo "  ./bin/external_str data/points.bin data/rtree.bin"
	@echo ""
	@echo "Rectangle R-Tree (textbook node layout):"
	@echo "  ./bin/gen_rects <N> [out.bin] [--clustered K]"
	@echo "  ./bin/str_rtree data/rects.bin data/tree.bin [--fill-leaf F] [--fill-internal F]"
	@echo "  ./bin/rect_rtree_query data/tree.bin verify data/rects.bin"
	@echo "  ./bin/rect_rtree_query data/tree.bin bench  data/rects.bin 0.01 20"
	@echo ""
	@echo "Benchmarks / tests:"
	@echo "  make bench                        Build the microbenchmarks"
	@echo "  ./bin/bench_sort_crossover 24 5   GPU vs CPU sort break-even"
	@echo "  ./bin/bench_pinned_memory 3       Pinned alloc cost vs H2D bandwidth"
	@echo "  ./bin/bench_context_init          CUDA start-up cost (run cold, then warm)"
	@echo "  ./bench/sweep_pinned_cap.sh <input.bin>"
	@echo "  ./bench/test_correctness.sh [N]   Full correctness matrix"
	@echo ""

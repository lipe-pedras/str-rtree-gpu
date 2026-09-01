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
#   make                                          # build everything
#   ./bin/gen_rects 60000000 data/rects.bin       # 1.44 GB of rectangles
#   ./bin/str_rtree data/rects.bin data/tree.bin
# ================================================================

SRC_DIR  := src
BIN_DIR  := bin
DATA_DIR := data
TMP_DIR  := tmp
ANIM_DIR := animation

MANIM_VENV := $(HOME)/.venvs/manim
MANIM_Q    := h        # l = fast draft, h = 1080p60; override: make render MANIM_Q=l

NVCC      := nvcc
CXX       := g++
NVCCFLAGS := -O3 -std=c++17 -arch=sm_86
CXXFLAGS  := -O3 -std=c++17

# ----------------------------------------------------------------
# Default target
# ----------------------------------------------------------------
.PHONY: all
all: $(BIN_DIR)/gpu_info \
     $(BIN_DIR)/str_rtree $(BIN_DIR)/str_rtree_cpu \
     $(BIN_DIR)/rect_rtree_query $(BIN_DIR)/gen_rects

# ----------------------------------------------------------------
# Binaries
# ----------------------------------------------------------------
RECT_HDRS := $(SRC_DIR)/rect_constants.h $(SRC_DIR)/rect_rtree_format.h \
             $(SRC_DIR)/kway_merge.h

$(BIN_DIR)/gpu_info: $(SRC_DIR)/gpu_info.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/str_rtree: $(SRC_DIR)/str_rtree_cuda.cu $(RECT_HDRS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/str_rtree_cpu: $(SRC_DIR)/str_rtree_cpu.cpp $(RECT_HDRS)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

$(BIN_DIR)/rect_rtree_query: $(SRC_DIR)/rect_rtree_query.cpp $(RECT_HDRS)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/gen_rects: $(SRC_DIR)/gen_rects.cpp $(RECT_HDRS)
	$(CXX) $(CXXFLAGS) $< -o $@

# ----------------------------------------------------------------
# Benchmarks (see bench/ and the results section of README.md)
# ----------------------------------------------------------------
BENCH_DIR := bench

.PHONY: bench
bench: $(BIN_DIR)/bench_sort_crossover $(BIN_DIR)/bench_pinned_memory \
       $(BIN_DIR)/bench_context_init $(BIN_DIR)/bench_kway_merge

$(BIN_DIR)/bench_sort_crossover: $(BENCH_DIR)/bench_sort_crossover.cu $(RECT_HDRS)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $< -o $@

$(BIN_DIR)/bench_pinned_memory: $(BENCH_DIR)/bench_pinned_memory.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/bench_context_init: $(BENCH_DIR)/bench_context_init.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR)/bench_kway_merge: $(BENCH_DIR)/bench_kway_merge.cu $(RECT_HDRS)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -Xcompiler -pthread $< -o $@

# ----------------------------------------------------------------
# Animation (see animation/README.md)
# ----------------------------------------------------------------

# Builds a deliberately tiny tree with the REAL binaries, checks the Python
# reference in str_reference.py against it page by page, and only then renders.
# The animation can therefore never show an algorithm the code does not run.
.PHONY: render
render: $(BIN_DIR)/gen_rects $(BIN_DIR)/str_rtree
	@mkdir -p $(ANIM_DIR)/data
	$(BIN_DIR)/gen_rects 60 $(ANIM_DIR)/data/rects60.bin \
	    --extent 100 --min-size 1 --max-size 3
	$(BIN_DIR)/str_rtree $(ANIM_DIR)/data/rects60.bin $(ANIM_DIR)/data/tree60.bin \
	    --fill-leaf 0.03 --fill-internal 0.03
	. $(MANIM_VENV)/bin/activate && \
	  python $(ANIM_DIR)/str_reference.py --validate $(ANIM_DIR)/data/tree60.bin && \
	  manim -q$(MANIM_Q) --media_dir $(ANIM_DIR)/media \
	        $(ANIM_DIR)/str_animation.py STRAlgorithm

# ----------------------------------------------------------------
# Utility targets
# ----------------------------------------------------------------
.PHONY: clean clean-tmp clean-data clean-anim help

# Remove compiled binaries (keeps directories)
clean:
	rm -f $(BIN_DIR)/gpu_info \
	      $(BIN_DIR)/str_rtree $(BIN_DIR)/str_rtree_cpu \
	      $(BIN_DIR)/rect_rtree_query $(BIN_DIR)/gen_rects \
	      $(BIN_DIR)/bench_sort_crossover $(BIN_DIR)/bench_pinned_memory \
	      $(BIN_DIR)/bench_context_init $(BIN_DIR)/bench_kway_merge

# Remove intermediate sort files left in tmp/
clean-tmp:
	find $(TMP_DIR) -name "*.bin" -o -name "*.tmp" | xargs rm -f

# Remove dataset files in data/
clean-data:
	find $(DATA_DIR) -name "*.bin" | xargs rm -f

# Remove rendered video and the animation's generated dataset
clean-anim:
	rm -rf $(ANIM_DIR)/media
	find $(ANIM_DIR)/data -name "*.bin" -delete 2>/dev/null || true

# ----------------------------------------------------------------
# Help
# ----------------------------------------------------------------
help:
	@echo ""
	@echo "  make                 Build all executables -> bin/"
	@echo "  make bench           Build the microbenchmarks"
	@echo "  make render          Render the STR animation"
	@echo ""
	@echo "  make clean           Remove compiled binaries"
	@echo "  make clean-tmp       Remove intermediate files in tmp/"
	@echo "  make clean-data      Remove dataset files in data/"
	@echo "  make clean-anim      Remove the rendered video and its dataset"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/gpu_info                    Suggested values for src/rect_constants.h"
	@echo "  ./bin/gen_rects <N> [out.bin] [--clustered K]"
	@echo "  ./bin/str_rtree     data/rects.bin data/tree.bin [--fill-leaf F] [--merge-threads N]"
	@echo "  ./bin/str_rtree_cpu data/rects.bin data/tree.bin --threads N   (CPU control group)"
	@echo "  ./bin/rect_rtree_query data/tree.bin verify data/rects.bin"
	@echo "  ./bin/rect_rtree_query data/tree.bin bench  data/rects.bin 0.01 20"
	@echo ""
	@echo "Benchmarks / tests:"
	@echo "  ./bin/bench_sort_crossover 24 5   GPU vs CPU sort break-even"
	@echo "  ./bin/bench_pinned_memory 3       Pinned alloc cost vs H2D bandwidth"
	@echo "  ./bin/bench_context_init          CUDA start-up cost (run cold, then warm)"
	@echo "  ./bench/sweep_pinned_cap.sh <input.bin>"
	@echo "  ./bin/bench_kway_merge --n 16 64 --k 4 16 --disk"
	@echo "  ./bench/compare_gpu_cpu.sh data/rects.bin   GPU vs CPU control group"
	@echo "  ./bench/test_correctness.sh [N]   Full correctness matrix"
	@echo ""
	@echo "Animation:"
	@echo "  make render                       Render the STR animation to animation/media"
	@echo "  make render MANIM_Q=l             Fast draft render"
	@echo "  make clean-anim                   Remove the rendered video and its dataset"
	@echo ""

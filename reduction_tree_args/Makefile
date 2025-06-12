# File: Makefile

# Compiler and flags
NVCC        = nvcc
TARGET      = reduction_tree_args
SRC         = reduction_tree_args.cu
FLAGS       = -O2 -arch=sm_89

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

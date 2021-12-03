CC := gcc
CFLAGS := -Wall -fopenmp
TARGETS := main

.PHONY: all clean
all: $(TARGETS)
clean:
	rm -rf $(TARGETS)

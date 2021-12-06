CC = gcc
BUILD_DIR = build
CMAKE_BUILD_TYPE ?= Debug

.PHONY: build configure clean

build: configure
	cmake --build $(BUILD_DIR)

configure:
	cmake -B $(BUILD_DIR) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

clean:
	rm -rf $(BUILD_DIR)

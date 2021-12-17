CC = gcc
PYTHON = python3

BUILD_DIR = build
CMAKE_BUILD_TYPE ?= Debug

WEIGHTS_PATH = data/weights.txt
IMAGE_PATH = data/image.txt
OUTPUT_FILE = data/output.txt
ORIGIN_IMAGE_PATH = data/image.png
ORIGIN_IMAGE_URL = https://s3.amazonaws.com/model-server/inputs/kitten.jpg

MAKEFLAGS += --no-print-directory # suppress entering or leaving directory messages

.PHONY: build configure clean run data

build: configure
	cmake --build $(BUILD_DIR)

configure:
	cmake -B $(BUILD_DIR) -DCMAKE_C_COMPILER=$(CC) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

clean:
	rm -rf $(BUILD_DIR)

run: build data
	$(BUILD_DIR)/vgg11_bn $(WEIGHTS_PATH) $(IMAGE_PATH) $(OUTPUT_FILE)

data: $(WEIGHTS_PATH) $(IMAGE_PATH)

$(WEIGHTS_PATH): data/weights.py
	$(PYTHON) data/weights.py $(WEIGHTS_PATH)

$(IMAGE_PATH): data/image.py $(ORIGIN_IMAGE_PATH)
	$(PYTHON) data/image.py $(ORIGIN_IMAGE_PATH) $(IMAGE_PATH)

$(ORIGIN_IMAGE_PATH):
	curl -o $(ORIGIN_IMAGE_PATH) $(ORIGIN_IMAGE_URL)

# Makefile for building the cuda-fglt program
# Copyright (C) 2023  Alexandros Athanasiadis
#
# This file is part of cuda-fglt
#
# cuda-fglt is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# knn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# The Compilers for C, C++ and CUDA
CC=gcc
CXX=g++
NVCC=nvcc

# Packages used and Compiler flags
PACKAGES=cusparse

CFLAGS=$(shell pkgconf --cflags-only-other $(PACKAGES)) -O3
LDFLAGS=$(shell pkgconf --libs $(PACKAGES))

# The name of the executable
EXEC_NAME=cuda-fglt

# The various directories used in the project
BUILD_DIR ?= ./build
BIN_DIR ?= ./bin

SRC_DIR ?= ./src
EXTERNAL_DIR ?= ./external
TESTS_DIR ?= ./tests

TOOLS_DIR ?= ./tools

# Source files for the project
SRCS := $(notdir $(shell find $(SRC_DIR) -name *.c -o -name *.cpp -o -name *.cu))

# Test files for the project
TESTS := $(basename $(notdir $(shell find $(TESTS_DIR) -name *.c -o -name *.cpp -o -name *.cu)))

# External files
EXTERNAL := $(basename $(notdir $(shell find $(EXTERNAL_DIR) -name *.c -o -name *.cpp -o -name *.cu)))

# The object files and dependency files
OBJS := $(patsubst %.c, $(BUILD_DIR)/%.o, $(filter-out $(EXEC_NAME).cpp, $(SRCS)))
OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(OBJS))
OBJS := $(patsubst %.cu, $(BUILD_DIR)/%.o, $(OBJS))

TEST_OBJS := $(patsubst %, $(BUILD_DIR)/%.o, $(TESTS))
EXTERNAL_OBJS := $(patsubst %, $(BUILD_DIR)/%.o, $(EXTERNAL))

DEPS := $(patsubst %.o, %.d, $(OBJS) $(TEST_OBJS) $(EXTERNAL_OBJS))

# Include directories for the compiler
INC_FLAGS := $(addprefix -I,$(SRC_DIR) $(EXTERNAL_DIR)) $(shell pkgconf --cflags-only-I $(PACKAGES))

# Setting VPATH to include the source file directories
VPATH_DIRS := $(shell find $(SRC_DIR) $(EXTERNAL_DIR) -type d)
VPATH=$(TESTS_DIR):$(shell tr ' ' ':' <<< '$(VPATH_DIRS)')

# C preprocessor flags
CPPFLAGS=$(INC_FLAGS) -MMD -MP

# CUDA compiler flags
NVCCFLAGS=-arch=sm_50

.PHONY: default all test tests clean

default: $(BIN_DIR)/$(EXEC_NAME)

all: $(BIN_DIR)/$(EXEC_NAME) tests

tests: $(addprefix $(BIN_DIR)/,$(TESTS))

$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(OBJS) $(EXTERNAL_OBJS) | $(BIN_DIR)
	$(NVCC) $(CPPFLAGS) $(CFLAGS) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

.PRECIOUS: $(BUILD_DIR)/%.o
$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(CFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Creating the directories
$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

# Remove all the produced files from the directory
clean: 
	rm -rf $(BIN_DIR) $(BUILD_DIR)

-include $(DEPS)

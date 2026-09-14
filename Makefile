CC = gcc
CFLAGS = -Wall -Wextra -O3 -ffast-math -funroll-loops -Iinclude
TARGET = bin/matrix_mul
SRCS = src/matrix_mul.c src/main.c
OBJS = $(SRCS:src/%.c=build/%.o)

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) -lm

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

build:
	mkdir -p build

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build bin

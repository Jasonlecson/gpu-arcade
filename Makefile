CC = gcc
CFLAGS = -O2 -Wall -I.
TARGET = gpu_arcade

SRCS = main.c src/game_snake.c src/game_tetris.c src/game_life.c src/game_pong.c src/game_breakout.c

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
  LDFLAGS = -lOpenCL -lncursesw -lm
else ifeq ($(UNAME_S),Darwin)
  LDFLAGS = -framework OpenCL -lncurses -lm
else
  LDFLAGS = -lOpenCL
endif

all: $(TARGET)

$(TARGET): $(SRCS) src/common.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET).exe

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

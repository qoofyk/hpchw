## Compiler, tools and options
CC      =  nvcc
CCFLAGS = -O3

## Files
OBJECTS = matmul.cu

TARGET  = matmul

## Implicit rules

## Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) -lcublas $(CCFLAGS)

clean:
	rm -f $(TARGET)
	rm -f *~ core

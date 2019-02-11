## Compiler, tools and options
CC      =  nvcc
FC      =  nvcc
LINK    =  nvcc
CCFLAGS = -O3

## Libraries
LIBS =  -lm -lcublas

## Files
OBJECTS = naive.o

TARGET  = naive

## Implicit rules
.SUFFIXES: .cu
.cu.o:
	$(CC) -c $(CCFLAGS) $<

## Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) -o $@ $(OBJECTS) $(CCFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)
	rm -f *~ core

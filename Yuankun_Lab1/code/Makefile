## Compiler, tools and options
CC      = cc
CCFLAGS = -O3

## Files
OBJECTS = dgemm.o
TARGET  = dgemm

## Implicit rules
.SUFFIXES: .c

.c.o:
	$(CC) -c $(CCFLAGS) $<

## Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *~ core

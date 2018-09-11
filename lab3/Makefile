## Compiler, tools and options
CC      = cc
FC      = ftn
LINK    = ftn
CCFLAGS = -O3

## Files
OBJECTS = main.o fun.o summa.o
TARGET  = summa

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

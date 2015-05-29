CC = gcc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -g -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
					 -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
					 -Wno-parentheses -Wsequence-point


GKINCHOME=/Users/mohitsharma/dev/gklib/trunk
GKLIBHOME=/Users/mohitsharma/dev/gklib/trunk/build/Darwin-x86_64/

#Standard Libraries
STDLIBS = -fopenmp -lpthread -lm

#external libraries
EXT_LIBS = -lGKlib
EXT_LIBS_DIR= -L$(GKLIBHOME)

LIBSDIR= -L.. -I/usr/include -L$(GKLIBHOME)
#LIBS = $(STDLIBS) $(EXT_LIBS)
#LIBS = -lm $(EXT_LIBS)
LIBS = -lm 

#CFLAGS = $(COPTIONS)  $(OPTFLAGS) -I$(GKINCHOME)
#CFLAGS = $(COPTIONS)  $(OPTFLAGS) 
CFLAGS = $(COPTIONS)   
#CFLAGS=  -I. $(OPTFLAGS) -I$(GKINCHOME) 


setrec: io.o datastruct.o setrec.o 
	$(CC) $^ $(CFLAGS) $(LIBS) $(LIBSDIR) -o $@


clean: 
	rm -f *.o setrec

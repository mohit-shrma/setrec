CC = /usr/bin/gcc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE

COPTIONS = -g -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
					 -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
					 -Wno-parentheses -Wsequence-point 

GKINCHOME=/home/grad02/mohit/George/GKlib/trunk
GKLIBHOME=/home/grad02/mohit/George/GKlib/trunk/build/Linux-x86_64/

#export LD_LIBRARY_PATH=/export/scratch/mohit/lib/gsl/lib
GSLINC=/export/scratch/mohit/lib/gsl/include
GSLLIB=/export/scratch/mohit/lib/gsl/lib

#Standard Libraries
STDLIBS = -fopenmp -lpthread -lm

#external libraries
EXT_LIBS = -lGKlib -lgsl -lgslcblas
EXT_LIBS_DIR = -L$(GKLIBHOME) -L$(GSLLIB) 

#LIBSDIR= -L.. -I/usr/include 
LIBSDIR= -L.. -I/usr/include 
#LIBS = $(STDLIBS) 
#LIBS = -lm 
LIBS = $(STDLIBS) $(EXT_LIBS) 

CFLAGS = $(COPTIONS)  $(OPTFLAGS) -I$(GKINCHOME) -I$(GSLINC)  
#CFLAGS = $(COPTIONS) -I$(GKINCHOME) -I$(GSLINC) 
#CFLAGS=  -I. $(OPTFLAGS) 

setrec: model.o modelItemMatFac.o modelMajority.o set.o util.o io.o datastruct.o setrec.o 
	$(CC) $^ $(CFLAGS) $(LIBS) $(LIBSDIR) $(EXT_LIBS_DIR) -o $@


clean: 
	rm -f *.o setrec

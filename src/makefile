CC = gcc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -g -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
					 -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
					 -Wno-parentheses -Wsequence-point


#Standard Libraries
STDLIBS = -fopenmp -lpthread -lm

#external libraries

LIBSDIR= -L.. -I/usr/include 
#LIBS = $(STDLIBS) 
#LIBS = -lm 
LIBS = -lm 

#CFLAGS = $(COPTIONS)  $(OPTFLAGS) 
#CFLAGS = $(COPTIONS)  $(OPTFLAGS) 
CFLAGS = $(COPTIONS)   
#CFLAGS=  -I. $(OPTFLAGS) 


setrec: model.o modelMajority.o  modelAddSim.o modelNoSim.o modelHingeProdSim.o modelCoOccSim.o modelItemMatFac.o modelRand.o modelBaseline.o modelSim.o modelWOSim.o set.o util.o io.o datastruct.o setrec.o 
	$(CC) $^ $(CFLAGS) $(LIBS) $(LIBSDIR) -o $@


clean: 
	rm -f *.o setrec

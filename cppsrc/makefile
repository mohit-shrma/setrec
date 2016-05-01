CXX=g++
RM=rm -f

GKINCHOME=/home/grad02/mohit/George/GKlib/trunk
GKLIBHOME=/home/grad02/mohit/George/GKlib/trunk/build/Linux-x86_64/
EIGENPATH=/home/grad02/mohit/lib/eigen
GSLINC=/home/grad02/mohit/dev/gsl/include
GSLLIB=/home/grad02/mohit/dev/gsl/lib

#Standard Libraries
STDLIBS=-lm -lpthread 

#external libraries
EXT_LIBS=-lGKlib -lgsl -lgslcblas
EXT_LIBS_DIR=-L$(GKLIBHOME) -L$(GSLLIB)

CPPFLAGS=-g -O3 -Wall -std=c++11 -I$(GKINCHOME) -I$(EIGENPATH) -I$(GSLINC)
LDFLAGS=-g
LDLIBS=$(STDLIBS) $(EXT_LIBS_DIR) $(EXT_LIBS)  

SRCS=io.cpp util.cpp mathUtil.cpp Model.cpp ModelAverage.cpp ModelAverageWBias.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: setrec

setrec: $(OBJS)
	$(CXX) $(LDFLAGS) -o setrec $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) tool

include .depend




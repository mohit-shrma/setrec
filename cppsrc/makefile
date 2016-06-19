CXX=g++
RM=rm -f

GKINCHOME=/home/grad02/mohit/George/GKlib/trunk
GKLIBHOME=/home/grad02/mohit/George/GKlib/trunk/build/Linux-x86_64/
EIGENPATH=/home/grad02/mohit/lib/eigen
GSLINC=/home/grad02/mohit/dev/gsl/include
GSLLIB=/home/grad02/mohit/dev/gsl/lib
SVDLIBPATH=/home/grad02/mohit/lib/SVDLIBC

#Standard Libraries
STDLIBS=-lm -lpthread 

#external libraries
EXT_LIBS=-lGKlib -lgsl -lgslcblas -lsvd
EXT_LIBS_DIR=-L$(GKLIBHOME) -L$(GSLLIB) -L$(SVDLIBPATH)  

CPPFLAGS=-g -O3 -Wall -std=c++11 -I$(GKINCHOME) -I$(EIGENPATH) -I$(GSLINC) -I$(SVDLIBPATH)
#CPPFLAGS=-g -Wall -std=c++11 -I$(GKINCHOME) -I$(EIGENPATH) -I$(GSLINC) -I$(SVDLIBPATH)
LDFLAGS=-g
LDLIBS=$(STDLIBS) $(EXT_LIBS_DIR) $(EXT_LIBS)  

SRCS=io.cpp util.cpp svdFrmsvdlib.cpp mathUtil.cpp Model.cpp ModelAverage.cpp ModelMFWBias.cpp ModelAverageWBias.cpp ModelAverageWSetBias.cpp ModelAverageWPart.cpp ModelBaseline.cpp ModelAverageBiasesOnly.cpp ModelAverageWGBias.cpp ModelItemAverage.cpp ModelAverageSetBiasWPart.cpp  ModelAverageHingeWBias.cpp ModelAverageHinge.cpp ModelAverageLogWBias.cpp ModelAverageBPR.cpp ModelAverageBPRWBias.cpp ModelAverageBPRWBiasTop.cpp  ModelBPR.cpp ModelBPRTop.cpp main.cpp
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




#ifndef _IO_H_
#define _IO_H_

#include <stdio.h>
#include <string.h>
#include "stdlib.h"
#include "datastruct.h"

void loadData(Data *data, Params *params);
void writeData(Data *data);
void RatingSet_load(RatingSet *ratSet, char *ip, int setSize);
void RatingSet_write(RatingSet *ratSet);

#endif


#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "model.h"

typedef struct{
  Model proto;
} ModelBPR;


void ModelBPR_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelBPR(Data *data, Params *params, float *valTest);

#endif




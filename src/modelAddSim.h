#ifndef _MODEL_ADDSIM_H_
#define _MODEL_ADDSIM_H_

#include "model.h"

typedef struct {
  Model proto;
  float *simCoeff;
} ModelAddSim;

void ModelAddSim_init(void *self, Params *params);
void ModelAddSim_free(void *self);
float ModelAddSim_objective(void *self, Data *data, float **sim);
void ModelAddSim_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelAddSim(Data *data, Params *params, float *valTest);

#endif



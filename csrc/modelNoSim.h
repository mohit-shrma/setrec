#ifndef _MODELNOSIM_H_
#define _MODELNOSIM_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelNoSim;


float ModelNoSim_objective(void *self, Data *data, float **sim);
void ModelNoSim_train(void *self, Data *data, Params *params, float **sim,
    float *valTest);
void modelNoSim(Data *data, Params *params, float *valTest);

#endif


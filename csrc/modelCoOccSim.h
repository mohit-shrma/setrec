#ifndef _MODEL_CO_OCC_H_
#define _MODEL_CO_OCC_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelCoOccSim;

float ModelCoOccSim_objective(void *self, Data *data, float **sim);
void ModelCoOccSim_train(void *self, Data *data, Params *params, float **sim, float *valTest);
void modelCoOccSim(Data *data, Params *params, float *valTest);

#endif

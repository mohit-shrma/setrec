#ifndef _MODELSIM_H_
#define _MODELSIM_H_

#include "model.h"

typedef struct {
  Model proto;
  //add below any model specific implementation
} ModelSim;

float ModelSim_objective(void *self, Data *data);
float ModelSim_setScore(void *self, int user, int *set, int setSz, float **sim);
void ModelSim_train(void *self, Data *data, Params *params, float **sim);
void modelSim(Data *data, Params *params);

#endif

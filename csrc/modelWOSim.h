#ifndef _MODELWOSIM_H_
#define _MODELWOSIM_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelWOSim;

float ModelWOSim_objective(void *self, Data *data, float **sim);
float ModelWOSim_setScore(void *self, int user, int *set, int setSz, float** sim);
void ModelWOSim_train(void *self, Data *data, Params *params, float** sim, float* valTest);
void modelWOSim(Data *data, Params *params, float* valTest);
#endif

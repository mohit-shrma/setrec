#ifndef _MODEL_MAJORITY_H_
#define _MODEL_MAJORITY_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelMajority;

float ModelMajority_setScore(void *self, int u, int *set, int setSz, 
    float **sim);
float ModelMajority_objective(void *self, Data *data, float **sim);
void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelMajority(Data *data, Params *params, float *valTest);

#endif

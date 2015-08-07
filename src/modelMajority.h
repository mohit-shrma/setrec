#ifndef _MODEL_MAJORITY_H_
#define _MODEL_MAJORITY_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelMajority;

void ModelMajority_init(void *self, Params *params);
void ModelMajority_free(void *self);
float ModelMajority_objective(void *self, Data *data, float **sim);
void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelMajority(Data *data, Params *params, float *valTest);

#endif

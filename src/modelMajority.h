#ifndef _MODEL_MAJORITY_H_
#define _MODEL_MAJORITY_H_

#include "model.h"

typedef struct {
  Model proto;
  float constrainWt;
  float rhoRMS; //AdaDelta
  float epsRMS; //AdaDelta
} ModelMajority;

float ModelMajority_setScore(void *self, int u, int *set, int setSz, 
    float **sim);
float ModelMajority_objective(void *self, Data *data, float **sim);
void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest);
void modelMajority(Data *data, Params *params, ValTestRMSE *valTest);

#endif

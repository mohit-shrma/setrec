#ifndef _MODEL_MAJORITY_H_
#define _MODEL_MAJORITY_H_

#include "model.h"

typedef struct {
  Model proto;
  float constrainWt;
  float rhoRMS; //AdaDelta
  float epsRMS; //AdaDelta
  float momentum; //SGD w momentum
  float beta1;
  float beta2;
} ModelMajority;

extern Model ModelMajorityProto;

float ModelMajority_setScore(void *self, int u, int *set, int setSz, 
    float **sim);
float ModelMajority_objective(void *self, Data *data, float **sim);
void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest);
void modelMajority(Data *data, Params *params, ValTestRMSE *valTest);
float ModelMajority_learnRateSearch(void *self, Data *data, ItemRat **itemrats,
    float *sumitemlatfac, float *igrad, float *ugrad, int seed);
#endif

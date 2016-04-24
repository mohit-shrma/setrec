#ifndef _MODEL_MAJSIGMOID_H_
#define _MODEL_MAJSIGMOID_H_

#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS; //rmsProp
  float regUm;
  float g_k; //global steepness parameter
  //TODO: from commandline
  float regG_k; //global steepness regularization
} ModelAvgSigmoid;

extern Model ModelAvgSigmoidProto;
float ModelAvgSigmoid_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelAvgSigmoid_objective(void *self, Data *data, float **sim);
void ModelAvgSigmoid_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelAvgSigmoid(Data *data, Params *params, ValTestRMSE *valTest);

#endif

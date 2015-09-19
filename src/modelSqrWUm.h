#ifndef _MODEL_SQR_UM_H_
#define _MODEL_SQR_UM_H_

#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS;
  float regUm;
} ModelSqrWUm;

float ModelSqrWUm_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelSqrWUm_objective(void *self, Data *data, float **sim);
void ModelSqrWUm_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelSqrWUm(Data *data, Params *params, ValTestRMSE *valTest);

#endif



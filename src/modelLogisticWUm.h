#ifndef _MODEL_LOG_UM_H_
#define _MODEL_LOG_UM_H_

#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS;
  float regUm;
} ModelLogisticWUm;

float ModelLogisticWUm_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelLogisticWUm_objective(void *self, Data *data, float **sim);
void ModelLogisticWUm_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelLogisticWUm(Data *data, Params *params, ValTestRMSE *valTest);

#endif

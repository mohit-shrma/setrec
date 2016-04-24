#ifndef _MODEL_LOG01_UM_H_
#define _MODEL_LOG01_UM_H_

#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS;
  float regUm;
} ModelLogistic01WUm;

float ModelLogistic01WUm_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelLogistic01WUm_objective(void *self, Data *data, float **sim);
void ModelLogistic01WUm_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelLogistic01WUm(Data *data, Params *params, ValTestRMSE *valTest);

#endif

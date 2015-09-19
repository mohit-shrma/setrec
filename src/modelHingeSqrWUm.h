#ifndef _MODEL_HINGE_SQR_UM_H_
#define _MODEL_HINGE_SQR_UM_H_
 
#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS;
  float regUm;
} ModelHingeSqrWUm;


float ModelHingeSqrWUm_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelHingeSqrWUm_objective(void *self, Data *data, float **sim);
void ModelHingeSqrWUm_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelHingeSqrWUm(Data *data, Params *params, ValTestRMSE *valTest);

#endif

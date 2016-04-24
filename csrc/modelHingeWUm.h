#ifndef _MODEL_HINGE_UM_H_
#define _MODEL_HINGE_UM_H_
 
#include "model.h"

typedef struct {
  Model proto;
  float *u_m;
  float rhoRMS;
  float regUm;
} ModelHingeWUm;

extern Model ModelHingeWUmProto;

float ModelHingeWUm_setScore(void *self, int u, int *set, int setSz, 
    float**sim);
float ModelHingeWUm_objective(void *self, Data *data, float **sim);
void ModelHingeWUm_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest);
void modelHingeWUm(Data *data, Params *params, ValTestRMSE *valTest);

#endif

#ifndef _MODEL_ITEM_MATFAC_H
#define _MODEL_ITEM_MATFAC_H 

#include "model.h"
#include "modelMajority.h"


typedef struct {
  Model proto;
} ModelItemMatFac;

extern Model ModelMajorityProto;

void ModelItemMatFac_train(void *self, Data *data, Params *params, float **sim, ValTestRMSE *valTest);
float ModelItemMatFac_validationErr(void *self, Data *data, float **sim);
float ModelItemMatFac_objective(void *self, Data *data, float **sim);
void modelItemMatFac(Data *data, Params *params, ValTestRMSE *valTest);

#endif

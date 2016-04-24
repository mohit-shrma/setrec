#ifndef _MODEL_MAJFEAT_H_
#define _MODEL_MAJFEAT_H_

#include "model.h"

typedef struct {
  Model proto;
  float rhoRMS; 
  float epsRMS;
  float constrainWt;
} ModelMajFeat;

extern Model ModelMajFeatProto;
void modelMajFeat(Data *data, Params *params, ValTestRMSE *valTest);

#endif

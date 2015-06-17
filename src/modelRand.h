#ifndef _MODELRAND_H_
#define _MODELRAND_H_

#include "model.h"

typedef struct {
  Model proto;
  //add below model specific implementation
  
  //average label of sets
  float avgLabel;

  //no. of positive and negative sets
  int posCount, negCount;

} ModelRand;

float ModelRand_testErr(void *Self, Data *data, float **sim);
float ModelRand_validationErr(void *self, Data *data, float **sim);
void ModelRand_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelRand(Data *data, Params *params, float *valTest);

#endif

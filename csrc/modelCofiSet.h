#ifndef _MODEL_COFI_H_
#define _MODEL_COFI_H_

#include "model.h"

typedef struct {
  Model proto;
} ModelCofi;

float ModelCofi_setScore(void *self, int u, int *set, int setSz, 
    float **sim);
float ModelCofi_objective(void *self, Data *data, float **sim);
void ModelCofi_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest);
void modelCofi(Data *data, Params *params, float *valTest);

#endif


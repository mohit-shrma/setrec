#ifndef _MODELBASE_H_
#define _MODELBASE_H_

#include "model.h"

typedef struct {
  Model proto;
  //add below any model specific implementation
} ModelBase;

void ModelBase_train(void *self, Data *data, Params *params, float **sim);
float ModelBase_testErr(void *self, Data *data, float **sim);
float ModelBase_validationErr(void *self, Data *data, float **sim);
void modelBase(Data *data, Params *params);
#endif

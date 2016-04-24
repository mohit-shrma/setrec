#ifndef _MODELITEM_H_
#define _MODELITEM_H_

#include "model.h"

typedef struct {
  Model proto;
  //add below any model specific implementation
} ModelItem;


float ModelItem_objective(void *self, Data *data);
float ModelItem_setScore(void *self, int user, int *set, int setSz, float **sim);
void ModelItem_train(void *self, Data *data, Params *params, float **sim);

#endif

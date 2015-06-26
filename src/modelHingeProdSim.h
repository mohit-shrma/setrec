#ifndef _M_HINGE_PROD_SIM_
#define _M_HINGE_PROD_SIM_

#include "model.h"

typedef struct {
  Model proto;
} MHingeProdSim;

float MHingeProdSim_objective(void *self, Data *data, float **sim);  
float MHingeProdSim_setScore(void *self, int user, int *set, int setSz, float **sim);
void MHingeProdSim_train(void *self, Data *data, Params *params, float **sim, float *valTest);
void mHingeProdSim(Data *data, Params *params, float *valTest);
#endif

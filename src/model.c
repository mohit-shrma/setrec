#include "model.h"













void sgdUpdate() {
}

void model(Data *data, Params *params) {
 
  Model *model;


  model = (Model *) malloc(sizeof(Model));
  Model_init(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI)

  
  


  Model_free(model);
}




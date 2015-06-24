#include "modelCoOccSim.h"



void modelCoOccSim(Data *data, Params *params, float *valTest) {
  
  int i, j;
  float **sim = NULL;

  //allocate storage for model
  ModelCoOccSim *modelCoOccSim = NEW(ModelCoOccSim, 
      "set pred using co occ jaccard similarity");
  modelCoOccSim->_(init) (modelCoOccSim, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate);

  //allocate space for sim
  if (params->useSim) {
    sim = (float **) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  //compute jaccard similarities
  


  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }

  modelCoOccSim->_(free)(modelCoOccSim);
}


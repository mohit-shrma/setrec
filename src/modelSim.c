#include "model.h"


float setPref(int user, int *set, int setSz, Model *model) {
  
  int i, j;
  int item;
  float *itemsLatFac = NULL;
  float pref = 0.0;

  itemsLatFac = (float*) malloc(sizeof(float)*model->facDim);
  memset(itemsLatFac, 0, sizeof(float)*model->facDim);

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->facDim; k++) {
      itemsLatFac[k] += model->iFac[item][k];
    }
  }

  pref = (1.0/(float)setSz)*dotProd(model->uFac[user], itemsLatFac, 
      model->facDim); 

  free(itemsLatFac);
  
  return pref;
}


//TODO: different nae to avoid ambiguous match in model.c
float computeObjective(Data *data, Model *model) {

  int u, i, s, item, setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, setPref = 0;
  float uRegErr = 0, iRegErr = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //TODO: ambig
      setPref = setPref(u, set, setSz, model);
      

    }
  }



}





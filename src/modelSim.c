#include "model.h"

//r_us = 1/|s| u^T sum_{i \in s}<v_i>
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

//TODO: put in common place
float setSim(int *set, int setSz, Model *model, float **sim) {
  float setSim = 0.0;
  int i, j, nPairs;
  nPairs = (setSz * (setSz-1)) / 2;
  if (sim != NULL) {
    for (i = 0; i < setSz; i++) {
      for (j = i+1; j < setSz; j++) {
        setSim += sim[set[i]][set[j]];
      }
    }
  } else {
    for (i = 0; i < setSz; i++) {
      for (j = i+1; j < setSz; j++) {
        setSim += dotProd(model->iFac[set[i]], model->iFac[set[j]], 
            model->facDim);
      }
    }
  }
  setSim = setSim/nPairs;
  return setSim;
}


//TODO: different name to avoid ambiguous match in model.c
float computeObjective(Data *data, Model *model) {

  int u, i, s, item, setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0, setSim = 0;
  float uRegErr = 0, iRegErr = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      userSetPref = setPref(u, set, setSz, model);
      //set sim
      setSim = setSim(set, setSz, model, NULL);   
      diff = userSetPref - userSet->labels[s];
      rmse = diff*diff*setSim;
    }
    uRegErr += dotProd(model->uFac[u], model->uFac[u], model->facDim);
  }
  uRegErr = uRegErr*model->regU;

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->iFac[i], model->iFac[i], model->facDim); 
  }
  iRegErr *= model->regI;
  printf("\nObj: %f RMSE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr),
      rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


void computeUGrad(Model *model, int user, int *set, int setSz, 
  float avgSimSet, float r_us, float *sumItemLatFac, float *uGrad) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->facDim);
  
  temp = 2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->uFac[u], 
        sumItemLatFac));
  temp = temp * (-1.0/(setSz*1.0)) * avgSimSet;
  for (j = 0; j < model->facDim; j++) {
    uGrad[j] = sumItemLatFac[j]*temp;
  }

}


void computeIGrad(Model *model, int user, int item, int *set, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp, nPairs, comDiff;
  
  nPairs = (setSz * (setSz-1))/2;
  memset(iGrad, 0, sizeof(float)*model->facDim);
  
  comDiff = r_us - (1.0/(1.0*setSz))*dotProd(model->uFac[user], sumItemLatFac);
  temp = 2.0 * comDiff * (-1.0/(1.0*setSz)) * avgSimSet;
  for (j = 0; j < model->facDim; j++) {
    iGrad[j] = temp*model->uFac[user][j];
  }

  temp = comDiff*comDiff*(1.0/nPairs);
  for (j = 0; j < model->facDim; j++) {
    iGrad[j] += temp*(sumItemLatFac[j] - model->iFac[item][j]);
  }

}


void trainModel(Model *model, Data *data, Params *params, float **sim) {

  int iter, u, i, j, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us; //actual set preference
  float avgSetSim, temp, nPairs;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->facDim);
  float* iGrad = (float*) malloc(sizeof(float)*model->facDim);
  float* uGrad = (float*) malloc(sizeof(float)*model->facDim);
  

  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
      userSet = data->userSets[u];
      isTestValSet = 0;
      //select a non-test non-val set for user
      s = rand() % userSet->numSets;
      for (i = 0; i < userSet->szTestSet; i++) {
        if (s == userSet->testSets[i]) {
          isTestValSet = 1;
          break;
        }
      }
      
      for (i = 0; i < userSet->szValSet && !isTestValSet; i++) {
        if (s == userSet->valSets[i]) {
          isTestValSet = 1;
          break;
        }
      }
      
      if (isTestValSet) {
        continue;
      }
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      r_us = userSet->labels[s];
      avgSetSim = setSim(set, setSz, model, NULL); 
      nPairs = (setSz * (setSz - 1))/2
      //update user and latent factor
      
      memset(sumItemLatFac, 0, sizeof(float)*model->facDim);
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->facDim; j++) {
          sumItemLatFac[j] += model->iFac[item][j];
        }
      }
      
      //compute user gradient
      computeUGrad(model, u, set, setSz, avgSetSim, r_us, sumItemLatFac, uGrad);

      
      //TODO: update user + reg

      //compute  gradient of item in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGrad(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad);
        
        //TODO: update item + reg

      }

    }
  }

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


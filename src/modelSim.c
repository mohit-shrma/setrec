
#include "modelSim.h"

//r_us = 1/|s| u^T sum_{i \in s}<v_i>
float ModelSim_setScore(void *self, int user, int *set, int setSz, float **sim) {
  
  int i, j, k;
  int item;
  float *itemsLatFac = NULL;
  float pref = 0.0;
  ModelSim *model = self;

  itemsLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(itemsLatFac, 0, sizeof(float)*model->_(facDim));

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->_(facDim); k++) {
      itemsLatFac[k] += model->_(iFac)[item][k];
    }
  }

  pref = (1.0/(float)setSz)*dotProd(model->_(uFac)[user], itemsLatFac, 
      model->_(facDim)); 

  free(itemsLatFac);
  
  return pref;
}

//TODO: put in common place
float setSimilarity(int *set, int setSz, void *self, float **sim) {
  float setSim = 0.0;
  int i, j, nPairs;
  ModelSim *model = self;
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
        setSim += dotProd(model->_(iFac)[set[i]], model->_(iFac)[set[j]], 
            model->_(facDim));
      }
    }
  }
  setSim = setSim/nPairs;
  return setSim;
}

//TODO: need sim matrix in computing objective
//different name to avoid ambiguous match in model.c
float ModelSim_objective(void *self, Data *data) {

  int u, i, s, item, setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0, setSim = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelSim *model = self;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      //TODO
      userSetPref = model->_(setScore)(model, u, set, setSz, NULL);
      //set sim
      setSim = setSimilarity(set, setSz, model, NULL);   
      diff = userSetPref - userSet->labels[s];
      rmse = diff*diff*setSim;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  printf("\nObj: %f RMSE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr),
      rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


void computeUGrad(ModelSim *model, int user, int *set, int setSz, 
  float avgSimSet, float r_us, float *sumItemLatFac, float *uGrad) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  
  temp = 2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->_(uFac)[user], 
        sumItemLatFac, model->_(facDim)));
  temp = temp * (-1.0/(setSz*1.0)) * avgSimSet;
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = sumItemLatFac[j]*temp;
  }

}


void computeIGrad(ModelSim *model, int user, int item, int *set, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp, nPairs, comDiff;
  
  nPairs = (setSz * (setSz-1))/2;
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  
  comDiff = r_us - (1.0/(1.0*setSz))*dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim));
  temp = 2.0 * comDiff * (-1.0/(1.0*setSz)) * avgSimSet;
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = temp*model->_(uFac)[user][j];
  }

  temp = comDiff*comDiff*(1.0/nPairs);
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] += temp*(sumItemLatFac[j] - model->_(iFac)[item][j]);
  }

}


void ModelSim_train(void *self, Data *data, Params *params, float **sim) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us; //actual set preference
  float avgSetSim, temp; 
  ModelSim *model         = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));

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
      
      set       = userSet->uSets[s];
      setSz     = userSet->uSetsSize[s];
      r_us      = userSet->labels[s];
      avgSetSim = setSimilarity(set, setSz, model, NULL);
      //update user and latent factor
      
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      
      //compute user gradient
      computeUGrad(model, u, set, setSz, avgSetSim, r_us, sumItemLatFac, uGrad);
      //update user + reg
      for (k = 0; k < model->_(facDim); k++) {
        model->_(uFac)[u][k] -= model->_(learnRate)*(uGrad[k] + model->_(regU)*model->_(uFac)[u][k]);
      }

      //compute  gradient of item and updat their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGrad(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        for (k = 0; k < model->_(facDim); k++) {
          model->_(iFac)[item][k] -= model->_(learnRate)*(iGrad[k] + model->_(regI)*model->_(iFac)[item][k]);
        }
      }

    }

    //update sim
    model->_(updateSim)(model, sim);
    //objective check
    if (iter % 100 == 0) {
      model->_(objective)(model, data);
      //validation err
      model->_(validationErr) (model, data, sim);
    }
    
  }

  //get test eror
  model->_(testErr) (model, data, sim);

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


Model ModelSimProto = {
  .objective        = ModelSim_objective,
  .setScore         = ModelSim_setScore,
  .train            = ModelSim_train
};


void modelSim(Data *data, Params *params) {
 
  float **sim  = NULL;

  int i, j, k;

  //init random with seed
  srand(params->seed);

  //allocate storage for model
  ModelSim *modelSim = NEW(ModelSim, "set prediction model with sim");
  
  //allocate space for sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  //train model 
  modelSim->_(train)(modelSim, data,  params, sim);

  //free up allocated space
  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }
  modelSim->_(free)(modelSim);

}

#include "model.h"


void updateSim(float **sim, Model *model) {
  int i, j;
  for (i = 0; i < model->nItems; i++) {
    sim[i][i] = 1.0;
    for (j = i+1; j < model->nItems; j++) {
      sim[i][j] = sim[j][i] = dotProd(model->iFac[i], model->iFac[j], model->facDim);
    }
  }
}


float computeObjective(Data *data, Model *model) {
  
  int u, i, item;
  UserSets *userSet;
  float rmse = 0, diff = 0;
  float uRegErr = 0, iRegErr = 0;

  for (u = 0 ; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->items[i];
      diff = userSet->itemWts[i] - dotProd(model->uFac[u], model->iFac[item], model->facDim);
      rmse += diff*diff;
    }
    uRegErr += dotProd(model->uFac[u], model->uFac[u], model->facDim);
  }
  uRegErr *= model->regU;

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->iFac[i], model->iFac[i], model->facDim);
  }
  iRegErr *= model->regI;
  printf("\nObj: %f RMSE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


void trainModel(Model *model, Data *data, Params *params, float **sim) {
  int iter, u, i, j, k;
  int numItems, item, itemInd;
  float uTv, diff, Wui;
  UserSets *userSet;

  for (iter = 0; iter < params->maxIter; iter++) {
    //go over users
    //TODO: experiment with ordering i.e. item first then users
    for (u = 0; u < data->nUsers; u++) {
      
      userSet = data->userSets[u];

      //for each user select an item
      itemInd = rand() % userSet->nUserItems;
      i = userSet->items[itemInd]; //item
      Wui = userSet->itemWts[itemInd]; //user score on item

      //update model for u and i
      uTv = dotProd(model->uFac[u], model->iFac[i], model->facDim);
      
      //update user and item latent factor
      diff = Wui - uTv;
      for (k = 0; k < model->facDim; k++) {
        model->uFac[u][k] += model->learnRate*(diff*model->iFac[i][k] - model->regU*model->uFac[u][k]);
        model->iFac[i][k] += model->learnRate*(diff*model->uFac[u][k] - model->regI*model->iFac[i][k]);
      }

    }

    if (params->useSim) {
      //update sim
      updateSim(sim, model);
      
      //update W if using sim: decide here or above in loop
      UserSets_updWt(userSet, sim);
    }

    //objective check
    if (iter%1000 == 0) {
      computeObjective(data, model);
    }
  }

}

//TODO
float setRating() {
  int i;
  return 0.0;
}


//TODO
float validationErr(Model *model, Data *data) {
  return 0.0;
}

//TODO
float testErr(Model *model Data *data) {
  return 0.0;
}

void model(Data *data, Params *params) {
 
  Model *model = NULL;
  float **sim  = NULL;

  int i, j, k;

  //init random with seed
  srand(params->seed);

  //allocate storage for model
  model = (Model *) malloc(sizeof(Model));
  Model_init(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);

  //allocate space for sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  //initialize weights based on user sets
  for (i = 0; i < data->nUsers; i++) {
    UserSets_initWt(data->userSets[i]);
  }
  
  //train model 
  trainModel(model, data, params, sim);

  //test model


  //free up allocated space
  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }
  Model_free(model);
}




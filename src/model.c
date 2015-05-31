#include "model.h"

/*

gk_csr_t *createWeightMat(Data *data) {
  int i, j, nnz;
  gk_csr_t *W = NULL;
  
  //compute nnz
  nnz = 0;
  for (i = 0; i < data->nUSers; i++) {
    //TODO: verify nnz count with raw file
    nnz += data->userSets[i]->nUserItems;
  }
  
  W = gk_csr_Create();
  
  W->nrows = data->nUsers;
  W->ncols = data->nItems;

  W->rowptr = gk_zmalloc(W->nrows + 1, "rowptr");
  //TODO: find nnz 
  W->rowind = gk_imalloc(nnz, "rowind");
  W->rowval = gk_fsmalloc(nnz, 1.0, "rowval");

  k = 0; //ptr to non-zero values array or cols
  for (i = 0; i < data->nUsers; i++) {
    //start filling values at k
    W->rowptr[i] = k;
    for (j = 0; j < data->userSets[i]->nUserItems; j++) {
      //TODO: make sure you are filling in sorted order col ind
      W->rowind[k++] = data->userSets[i]->items[j];
    }
  }
  
  //set last ptr in row to (nnz + 1) 
  W->rowptr[i] = k;

  assert(k == nnz);

  return W;
}


void initMat(gk_csr_t *W, Data *data) {
  
  int i, j, k;
  int ii, jj, kk;

  int item, sizeItemSet, setInd;
  float score;
  UserSets *userSet = NULL;

  for (i = 0; data->nUsers; i++) {
    for (ii = W->rowptr[i]; ii < W->rowptr[i+1]; ii++) {
      item = W->rowind[ii];
      score = 0;
      userSet = data->userSets[i];
      sizeItemSet = data->userSets[i]->itemSetsSize[item];
      for (j = 0; j < sizeItemSet; j++) {
        setInd = userSet->itemSets[item][j]; 
        score += userSet->labels[setInd]/userSet->uSetsSize[setInd];
      }
      score = score/sizeItemSet;
      W->rowval[ii] = score;
    }
  }
  
}
*/

void updateSim(float **sim, Model *model) {
  int i, j;
  for (i = 0; i < model->nItems; i++) {
    sim[i][i] = 1.0;
    for (j = i+1; j < model->nItems; j++) {
      sim[i][j] = sim[j][i] = dotProd(model->iFac[i], model->iFac[j], 
                                        model->facDim);
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
  printf("\nRMSE: %f uRegErr: %f iRegErr: %f", rmse, uRegErr, iRegErr);
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
      i = userSets->items[itemInd]; //item
      Wui = userSets->itemWts[itemInd]; //user score on item

      //update model for u and i
      uTv = dotProd(uFac[u], iFac[i], facDim);
      
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
    if (iter%2 == 0) {
      computeObjective(data, model);
    }
  }

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
  gk_csr_Free(W);
}




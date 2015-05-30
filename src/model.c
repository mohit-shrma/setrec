#include "model.h"













void sgdUpdate() {
}



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


void trainModel(Model *model, Data *data, Params *params, gk_csr_t *W) {
  int iter, u, i, j, k;
  int numItems, item;
  float uTv, diff, Wui;
  UserSets *userSet;

  //TODO: initilize model latent factors

  for (iter = 0; iter < params->maxIter; iter++) {
    //go over users
    //TODO: experiment with ordering i.e. item first then users
    for (u = 0; u < data->nUsers; u++) {
      
      userSet = data->userSets[u];
      
      //for each user select an item
      i = userSets->items[rand()%userSet->nUserItems];
      
      //update model for u and i
      uTv = dotProd(uFac[u], iFac[i], facDim);
      
      for (ii = W->rowptr[u]; ii < W->rowptr[u+1]; ii++) {
        if (W->rowind[ii] == i) {
          Wui = W->rowval[ii];
          break;
        }
      }
      
      //make sure item is found
      assert(ii != W->rowptr[u+1]);
      
      //update user and item latent factor
      diff = Wui - uTv;
      for (k = 0; k < model->facDim; k++) {
        model->uFac[u][k] += model->learnRate*(diff*model->iFac[i][k] - model->regU*model->uFac[u][k]);
        model->iFac[i][k] += model->learnRate*(diff*model->uFac[u][k] - model->regI*model->iFac[i][k]);
      }

    }

    //update W: decide here or above in loop

    //update sim
    
    //objective check
  }

}


void model(Data *data, Params *params) {
 
  Model *model = NULL;
  float **sim  = NULL;
  gk_csr_t *W  = NULL;

  int i, j, k;

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

  //init random with seed
  srand(params->seed);
  
  //create user weights sparse matrix
  W = createWeightMat(data);

  //initialize weight matrix based on user sets
  initMat(W, data);
  
  //train model 
  

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




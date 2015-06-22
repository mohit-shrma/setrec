#include "modelItemMatFac.h"


void initUserItemWeights(Data *data) {

  int u, i, j, setInd;
  UserSets *userSet = NULL;
  ItemWtSets *itemWtSets = NULL;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    //assign score to items preferred by user
    for (i = 0; i < userSet->nUserItems; i++) {
      itemWtSets = userSet->itemWtSets[i];

      //init score
      itemWtSets->wt = 0;

      //go through the sets where item appears
      for (j = 0; j < itemWtSets->szItemSets; j++) {
        setInd = itemWtSets->itemSets[j];
        itemWtSets->wt += userSet->labels[setInd] / userSet->uSetsSize[setInd];
      }
      itemWtSets->wt = itemWtSets->wt / itemWtSets->szItemSets;
    }
  }

}


//NOTE: this is not exactly validation but ratio of rmse to matrix norm
float ModelItemMatFac_validationErr(void *self, Data *data, float **sim) {
  
  int u, i, item;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  float rmse = 0, matNorm = 0, ratio = 0, diff;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      diff = (userSet->itemWtSets[i]->wt - 
          dotProd(model->_(uFac)[u], model->_(iFac)[item], model->_(facDim)));
      rmse += diff*diff;
      matNorm += userSet->itemWtSets[i]->wt * userSet->itemWtSets[i]->wt;
    }
  }

  ratio = rmse / matNorm;

  return ratio;
}


float ModelItemMatFac_objective(void *self, Data *data) {
  
  int u, i, item;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  float rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      diff = (userSet->itemWtSets[i]->wt - 
          dotProd(model->_(uFac)[u], model->_(iFac)[item], model->_(facDim)));
      rmse += diff*diff;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim));
  }
  iRegErr = iRegErr*model->_(regI);
  
  obj = rmse + uRegErr + iRegErr;

  printf("\nObj: %f SE: %f uRegEr: %f iRegErr:%f", obj, rmse, uRegErr, iRegErr);

  return obj;
}


void ModelItemMatFac_computeUGrad(ModelItemMatFac *model, int user, int item, float r_ui, 
    float *uGrad) {
  int i;
  float diff = 0;
  
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  diff = r_ui - dotProd(model->_(uFac)[user], model->_(iFac)[item], 
      model->_(facDim));
  
  for (i = 0; i < model->_(facDim); i++) {
    uGrad[i] = 2*diff*-1*model->_(iFac)[item][i] + 
      model->_(regU)*model->_(uFac)[user][i]; 
  }

}


void ModelItemMatFac_computeIGrad(ModelItemMatFac *model, int user, int item, float r_ui,
    float *iGrad) {
  int i;
  float diff = 0;

  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  diff = r_ui - dotProd(model->_(uFac)[user], model->_(iFac)[item], 
      model->_(facDim));
  
  for (i = 0; i < model->_(facDim); i++) {
    iGrad[i] = 2*diff*-1*model->_(uFac)[user][i] +
      model->_(regI)*model->_(iFac)[item][i];
  }

}


void ModelItemMatFac_train(void *self, Data *data, Params *params, float **sim,
    float *valTest) {
  
  int u, i, j, iter, item;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  ItemWtSets *itemWtSets = NULL;
  float *iGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float *) malloc(sizeof(float)*model->_(facDim));

  model->_(objective) (model, data);

  for (iter = 0; iter < params->maxIter; iter++) {
    //update user and item latent factor pair
    for (u = 0; u < data->nUsers; u++) {
      
      userSet = data->userSets[u];
      
      //select an item from users' preferences
      itemWtSets = userSet->itemWtSets[rand()%userSet->nUserItems];
     
      //user gradient
      ModelItemMatFac_computeUGrad(model, u, itemWtSets->item, itemWtSets->wt, uGrad);

      //update user
      for (i = 0; i < model->_(facDim); i++) {
        model->_(uFac)[u][i] -= model->_(learnRate)*uGrad[i];
      }

      //item gradient
      ModelItemMatFac_computeIGrad(model, u, itemWtSets->item, itemWtSets->wt, iGrad);

      //update item
      for (i = 0; i < model->_(facDim); i++) {
        model->_(iFac)[itemWtSets->item][i] -= model->_(learnRate)*iGrad[i];
      }

    }

    //validation
    if (iter % VAL_ITER == 0) {
      valTest[0] = model->_(validationErr) (model, data, NULL);
      printf("\nIter:%d ErrToMat ratio:%f", iter, valTest[0]);
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      model->_(objective) (model, data);
    }

  }

  if (iter % OBJ_ITER == 0) {
    model->_(objective) (model, data);
  }
  
  valTest[0] = model->_(validationErr) (model, data, NULL);
  printf("\nIter:%d ErrToMat ratio:%f", iter, valTest[0]);
  
  free(iGrad);
  free(uGrad);
}


Model ModelItemMatFacProto = {
  .objective               = ModelItemMatFac_objective,
  .train                   = ModelItemMatFac_train,
  .validationErr           = ModelItemMatFac_validationErr
};


void modelItemMatFac(Data *data, Params *params, float *valTest) {
  
  //allocate storage for model
  ModelItemMatFac *modelItemMatFac = NEW(ModelItemMatFac, 
      "mat fac model for user-item ratings");
  modelItemMatFac->_(init) (modelItemMatFac, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate); 

  //initialize user-item weights
  initUserItemWeights(data);

  //train model
  modelItemMatFac->_(train)(modelItemMatFac, data, params, NULL, valTest);  

  modelItemMatFac->_(free)(modelItemMatFac);
}



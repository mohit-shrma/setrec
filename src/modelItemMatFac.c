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


float ModelItemMatFac_objective(void *self, Data *data, float **sim) {
  
  int u, i, item;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  float rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff, rat;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      rat = dotProd(model->_(uFac)[u], model->_(iFac)[item], model->_(facDim)); 
      diff = (userSet->itemWtSets[i]->wt - rat);
      //printf("\nU %d %d %f %f %f", u, item, rat, userSet->itemWtSets[i]->wt, diff);
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

  //printf("\nObj: %f SE: %f uRegEr: %f iRegErr:%f", obj, rmse, uRegErr, iRegErr);

  return obj;
}


float ModelItemMatFac_objectiveSamp(void *self, Data *data, float **sim) {
  
  int u, i, ii, item, itemInd;
  gk_csr_t *trainMat = data->trainMat;
  ModelItemMatFac *model = self;
  float rmse = 0, uRegErr = 0, iRegErr = 0, obj = 0, diff;
  float itemRat;

  for (u = 0; u < data->nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      itemRat = trainMat->rowval[ii];
      diff = itemRat - dotProd(model->_(uFac)[u], model->_(iFac)[item], model->_(facDim));
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

  //printf("\nObj: %f SE: %f uRegEr: %f iRegErr:%f", obj, rmse, uRegErr, iRegErr);

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
      2.0*model->_(regU)*model->_(uFac)[user][i]; 
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
      2.0*model->_(regI)*model->_(iFac)[item][i];
  }

}


float ModelItemMatFac_majSetScore(void *self, int u, int *set, int setSz, 
    float **sim) {
   
  int i, item, majSz;
  float r_us_est = 0;
  ModelItemMatFac *model = self;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }
  
  for (i = 0 ; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }
  
  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }
  
  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;

    if (i > 0) {
      assert(itemRats[i]->rating <= itemRats[i-1]->rating);
    }
  }
  r_us_est = r_us_est/majSz;

  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);

  return r_us_est;
}


void ModelItemMatFac_train(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest) {
  
  int u, i, j, iter, item, subIter, nnz;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  ItemWtSets *itemWtSets = NULL;
  float *iGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float prevVal = 0, prevObj = 0;

  model->_(objective) (model, data, sim);

  nnz = 0;
  for (u = 0; u < params->nUsers; u++) {
    nnz += data->userSets[u]->nUserItems;
  }
  
  printf("\nNNZ = %d", nnz);

  for (iter = 0; iter < params->maxIter; iter++) {
    //update user and item latent factor pair
    for (subIter = 0; subIter < nnz; subIter++) {
      //sample u
      u = rand() % params->nUsers;
      userSet = data->userSets[u];
      
      //select an item from users' preferences
      itemWtSets = userSet->itemWtSets[rand()%userSet->nUserItems];
     
      //user gradient
      ModelItemMatFac_computeUGrad(model, u, itemWtSets->item, itemWtSets->wt, uGrad);

      //update user
      for (i = 0; i < model->_(facDim); i++) {
        model->_(uFac)[u][i] -= (model->_(learnRate)/(1.0 + model->_(learnRate)*model->_(regU)*iter))*uGrad[i];
      }

      //item gradient
      ModelItemMatFac_computeIGrad(model, u, itemWtSets->item, itemWtSets->wt, iGrad);

      //update item
      for (i = 0; i < model->_(facDim); i++) {
        model->_(iFac)[itemWtSets->item][i] -= (model->_(learnRate)/(1.0 + model->_(learnRate)*model->_(regI)*iter))*iGrad[i];
      }
    }


    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->valItemsRMSE = model->_(indivItemCSRErr) (model, data->valMat, NULL);
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter:%d Obj: %.10e valRMSE:%f", iter, valTest->setObj, 
          valTest->valItemsRMSE);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    
  }

  valTest->valItemsRMSE = model->_(indivItemCSRErr) (model, data->valMat,
      NULL);
  printf("\nValidation items error: %f", valTest->valItemsRMSE);

  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }
 
  valTest->trainSetRMSE = model->_(trainErr) (model, data, NULL);
  printf("\nTrain set error(matfac): %f", valTest->trainSetRMSE);

  valTest->testSetRMSE = model->_(testErr)(model, data, NULL); 
  printf("\nTest set error(matfac): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(matfac): %f", valTest->testItemsRMSE);
 
  //get train error
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(matfac): %f", valTest->trainItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));
  
  free(iGrad);
  free(uGrad);
}


void ModelItemMatFac_trainRMSProp(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest) {
  
  int u, i, j, iter, bestIter, item, subIter, nnz;
  UserSets *userSet          = NULL;
  ModelItemMatFac *model     = self;
  ModelItemMatFac *bestModel = NULL;
  ItemWtSets *itemWtSets     = NULL;

  bestModel = NEW(ModelItemMatFac, "model that achieved the lowest obj");
  bestModel->_(init) (bestModel, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate);

  float *iGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  float prevVal = 0, prevObj = 0, bestObj = -1;

  model->_(objective) (model, data, sim);

  nnz = 0;
  for (u = 0; u < params->nUsers; u++) {
    nnz += data->userSets[u]->nUserItems;
  }
  
  printf("\nNNZ = %d", nnz);

  for (iter = 0; iter < params->maxIter; iter++) {
    //update user and item latent factor pair
    for (subIter = 0; subIter < nnz; subIter++) {
     
      //sample u
      u = rand() % params->nUsers;
      userSet = data->userSets[u];
      
      //select an item from users' preferences
      itemWtSets = userSet->itemWtSets[rand()%userSet->nUserItems];
     
      //user gradient
      ModelItemMatFac_computeUGrad(model, u, itemWtSets->item, itemWtSets->wt, uGrad);

      //update user
      for (j = 0; j < model->_(facDim); j++) {
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }

      //item gradient
      ModelItemMatFac_computeIGrad(model, u, itemWtSets->item, itemWtSets->wt, iGrad);

      //update item
      for (j = 0; j < model->_(facDim); j++) {
        iGradsAcc[itemWtSets->item][j] = iGradsAcc[itemWtSets->item][j]*params->rhoRMS + 
          (1.0-params->rhoRMS)*iGrad[j]*iGrad[j]; 
        model->_(iFac)[itemWtSets->item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[itemWtSets->item][j] + 0.0000001))*iGrad[j];
      }

    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      if (model->_(isTerminateModel)(model, bestModel, iter, &bestIter, &bestObj, 
          &prevObj, valTest, data)) {
        break;
      }
    }

  } 

  valTest->setObj = bestObj;
  printf("\nNum Iter: %d best Obj: %.10e bestIter: %d", iter, bestObj, bestIter);

  valTest->valItemsRMSE = bestModel->_(indivItemCSRErr) (bestModel, data->valMat, NULL);
  printf("\nValidation items error: %f", valTest->valItemsRMSE);

  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->testMat, 20); 
  printf("\nTest spearman: %f", valTest->testSpearman);

  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->valMat, 20);
  printf("\nVal spearman: %f", valTest->valSpearman);

  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }
 
  valTest->trainSetRMSE = bestModel->_(trainErr) (bestModel, data, NULL);
  printf("\nTrain set error(matfac): %f", valTest->trainSetRMSE);

  valTest->testSetRMSE = bestModel->_(testErr)(bestModel, data, NULL); 
  printf("\nTest set error(matfac): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = bestModel->_(indivItemCSRErr) (bestModel, data->testMat, NULL);
  printf("\nTest items error(matfac): %f", valTest->testItemsRMSE);
 
  //get train error
  valTest->trainItemsRMSE = bestModel->_(indivTrainSetsErr) (bestModel, data);
  printf("\nTrain set indiv error(matfac): %f", valTest->trainItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));


  bestModel->_(copy)(bestModel, model);

  bestModel->_(free)(bestModel);

  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);

  free(iGrad);
  free(uGrad);
}


void ModelItemMatFac_trainSamp(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest) {
  
  int u, i, j, iter, subIter, item, nUserItems, itemInd;
  int nnz = 0;
  ModelItemMatFac *model = self;
  gk_csr_t *trainMat = data->trainMat;
  float *iGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float prevVal = 0, prevObj = 0, itemRat;

  //find nnz
  for (u = 0; u < params->nUsers; u++) {
    nnz += trainMat->rowptr[u+1] - trainMat->rowptr[u]; 
  }
  printf("\nTrain nnz = %d", nnz);

  printf("\nInit Obj: %f", model->_(objective) (model, data, sim));

  for (iter = 0; iter < params->maxIter; iter++) {
    //update user and item latent factor pair
    for (subIter = 0; subIter < nnz; subIter++) {
     
      //sample u
      u = rand() % params->nUsers;
      
      //sample item rated by user
      nUserItems = trainMat->rowptr[u+1] - trainMat->rowptr[u];
      itemInd = rand()%nUserItems; 
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      itemRat = trainMat->rowval[trainMat->rowptr[u] + itemInd]; 
      
      //user gradient
      ModelItemMatFac_computeUGrad(model, u, item, itemRat, uGrad);

      //update user
      for (i = 0; i < model->_(facDim); i++) {
        model->_(uFac)[u][i] -= (model->_(learnRate)/(1.0 + model->_(learnRate)*model->_(regU)*iter))*uGrad[i];
      }

      //item gradient
      ModelItemMatFac_computeIGrad(model, u, item, itemRat, iGrad);

      //update item
      for (i = 0; i < model->_(facDim); i++) {
        model->_(iFac)[item][i] -= (model->_(learnRate)/(1.0 + model->_(learnRate)*model->_(regI)*iter))*iGrad[i];
      }

    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->valItemsRMSE = model->_(indivItemCSRErr) (model, data->valMat, NULL);
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter:%d Obj: %f valRMSE:%f", iter, valTest->setObj, 
          valTest->valItemsRMSE);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
  }

  printf("\nFinal Obj: %f", model->_(objective) (model, data, sim));
  
  valTest->valItemsRMSE = model->_(indivItemCSRErr) (model, data->valMat, NULL);
  //printf("\nIter:%d ErrToMat ratio:%f", iter, valTest[0]);
 
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }
  
  //valTest->trainSetRMSE = model->_(trainErr) (model, data, NULL);
  //printf("\nTrain set error(matfac): %f", valTest->trainSetRMSE);

  //valTest->testSetRMSE = model->_(testErr)(model, data, NULL); 
  //printf("\nTest set error(matfac): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(matfac): %f", valTest->testItemsRMSE);
 
  //get train error
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(matfac): %f", valTest->trainItemsRMSE);

  //valTest->trainItemsRMSE = model->_(indivItemCSRErr)(model, data->trainMat, NULL);
  //printf("\nTrain items indiv error(matfac): %f", valTest->trainItemsRMSE);

  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //writeMat(model->_(uFac), data->nUsers, data->facDim, "copyUFac.txt");
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 
  //writeMat(model->_(iFac), data->nItems, data->facDim, "copyIFac.txt");
  //printf("\nTrain items indiv error(matfac orig): %f", 
  //    model->_(indivItemCSRErr)(model, data->trainMat, "origMatfacRes.txt"));

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));
  
  free(iGrad);
  free(uGrad);
}


Model ModelItemMatFacProto = {
  .objective               = ModelItemMatFac_objective,
  .setScore                = ModelItemMatFac_majSetScore,
  .train                   = ModelItemMatFac_trainRMSProp,
  .validationErr           = ModelItemMatFac_validationErr
};


void modelItemMatFac(Data *data, Params *params, ValTestRMSE *valTest) {
  
  //allocate storage for model
  ModelItemMatFac *model = NEW(ModelItemMatFac, 
      "mat fac model for user-item ratings");
  model->_(init) (model, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate); 

  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 
  
  //valTest->testSetRMSE = model->_(testErr)(model, data, NULL); 
  //printf("\nTest set error(matfac): %f", valTest->testSetRMSE);
  
  //initialize user-item weights
  //initUserItemWeights(data);

  //load user item weights from train
  loadUserItemWtsFrmTrain(data);
  
  printf("\nmodel mat fac Init obj: %.10e", model->_(objective)(model, data, NULL));
  //get train error
  //valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  //printf("\nTrain set indiv error(matfac): %f", valTest->trainItemsRMSE);

  //train model
  model->_(train)(model, data, params, NULL, valTest);  
 
  //compare model with loaded latent factors if any
  //model->_(cmpLoadedFac)(model, data);

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "modelMatfacUFac.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "modelMatfacIFac.txt");

  model->_(free)(model);
}



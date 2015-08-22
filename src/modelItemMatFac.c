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
  
  int u, i, j, iter, item;
  UserSets *userSet = NULL;
  ModelItemMatFac *model = self;
  ItemWtSets *itemWtSets = NULL;
  float *iGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float *) malloc(sizeof(float)*model->_(facDim));
  float prevVal = 0, prevObj = 0;

  model->_(objective) (model, data, sim);

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
    /*
    if (iter % VAL_ITER == 0) {
      valTest->valItemsRMSE = model->_(indivItemSetErr) (model, data->valSet);
      //printf("\nIter:%d validation err:%f", iter, valTest->valItemsRMSE);
      if (fabs(prevVal - valTest->valItemsRMSE) < EPS) {
        printf("\nConverged in iterations: %d currVal:%f prevVal:%f", iter, 
            valTest->valItemsRMSE, prevVal);
        break;
      }
      prevVal = valTest->valItemsRMSE;
    }
    */

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    
  }

  valTest->valItemsRMSE = model->_(indivItemSetErr) (model, data->valSet);
 
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }
 
  valTest->trainSetRMSE = model->_(trainErr) (model, data, NULL);
  printf("\nTrain set error(matfac): %f", valTest->trainSetRMSE);

  valTest->testSetRMSE = model->_(testErr)(model, data, NULL); 
  printf("\nTest set error(matfac): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
  printf("\nTest items error(matfac): %f", valTest->testItemsRMSE);
 
  //get train error
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(matfac): %f", valTest->trainItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));
  
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
        model->_(uFac)[u][i] -= model->_(learnRate)*uGrad[i];
      }

      //item gradient
      ModelItemMatFac_computeIGrad(model, u, item, itemRat, iGrad);

      //update item
      for (i = 0; i < model->_(facDim); i++) {
        model->_(iFac)[item][i] -= model->_(learnRate)*iGrad[i];
      }

    }

    //validation
    /*
    if (iter % VAL_ITER == 0) {
      valTest->valItemsRMSE = model->_(indivItemSetErr) (model, data->valSet);
      //printf("\nIter:%d validation err:%f", iter, valTest[0]);
      if (fabs(prevVal - valTest->valItemsRMSE) < EPS) {
        printf("\nConverged in iterations: %d currVal:%f prevVal:%f", iter, valTest->valItemsRMSE, prevVal);
        break;
      }
      prevVal = valTest->valItemsRMSE;
    }
    */

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
  }

  printf("\nFinal Obj: %f", model->_(objective) (model, data, sim));
  
  valTest->valItemsRMSE = model->_(indivItemSetErr) (model, data->valSet);
  //printf("\nIter:%d ErrToMat ratio:%f", iter, valTest[0]);
 
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }
  
  //valTest->trainSetRMSE = model->_(trainErr) (model, data, NULL);
  //printf("\nTrain set error(matfac): %f", valTest->trainSetRMSE);

  //valTest->testSetRMSE = model->_(testErr)(model, data, NULL); 
  //printf("\nTest set error(matfac): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
  .train                   = ModelItemMatFac_train,
  .validationErr           = ModelItemMatFac_validationErr
};


void modelItemMatFac(Data *data, Params *params, ValTestRMSE *valTest) {
  
  //allocate storage for model
  ModelItemMatFac *model = NEW(ModelItemMatFac, 
      "mat fac model for user-item ratings");
  model->_(init) (model, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate); 

  //initialize user-item weights
  //initUserItemWeights(data);

  //load user item weights from train
  loadUserItemWtsFrmTrain(data);

  //train model
  model->_(train)(model, data, params, NULL, valTest);  
 
  //compare model with loaded latent factors if any
  //model->_(cmpLoadedFac)(model, data);

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "modelMatfacUFac.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "modelMatfacIFac.txt");

  model->_(free)(model);
}



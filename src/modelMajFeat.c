#include "modelMajFeat.h"


float ModelMajFeat_itemScoreFeatAvg(void *self, int u, int item, 
    gk_csr_t *featMat) {
  
  ModelMajFeat *model = self;
  int i, j, ii, jj;
  int nFeatures = featMat->rowptr[item+1] - featMat->rowptr[item];
  float rat = 0;

  //go through item features to compute ratings
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
      rat += dotProd(model->_(iFac)[featMat->rowind[ii]], model->_(uFac)[u], 
          model->_(facDim));
  }
  rat = rat/nFeatures;
  
  return rat;
}


float ModelMajFeat_objectiveAvg(void *self, Data *data, float **sim) {
  
  ModelMajFeat *model = self;
  int u, i, j, ii, jj, nRelUsers, item;
  float rmse = 0, diff = 0, actRat = 0, estRat = 0, temp = 0;
  float uRegErr = 0, uNorm = 0, fRegErr = 0, iNorm = 0, obj;
  gk_csr_t *trainMat = data->trainMat;
  int nFeatures = data->itemFeatMat->ncols;

  for (u = 0; u < data->nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      actRat = trainMat->rowval[ii];
      estRat = ModelMajFeat_itemScoreFeatAvg(model, u, item, data->itemFeatMat);
      diff = actRat - estRat;
      rmse += diff*diff;
    }
    temp = dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    uRegErr += temp;   
    uNorm += temp;
  }
  uRegErr = uRegErr*model->_(regU);
  
  for (i = 0; i < nFeatures; i++) {
    temp = dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim));
    fRegErr += temp;
    iNorm += temp;
  }
  fRegErr = fRegErr*model->_(regI);

  obj = rmse + uRegErr + fRegErr;

  return obj;
}


void ModelMajFeat_trainRMSPropAvg(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, bestIter, u, i, j, nnz, item, itemInd, featInd;
  int ii, jj;
  int nUsers, nItems, nFeatures, nUItems, nIFeat;
  float r_ui, r_ui_est, temp, bestObj, prevObj;
  gk_csr_t *trainMat = NULL;
  gk_csr_t *itemFeatMat = NULL;

  trainMat    = data->trainMat;
  itemFeatMat = data->itemFeatMat;
  nUsers      = data->trainMat->nrows;
  nItems      = data->trainMat->ncols;
  nFeatures      = data->itemFeatMat->ncols;
  bestIter = 0;
  bestObj = 0;

  assert(nItems == data->itemFeatMat->nrows);

  ModelMajFeat *model = self;
  ModelMajFeat *bestModel = NEW(ModelMajFeat, "model that achieved lowest obj");
  bestModel->_(init) (bestModel, nUsers, nFeatures, params->facDim, params->regU, 
      params->regI, params->learnRate);

  float *sumFeatLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float*) malloc(sizeof(float)*model->_(facDim));

  float **fGradsAcc = (float**) malloc(sizeof(float*)*nFeatures);
  for (i = 0; i < nFeatures; i++) {
    fGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(fGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  
  float **uGradsAcc = (float**) malloc(sizeof(float*)*nUsers);
  for (i = 0; i < nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  nnz = 0;
  for (u = 0; u < nUsers; u++) {
    nnz += trainMat->rowptr[u+1] - trainMat->rowptr[u]; 
  }
  printf("\nNNZ: %d", nnz);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < nnz; subIter++) {
      
      //sample u
      u = rand() % nUsers;
      nUItems = trainMat->rowptr[u+1] - trainMat->rowptr[u];
      if (nUItems == 0) {
        continue;
      }

      //sample item
      itemInd = rand() % nUItems;
      item = trainMat->rowind[trainMat->rowptr[u] + itemInd];
      r_ui = trainMat->rowval[trainMat->rowptr[u] + itemInd];
      memset(sumFeatLatFac, 0, sizeof(float)*model->_(facDim));
      nIFeat = 0;
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
          ii++) {
        nIFeat++;
        featInd = itemFeatMat->rowind[ii];
        for (j = 0; j < model->_(facDim); j++) {
          sumFeatLatFac[j] += model->_(iFac)[featInd][j];
        }
      } 
      r_ui_est = dotProd(sumFeatLatFac, model->_(uFac)[u], model->_(facDim));
      r_ui_est = r_ui_est/nIFeat;
      
      //compute u gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_ui_est - r_ui)*sumFeatLatFac[j]*(1.0/nIFeat);
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        //accumulate gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j]+ 0.0000001))*uGrad[j];
      }
     
      //compute features gradient n update
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
           ii++) {
        featInd = itemFeatMat->rowind[ii];
        //get feature gradient
        for (j = 0; j < model->_(facDim); j++) {
          fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(iFac)[featInd][j]*(1.0/nIFeat);
          fGrad[j] += 2.0*model->_(regI)*model->_(iFac)[featInd][j];
          //accumulate gradient square
          fGradsAcc[featInd][j] = params->rhoRMS*fGradsAcc[featInd][j] +
            (1.0 - params->rhoRMS)*fGrad[j]*fGrad[j];
          //update
          model->_(iFac)[featInd][j] -= (model->_(learnRate)/sqrt(fGradsAcc[featInd][j] + 0.0000001))*fGrad[j];
        }
      }
      
    }
    
    //objective check 
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective) (model, data, NULL);
      valTest->valSpearman = model->_(coldHitRateTr)(model, trainMat,
          data->valMat, data->itemFeatMat, data->valItemIds, data->nValItems, 
          10);
      printf("\nIter: %d Obj: %.10e valHR: %f", iter, valTest->setObj, 
          valTest->valSpearman);
      if (iter > 0) {
        if (valTest->setObj < bestObj) {
          model->_(copy)(model, bestModel);
          bestObj = valTest->setObj;
          bestIter = iter;
        }

        if (iter - bestIter > 500) {
          //not able to beat best after 500 iterations
          printf("\nbestIter: %d bestObj: %.10e currIter: %d currObj: %.10e",
              bestIter, bestObj, iter, valTest->setObj);
        }
        
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e",
              iter, prevObj, valTest->setObj);
          break;
        }
      }

      if (iter == 0) {
        bestObj = valTest->setObj;
        bestIter = iter;
      }
      
      prevObj = valTest->setObj;
    }

  }

  printf("\nEnd Obj: %.10e bestObj: %.10e bestIter: %d", valTest->setObj, 
      bestObj, bestIter);
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSpearman = model->_(coldHitRateTr)(bestModel, trainMat,
      data->valMat, data->itemFeatMat, data->valItemIds, data->nValItems, 
      10);
  printf("\nbest valHR: %f", valTest->valSpearman);

  valTest->testSpearman = model->_(coldHitRateTr)(bestModel, trainMat,
      data->testMat, data->itemFeatMat, data->testItemIds, data->nTestItems, 
      10);
  printf("\nbest testHR: %f", valTest->testSpearman);

  bestModel->_(copy) (bestModel, model);
  bestModel->_(free) (bestModel);

  for (i = 0; i < nFeatures; i++) {
    free(fGradsAcc[i]);
  }
  free(fGradsAcc);

  for (i = 0; i < nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  free(sumFeatLatFac);
  free(fGrad);
  free(uGrad);
}


Model ModelMajFeatProto = {
  .objective            = ModelMajFeat_objectiveAvg,
  .itemFeatScore        = ModelMajFeat_itemScoreFeatAvg,
  .train                = ModelMajFeat_trainRMSPropAvg
};



void modelMajFeat(Data *data, Params *params, ValTestRMSE *valTest) {

  int nUsers, nItems, nFeatures;
  
  nUsers = data->trainMat->nrows;
  nItems = data->trainMat->ncols;
  nFeatures = data->itemFeatMat->ncols;

  printf("\nnUsers: %d", nUsers);
  printf("\nnItems: %d", nItems);
  printf("\nnFeatures: %d", nFeatures);

  ModelMajFeat *model = NEW(ModelMajFeat, "prediction using features");
  model->_(init)(model, nUsers, nFeatures, params->facDim, params->regU,
      params->regI, params->learnRate); 
  model->rhoRMS = params->rhoRMS;
  model->epsRMS = params->epsRMS;

  model->_(train) (model, data, params, NULL, valTest);

  //save model latent factors
  writeMat(model->_(uFac), nUsers, model->_(facDim), "majFeatUFac.txt");
  writeMat(model->_(iFac), nFeatures, model->_(facDim), "majFeatFFac.txt");

  model->_(free)(model);
}



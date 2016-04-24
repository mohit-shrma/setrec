#include "modelMajFeat.h"


float ModelMajFeat_itemScoreFeatAvg(void *self, int u, int item, 
    gk_csr_t *featMat) {
  
  ModelMajFeat *model = self;
  int i, j, ii, jj, fInd;
  int nFeatures = featMat->rowptr[item+1] - featMat->rowptr[item];
  float rat = 0;

  //go through item features to compute ratings
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
    fInd = featMat->rowind[ii]; 
    rat += dotProd(model->_(iFac)[fInd], model->_(uFac)[u], 
        model->_(facDim));
  }
  rat = rat/nFeatures;
  
  return rat;
}


float ModelMajFeat_itemScoreFeatSum(void *self, int u, int item, 
    gk_csr_t *featMat) {
  
  ModelMajFeat *model = self;
  int i, j, ii, jj, fInd;
  int nFeatures = featMat->rowptr[item+1] - featMat->rowptr[item];
  float rat = 0, fVal;

  //go through item features to compute ratings
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
    fInd = featMat->rowind[ii]; 
    fVal = featMat->rowval[ii];
    rat += fVal*dotProd(model->_(iFac)[fInd], model->_(uFac)[u], 
        model->_(facDim));
  }
  
  return rat;
}


float ModelMajFeat_objectiveSum(void *self, Data *data, float **sim) {
  
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
      estRat = ModelMajFeat_itemScoreFeatSum(model, u, item, data->itemFeatMat);
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


float ModelMajFeat_itemScoreFeatAvgCons(void *self, int u, int item, 
    gk_csr_t *featMat, int *nConstViol, float *featFacSum) {
  
  ModelMajFeat *model = self;
  int i, j, ii, jj, featInd;
  int nFeatures = featMat->rowptr[item+1] - featMat->rowptr[item];
  float rat = 0, featRat = 0;
  *nConstViol = 0;
  memset(featFacSum, 0, sizeof(float)*model->_(facDim));

  //go through item features to compute ratings
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
    featInd = featMat->rowind[ii];
    featRat = dotProd(model->_(iFac)[featInd], model->_(uFac)[u], 
        model->_(facDim));
    if (featRat > MAX_RAT) {
      *nConstViol = *nConstViol + 1;
      for (j = 0; j < model->_(facDim); j++) {
        featFacSum[j] += model->_(iFac)[featInd][j];
      }
    }
    rat += featRat;
  }
  rat = rat/nFeatures;
  
  return rat;
}


float ModelMajFeat_objectiveAvgCons(void *self, Data *data, float **sim) {
  
  ModelMajFeat *model = self;
  int u, i, j, ii, jj, nRelUsers, item;
  float rmse = 0, diff = 0, actRat = 0, estRat = 0, temp = 0;
  float uRegErr = 0, uNorm = 0, fRegErr = 0, fNorm = 0, obj;
  gk_csr_t *trainMat = data->trainMat;
  int nFeatures = data->itemFeatMat->ncols;
  float *featFacSum = (float*) malloc(sizeof(float)*model->_(facDim));
  int nConstViol = 0;
  int constViol = 0;

  for (u = 0; u < data->nUsers; u++) {
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      actRat = trainMat->rowval[ii];
      nConstViol = 0;
      estRat = ModelMajFeat_itemScoreFeatAvgCons(model, u, item, 
          data->itemFeatMat, &nConstViol, featFacSum);
      diff = actRat - estRat;
      rmse += diff*diff;
      if (nConstViol) {
        constViol++;
        rmse += model->constrainWt*(dotProd(model->_(uFac)[u], featFacSum, 
                                model->_(facDim)) - nConstViol*MAX_RAT);
      }
    }
    temp = dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    uRegErr += temp;   
    uNorm += temp;
  }
  uRegErr = uRegErr*model->_(regU);
  
  for (i = 0; i < nFeatures; i++) {
    temp = dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim));
    fRegErr += temp;
    fNorm += temp;
  }
  fRegErr = fRegErr*model->_(regI);

  obj = rmse + uRegErr + fRegErr;
  free(featFacSum);

  printf("\nconstViol: %d uNorm: %f fNorm: %f rmse: %f uRegErr: %f fRegErr: %f",
      constViol, uNorm, fNorm, rmse, uRegErr, fRegErr);
  
  return obj;
}


void ModelMajFeat_gradientCheck(void *self, int u, int item, float r_ui, 
    gk_csr_t *featMat) {
  
  ModelMajFeat *model = self;
  int i, j, ii, jj, featInd, nIFeats;
  float lossRight, lossLeft, gradE, r_ui_est, r_ui_est_pert, featRat;
  float *sumFeatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *sumConstFeatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fac   = (float*) malloc(sizeof(float)*model->_(facDim));
  int nFeatures = featMat->ncols;
  bool isConstViol = false;
  bool *isFeatConstViol = (bool*) malloc(sizeof(bool)*nFeatures);

  memset(isFeatConstViol, false, sizeof(bool)*nFeatures);
  memset(sumFeatFac, 0, sizeof(float)*model->_(facDim));
  memset(sumConstFeatFac, 0, sizeof(float)*model->_(facDim));
  
  nIFeats = featMat->rowptr[item+1] - featMat->rowptr[item];
  r_ui_est = 0;
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
    featInd = featMat->rowind[ii];
    featRat = dotProd(model->_(uFac)[u], model->_(iFac)[featInd], 
        model->_(facDim));
    r_ui_est += featRat;
    if (featRat > MAX_RAT) {
      //const violated
      isConstViol = true;
      isFeatConstViol[featInd] = true;
    }
    for (j = 0; j < model->_(facDim); j++) {
      sumFeatFac[j] += model->_(iFac)[featInd][j];
      if (featRat > MAX_RAT) {
        //const violated
        sumConstFeatFac[j] += model->_(iFac)[featInd][j];
      }
    }
  }
  r_ui_est = r_ui_est/nIFeats;

  //sample a feature
  featInd = featMat->rowind[featMat->rowptr[item] + rand()%nIFeats];
  featRat = dotProd(model->_(uFac)[u], model->_(iFac)[featInd], model->_(facDim));
  //feature gradient
  memset(fGrad, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(uFac)[u][j]/nIFeats;
    if (featRat > MAX_RAT) {
      //const viol
      fGrad[j] += model->constrainWt*model->_(uFac)[u][j];
    }
  }
  
  //perturb feature with +E and compute loss
  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumFeatFac[j] + 0.0001;
  }
  r_ui_est_pert = dotProd(model->_(uFac)[u], fac, model->_(facDim))/nIFeats;
  lossRight = pow(r_ui_est_pert - r_ui, 2);
  //compute feat rat on perturb feat fac
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(iFac)[featInd][j] + 0.0001; 
  }
  featRat = dotProd(model->_(uFac)[u], fac, model->_(facDim));
  if (featRat > MAX_RAT) {
    for (j = 0; j < model->_(facDim); j++) {
      if (isFeatConstViol[featInd]) {
        //constraint already violated, only add perturbation
        fac[j] = sumConstFeatFac[j] + 0.0001;
      } else {
        fac[j] = sumConstFeatFac[j] + fac[j];
      }
    }
    lossRight += model->constrainWt*dotProd(model->_(uFac)[u], 
        fac, model->_(facDim));
  }
 
  //perturb feature with -E and compute loss
  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumFeatFac[j] - 0.0001;
  }
  r_ui_est_pert = dotProd(model->_(uFac)[u], fac, model->_(facDim))/nIFeats;
  lossLeft = pow(r_ui_est_pert - r_ui, 2);
  //compute feat rat on perturb feat fac
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(iFac)[featInd][j] - 0.0001; 
  }
  featRat = dotProd(model->_(uFac)[u], fac, model->_(facDim));
  if (featRat > MAX_RAT) {
    for (j = 0; j < model->_(facDim); j++) {
      if (isFeatConstViol[featInd]) {
        //constraint already violatedi, only add perturbation
        fac[j] = sumConstFeatFac[j] - 0.0001;
      } else {
        fac[j] = sumConstFeatFac[j] + fac[j];
      }
    }
    lossLeft += model->constrainWt*dotProd(model->_(uFac)[u], 
        fac, model->_(facDim));
  }
  
  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*fGrad[j]*0.0001;
  }

  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\nu: %d item: %d feature: %d diff: %f div: %f lDiff:%f gradE:%f", 
        u, item, featInd, lossRight-lossLeft-gradE, (lossRight-lossLeft)/gradE, 
        lossRight-lossLeft, gradE);
  }

  
  //TODO: grad check w.r.t. user 


  free(isFeatConstViol);
  free(sumFeatFac);
  free(sumConstFeatFac);
  free(uGrad);
  free(fGrad);
  free(fac);
}


void ModelMajFeat_trainRMSPropAvgCons(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, bestIter, u, i, j, nnz, item, itemInd, featInd;
  int ii, jj;
  int nUsers, nItems, nFeatures, nUItems, nIFeat;
  int subConstViol;
  float r_ui, r_ui_est, temp, bestObj, prevObj, featRat;
  gk_csr_t *trainMat = NULL;
  gk_csr_t *itemFeatMat = NULL;

  trainMat    = data->trainMat;
  itemFeatMat = data->itemFeatMat;
  nUsers      = data->trainMat->nrows;
  nItems      = data->trainMat->ncols;
  nFeatures   = data->itemFeatMat->ncols;
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
  float *uGradConstViol = (float*) malloc(sizeof(float)*model->_(facDim));
  bool *isFeatConstViol = (bool*) malloc(sizeof(bool)*nFeatures);
  bool constViol = false;

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
    subConstViol = 0;
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

      //ModelMajFeat_gradientCheck(model, u, item, r_ui, itemFeatMat);
      //ModelMajFeat_gradientCheck(model, 316, 6300, r_ui, itemFeatMat);
      //continue;

      memset(sumFeatLatFac, 0, sizeof(float)*model->_(facDim));
      memset(uGradConstViol, 0, sizeof(float)*model->_(facDim));
      memset(isFeatConstViol, false, sizeof(bool)*nFeatures);
      constViol = false;
      nIFeat = 0;
      featRat = 0;
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
          ii++) {
        nIFeat++;
        featInd = itemFeatMat->rowind[ii];
        featRat = dotProd(model->_(uFac)[u], model->_(iFac)[featInd], 
            model->_(facDim));
        if (featRat > MAX_RAT) {
          isFeatConstViol[featInd] = true;
          constViol = true;
        }
        for (j = 0; j < model->_(facDim); j++) {
          sumFeatLatFac[j] += model->_(iFac)[featInd][j];
          if (isFeatConstViol[featInd]) {
            uGradConstViol[j] += model->_(iFac)[featInd][j];
          }
        }
      } 
      r_ui_est = dotProd(sumFeatLatFac, model->_(uFac)[u], model->_(facDim));
      r_ui_est = r_ui_est/nIFeat;

      if (constViol) {
        subConstViol++;
      }
      
      //compute u gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_ui_est - r_ui)*sumFeatLatFac[j]*(1.0/nIFeat);
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        uGrad[j] += params->constrainWt*uGradConstViol[j];

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
          fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(uFac)[u][j]*(1.0/nIFeat);
          fGrad[j] += 2.0*model->_(regI)*model->_(iFac)[featInd][j];
          
          if (constViol && isFeatConstViol[featInd]) {
            //constraint violated
            fGrad[j] += params->constrainWt*model->_(uFac)[u][j];
          }

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
      printf("\nIter: %d Obj: %.10e valHR: %f constViol: %d", iter, valTest->setObj, 
          valTest->valSpearman, subConstViol);
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

  printf("\niter: %d End Obj: %.10e bestObj: %.10e bestIter: %d", iter, 
      valTest->setObj, bestObj, bestIter);
  
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
  free(isFeatConstViol);
  free(uGradsAcc);
  free(sumFeatLatFac);
  free(fGrad);
  free(uGrad);
}


void ModelMajFeat_trainRMSPropAvg(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, bestIter, u, i, j, nnz, item, itemInd, featInd;
  int ii, jj;
  int nUsers, nItems, nFeatures, nUItems, nIFeat;
  float r_ui, r_ui_est, temp, bestObj, prevObj, featRat;
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
      featRat = 0;
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
          ii++) {
        nIFeat++;
        featInd = itemFeatMat->rowind[ii];
        featRat = dotProd(model->_(uFac)[u], model->_(iFac)[featInd], 
            model->_(facDim));
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
          fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(uFac)[u][j]*(1.0/nIFeat);
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


void ModelMajFeat_trainRMSPropSum(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, bestIter, u, i, j, nnz, item, itemInd, featInd;
  int ii, jj;
  int nUsers, nItems, nFeatures, nUItems, nIFeat;
  float r_ui, r_ui_est, temp, bestObj, prevObj, featRat, featVal;
  gk_csr_t *trainMat = NULL;
  gk_csr_t *itemFeatMat = NULL;

  trainMat    = data->trainMat;
  itemFeatMat = data->itemFeatMat;
  nUsers      = data->trainMat->nrows;
  nItems      = data->trainMat->ncols;
  nFeatures      = data->itemFeatMat->ncols;
  bestIter = 0;
  bestObj = 0;

  printf("\nnUsers: %d nItems: %d nFeatures: %d", nUsers, nItems, nFeatures);
  printf("\nNItems in featMat: %d", data->itemFeatMat->nrows);
  //assert(nItems == data->itemFeatMat->nrows);

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
      featRat = 0;
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
          ii++) {
        nIFeat++;
        featInd = itemFeatMat->rowind[ii];
        featVal = itemFeatMat->rowval[ii];
        featRat = featVal*dotProd(model->_(uFac)[u], model->_(iFac)[featInd], 
            model->_(facDim));
        for (j = 0; j < model->_(facDim); j++) {
          sumFeatLatFac[j] += featVal*model->_(iFac)[featInd][j];
        }
      } 
      r_ui_est = dotProd(sumFeatLatFac, model->_(uFac)[u], model->_(facDim));

      //compute u gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_ui_est - r_ui)*sumFeatLatFac[j];
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
        featVal = itemFeatMat->rowval[ii];
        //get feature gradient
        for (j = 0; j < model->_(facDim); j++) {
          fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(uFac)[u][j]*featVal;
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
      fflush(stdout);
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


void ModelMajFeat_trainSGDSum(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, bestIter, u, i, j, nnz, item, itemInd, featInd;
  int ii, jj;
  int nUsers, nItems, nFeatures, nUItems, nIFeat;
  float r_ui, r_ui_est, temp, bestObj, prevObj, featRat, featVal;
  gk_csr_t *trainMat = NULL;
  gk_csr_t *itemFeatMat = NULL;

  trainMat    = data->trainMat;
  itemFeatMat = data->itemFeatMat;
  nUsers      = data->trainMat->nrows;
  nItems      = data->trainMat->ncols;
  nFeatures      = data->itemFeatMat->ncols;
  bestIter = 0;
  bestObj = 0;

  printf("\nnUsers: %d nItems: %d nFeatures: %d", nUsers, nItems, nFeatures);
  printf("\nNItems in featMat: %d", data->itemFeatMat->nrows);
  //assert(nItems == data->itemFeatMat->nrows);

  ModelMajFeat *model = self;
  ModelMajFeat *bestModel = NEW(ModelMajFeat, "model that achieved lowest obj");
  bestModel->_(init) (bestModel, nUsers, nFeatures, params->facDim, params->regU, 
      params->regI, params->learnRate);

  float *sumFeatLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad = (float*) malloc(sizeof(float)*model->_(facDim));

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
      featRat = 0;
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
          ii++) {
        nIFeat++;
        featInd = itemFeatMat->rowind[ii];
        featVal = itemFeatMat->rowval[ii];
        featRat = featVal*dotProd(model->_(uFac)[u], model->_(iFac)[featInd], 
            model->_(facDim));
        for (j = 0; j < model->_(facDim); j++) {
          sumFeatLatFac[j] += featVal*model->_(iFac)[featInd][j];
        }
      } 
      r_ui_est = dotProd(sumFeatLatFac, model->_(uFac)[u], model->_(facDim));

      //compute u gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_ui_est - r_ui)*sumFeatLatFac[j];
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];

        //update
        model->_(uFac)[u][j] -= model->_(learnRate)*uGrad[j];
      }
     
      //compute features gradient n update
      for (ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1]; 
           ii++) {
        featInd = itemFeatMat->rowind[ii];
        featVal = itemFeatMat->rowval[ii];
        //get feature gradient
        for (j = 0; j < model->_(facDim); j++) {
          fGrad[j] = 2.0*(r_ui_est - r_ui)*model->_(uFac)[u][j]*featVal;
          fGrad[j] += 2.0*model->_(regI)*model->_(iFac)[featInd][j];
          
          //update
          model->_(iFac)[featInd][j] -= model->_(learnRate)*fGrad[j];
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
      fflush(stdout);
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

  free(sumFeatLatFac);
  free(fGrad);
  free(uGrad);
}


Model ModelMajFeatProto = {
  .objective            = ModelMajFeat_objectiveSum,
  .itemFeatScore        = ModelMajFeat_itemScoreFeatSum,
  //.train                = ModelMajFeat_trainRMSPropSum,
  .train                = ModelMajFeat_trainSGDSum
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
  model->constrainWt = params->constrainWt;

  model->_(train) (model, data, params, NULL, valTest);

  //save model latent factors
  //writeMat(model->_(uFac), nUsers, model->_(facDim), "majFeatUFac.txt");
  //writeMat(model->_(iFac), nFeatures, model->_(facDim), "majFeatFFac.txt");

  model->_(free)(model);
}



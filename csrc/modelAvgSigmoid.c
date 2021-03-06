#include "modelAvgSigmoid.h"


void ModelAvgSigmoid_gradCheck(void *self, int u, int *set, int setSz, float r_us) {
  int i, j, item;
  float commGradCoeff, r_us_est, avgRat;
  ModelAvgSigmoid *model = self;
  float *sumItemLatFac   = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad           = (float*) malloc(sizeof(float)*model->_(facDim));
  float *iGrad           = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fac             = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float lossRight, lossLeft, gradE, umGrad, g_kGrad;

  //compute common grad coeffs and est score
  memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (j = 0; j < model->_(facDim); j++) {
      sumItemLatFac[j] += model->_(iFac)[item][j];
    }
  }

  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz; 
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k);
  commGradCoeff = exp(-1.0*model->g_k*(avgRat - model->u_m[u]))*r_us_est*r_us_est;
  commGradCoeff = 2 * (r_us_est - r_us) * commGradCoeff;

  //find gradient w.r.t. u
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = model->g_k*commGradCoeff*sumItemLatFac[j]/setSz;
    //add regularization
    //uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
  }
 
  //perturb user with +E and -E and compute Loss 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(uFac)[u][j] + 0.0001;
  }
  avgRat = dotProd(fac, sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k);
  lossRight = pow(r_us_est - r_us, 2);
  
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(uFac)[u][j] - 0.0001;
  }
  avgRat = dotProd(fac, sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k);
  lossLeft = pow(r_us_est - r_us, 2);

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*uGrad[j]*0.0001;
  }
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\nu: %d diff: %f div: %f lDiff:%f gradE:%f", u, lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  //find gradient w.r.t. one of the item in set
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  item = set[rand()%setSz];
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = model->g_k*commGradCoeff*model->_(uFac)[u][j]/setSz;
  }

  //perturb item with +E and -E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] + 0.0001;
  }
  avgRat = dotProd(model->_(uFac)[u], fac, model->_(facDim))/setSz; 
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k);
  lossRight = pow(r_us_est - r_us, 2);

  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] - 0.0001;
  }
  avgRat = dotProd(model->_(uFac)[u], fac, model->_(facDim))/setSz; 
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k);
  lossLeft = pow(r_us_est - r_us, 2);

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*iGrad[j]*0.0001;
  }
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {

    printf("\ni: %d diff: %f div: %f lDiff:%f gradE:%f", item, lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  //find gradient w.r.t. u_m
  umGrad = -1.0*commGradCoeff*model->g_k;

  //perturb u_m with +E and -E
  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = sigmoid(avgRat - (model->u_m[u] + 0.0001), model->g_k);
  lossRight = pow(r_us_est - r_us, 2);

  r_us_est = sigmoid(avgRat - (model->u_m[u] - 0.0001), model->g_k);
  lossLeft = pow(r_us_est - r_us, 2);
  
  gradE = 2.0*umGrad*0.0001;
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {

    printf("\num: %d diff: %f div: %f lDiff:%f gradE:%f", u, lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  //find gradient w.r.t. g_k
  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz;  
  g_kGrad = commGradCoeff*(avgRat - model->u_m[u]);
  
  //perturb g_k with +E and -E
  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k + 0.0001);
  lossRight = pow(r_us_est - r_us, 2);

  r_us_est = sigmoid(avgRat - model->u_m[u], model->g_k - 0.0001);
  lossLeft = pow(r_us_est - r_us, 2);

  gradE = 2.0*g_kGrad*0.0001;

  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\ng_k: %d diff: %f div: %f lDiff:%f gradE:%f", u, lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }


  free(sumItemLatFac);
  free(iGrad);
  free(uGrad);
  free(fac);
}


float ModelAvgSigmoid_setScore(void *self, int u, int *set, int setSz, 
    float **sim) {
  
  int i, item;
  ModelAvgSigmoid *model = self;
  float r_us_est = 0, diff;

  for (i = 0; i < setSz; i++) {
    item     = set[i];
    r_us_est += dotProd(model->_(uFac)[u], model->_(iFac)[item],
        model->_(facDim));
  }
  r_us_est = r_us_est/setSz;
  diff = r_us_est - model->u_m[u];
  r_us_est = sigmoid(diff, model->g_k);
  
  return r_us_est;
}


float ModelAvgSigmoid_itemFeatScore(void *self, int u, int item, gk_csr_t *featMat) {
  
  ModelAvgSigmoid *model = self;
  int i, j, ii, jj;
  int nFeats = featMat->rowptr[item+1] - featMat->rowptr[item];
  int *set = (int *) malloc(sizeof(int)*nFeats);
  float score = 0;

  j = 0;
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
     set[j++] = featMat->rowind[ii];
  }
  assert(j == nFeats);
  score = ModelAvgSigmoid_setScore(model, u, set, nFeats, NULL);
  
  free(set);
  return score;
}


float ModelAvgSigmoid_objective(void *self, Data *data, float **sim) {
  
  int u, i, s;
  int setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float se = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0, umRegErr = 0;
  ModelAvgSigmoid *model = self;
  int nSets = 0;
  int uSets = 0;  
  float uNorm = 0, iNorm = 0, umNorm = 0;


  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
    for (s = 0; s < userSet->numSets; s++) {
      if (UserSets_isSetTestVal(userSet, s)) {
        continue;
      }
      uSets++;
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      userSetPref = model->_(setScore) (model, u, set, setSz, NULL);
      diff = userSetPref - userSet->labels[s];
      se += diff*diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    uNorm += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    umRegErr += model->regUm*model->u_m[u]*model->u_m[u];
    umNorm += model->u_m[u]*model->u_m[u];
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim));
    iNorm   += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);

  //TODO: regularization g_k parameter

  //printf("\nse: %f uRegErr: %f umRegErr: %f iRegErr: %f uNorm: %f umNorm:%f iNorm: %f", se, uRegErr,
  //    umRegErr, iRegErr, uNorm, umNorm, iNorm);

  return (se + uRegErr + umRegErr + iRegErr);
}


void ModelAvgSigmoid_trainRMSProp(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet, bestIter;
  int *set;
  float r_us, r_us_est, prevObj, dev, bestObj; 
  float temp; 
  ModelAvgSigmoid *model = self;
  ModelAvgSigmoid *bestModel = NEW(ModelAvgSigmoid, "best avg sigmoid model");
  bestModel->_(init) (bestModel, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate);
  bestModel->u_m = (float*) malloc(sizeof(float)*params->nUsers);
  memset(bestModel->u_m, 0, sizeof(float)*params->nUsers);

  float commGradCoeff = 0, avgRat = 0;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float umGrad         = 0;
  float g_kGrad        = 0;
  float g_kGradAcc     = 0;

  float *umGradAcc    = (float*) malloc(sizeof(float)*params->nUsers);
  memset(umGradAcc, 0, sizeof(float)*params->nUsers);
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  
  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }
  printf("\nnumAllSets: %d", numAllSets);

  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {
      //sample u
      u = rand() % data->nUsers;
      userSet = data->userSets[u];
      if (0 == userSet->numSets) {
        continue;
      }
      //select a non-test non-val set for user
      s = rand() % userSet->numSets;
      if (UserSets_isSetTestVal(userSet, s)) {
        continue;
      }
       
      set       = userSet->uSets[s];
      setSz     = userSet->uSetsSize[s];
      r_us      = userSet->labels[s];
      r_us_est  = 0.0;

      assert(setSz <= MAX_SET_SZ);
      if (setSz == 1) {
        //no update if set contain one item
        continue;
      }

      //perform gradient checks
      if (iter % 100 == 0 && subIter % 100 == 0) {
        ModelAvgSigmoid_gradCheck(model, u, set, setSz, r_us);
        //continue;
      }

      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      avgRat = dotProd(model->_(uFac)[u], sumItemLatFac,  model->_(facDim))/setSz;
      dev = avgRat - model->u_m[u];
      r_us_est = sigmoid(dev, model->g_k);

      //commGradCoeff = exp(-1.0*(dev))*pow(sigmoid(dev), 2);
      commGradCoeff = exp(-1.0*model->g_k*dev)*r_us_est*r_us_est;
      commGradCoeff = 2 * (r_us_est - r_us) * commGradCoeff;
      
      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));  
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = model->g_k*commGradCoeff*sumItemLatFac[j]/setSz;
        //add regularization
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //compute items gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        iGrad[j] = model->g_k*commGradCoeff*model->_(uFac)[u][j]/setSz;
      }
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          //get item gradients by adding regularization
          temp = iGrad[j] + 2.0*model->_(regI)*model->_(iFac)[item][j]; 
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*temp*temp;
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*temp;
        }
      }
      
      
      //compute user midp gradient
      umGrad = commGradCoeff*-1.0*model->g_k + 2.0*model->u_m[u]*model->regUm;
      
      //accumulate gradients square
      umGradAcc[u] = params->rhoRMS*umGradAcc[u] + (1.0 - params->rhoRMS)*umGrad*umGrad;

      //update
      model->u_m[u] -= (model->_(learnRate)/sqrt(umGradAcc[u] + 0.0000001))*umGrad;
       
      /*
      //compute global g_k gradient
      g_kGrad = commGradCoeff*dev + 2.0*model->g_k*model->regG_k;

      //accumulate grad sqr
      g_kGradAcc = params->rhoRMS*g_kGradAcc + (1.0 - params->rhoRMS)*g_kGrad*g_kGrad;
      g_kGradAcc = 1.0;

      //update
      model->g_k -= (model->_(learnRate)/sqrt(g_kGradAcc + 0.0000001))*g_kGrad;
    
      //if (model->g_k < 0) {
      //  model->g_k = 0;
      //}
    
      */
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      if (model->_(isTerminateModel)(model, bestModel, iter, &bestIter, &bestObj, 
          &prevObj, valTest, data)) {
        break;
      }
    }

  }

  printf("\nEnd Obj: %.10e bestObj: %.10e bestIter: %d", valTest->setObj, 
      bestObj, bestIter);
 
  valTest->valSetRMSE = bestModel->_(validationErr) (bestModel, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);

  valTest->trainSetRMSE = bestModel->_(trainErr)(bestModel, data, NULL); 
  printf("\nTrain set error: %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = bestModel->_(testErr) (bestModel, data, NULL); 
  printf("\nTest set error: %f", valTest->testSetRMSE);

  //get maximum rating on individual item in train
  /*
  temp = bestModel->_(getMaxEstTrainRat)(bestModel, data);
  printf("\nmaxRat: %f", temp);
  valTest->testItemsRMSE = bestModel->_(indivItemCSRScaledErr)(bestModel, 
      data->testMat, temp, NULL); 
  printf("\nTest items error indiv: %f", valTest->testItemsRMSE);

  valTest->trainItemsRMSE = bestModel->_(indivTrainSetsScaledErr)(bestModel, 
      data, temp);
  printf("\nTrain items error indiv: %f", valTest->trainItemsRMSE);
  */

  
  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                         data->valMat, 20); 
  //valTest->valSpearman = bestModel->_(coldHitRate)(bestModel, data->userSets,
  //    data->valMat, data->itemFeatMat, data->valItemIds, data->nValItems, 20); 
  printf("\nVal spearman: %f", valTest->valSpearman);

  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                          data->testMat, 20); 
  //valTest->testSpearman = bestModel->_(coldHitRate)(bestModel, data->userSets, 
  //    data->testMat, data->itemFeatMat, data->testItemIds, data->nTestItems, 20); 
  printf("\nTest spearman: %f", valTest->testSpearman);
   

  bestModel->_(copy) (bestModel, model);
  free(bestModel->u_m);
  bestModel->_(free) (bestModel);

  //free up memory
  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);
  free(umGradAcc);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


void ModelAvgSigmoid_trainSGD(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, prevObj; 
  float temp; 
  ModelAvgSigmoid *model = self;
  float commGradCoeff = 0, avgRat = 0;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float umGrad         = 0;

  float *umGradAcc    = (float*) malloc(sizeof(float)*params->nUsers);
  memset(umGradAcc, 0, sizeof(float)*params->nUsers);
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  
  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }
  printf("\nnumAllSets: %d", numAllSets);

  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));

  for (iter = 0; iter < params ->maxIter; iter++) {
    for (subIter = 0; subIter <  5; subIter++) {
      //sample u
      u = rand() % data->nUsers;
      userSet = data->userSets[u];
      //select a non-test non-val set for user
      s = rand() % userSet->numSets;
      if (UserSets_isSetTestVal(userSet, s)) {
        continue;
      }
       
      set       = userSet->uSets[s];
      setSz     = userSet->uSetsSize[s];
      r_us      = userSet->labels[s];
      r_us_est  = 0.0;

      assert(setSz <= 100);
      if (setSz == 1) {
        //no update if set contain one item
        continue;
      }

      //perform gradient checks
      ModelAvgSigmoid_gradCheck(model, u, set, setSz, r_us);
      continue;

      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      avgRat = dotProd(model->_(uFac)[u], sumItemLatFac,  model->_(facDim))/setSz;
      r_us_est = sigmoid(avgRat - model->u_m[u], 1.0);

      //commGradCoeff = exp(-1.0*(avgRat - model->u_m[u]))*pow(sigmoid(avgRat-model->u_m[u]), 2);
      commGradCoeff = exp(-1.0*(avgRat - model->u_m[u]))*r_us_est*r_us_est;
      commGradCoeff = 2 * (r_us_est - r_us) * commGradCoeff;

      /*
      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));  
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = commGradCoeff*sumItemLatFac[j]/setSz;
        //add regularization
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        model->_(uFac)[u][j] -= model->_(learnRate)*uGrad[j];
      }
      */

      //compute items gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        iGrad[j] = commGradCoeff*model->_(uFac)[u][j]/setSz;
      }

      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          //get item gradients by adding regularization
          temp = iGrad[j] + 2.0*model->_(regI)*model->_(iFac)[item][j]; 
          //update
          model->_(iFac)[item][j] -= model->_(learnRate)*temp;
        }
      }

      /*
      //compute user midp gradient
      umGrad = commGradCoeff*-1.0 + 2.0*model->u_m[u]*model->regUm;
      
      //update
      model->u_m[u] -= model->_(learnRate)*umGrad;
      */
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter: %d obj: %f trainRMSE: %f", iter, valTest->setObj, 
          model->_(indivTrainSetsErr) (model, data));
      if (iter > 5000) {
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

  printf("\nEnd Obj: %f", valTest->setObj);
  
  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //free up memory
  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);
  free(umGradAcc);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


void ModelAvgSigmoid_copy(void *self, void *dest) {
  
  int i, j;

  ModelAvgSigmoid *frmModel = self;
  ModelAvgSigmoid *toModel = dest;

  //copy to model
  toModel->_(nUsers)    = frmModel->_(nUsers);
  toModel->_(nItems)    = frmModel->_(nItems);
  toModel->_(regU)      = frmModel->_(regU);
  toModel->_(regI)      = frmModel->_(regI);
  toModel->_(facDim)    = frmModel->_(facDim);
  toModel->_(learnRate) = frmModel->_(learnRate);
  
  //TODO: copy model description

  for (i = 0; i < frmModel->_(nUsers); i++) {
    memcpy(toModel->_(uFac[i]), frmModel->_(uFac[i]), 
        sizeof(float)*frmModel->_(facDim));
  }
  
  for (i = 0; i < frmModel->_(nItems); i++) {
    memcpy(toModel->_(iFac[i]), frmModel->_(iFac[i]), 
        sizeof(float)*frmModel->_(facDim));
  }

  //copy um param
  memcpy(toModel->u_m, frmModel->u_m, sizeof(float)*toModel->_(nUsers));
  
  toModel->rhoRMS = frmModel->rhoRMS;
  toModel->regUm  = frmModel->regUm;
  toModel->g_k    = frmModel->g_k;
  toModel->regG_k = frmModel->regG_k;
}


Model ModelAvgSigmoidProto = {
  .objective               = ModelAvgSigmoid_objective,
  .setScore                = ModelAvgSigmoid_setScore,
  .itemFeatScore           = ModelAvgSigmoid_itemFeatScore,
  .copy                    = ModelAvgSigmoid_copy,
  .train                   = ModelAvgSigmoid_trainRMSProp
};


void modelAvgSigmoid(Data *data, Params *params, ValTestRMSE *valTest) {
  
  int u;
  UserSets *userSet = NULL;
  ModelAvgSigmoid *model = NEW(ModelAvgSigmoid, "sigmoid set prediction");
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);
  model->rhoRMS = params->rhoRMS;
  model->regUm = params->epsRMS;
  model->regG_k = model->regUm;

  printf("\nrhoRMS = %f", params->rhoRMS);
  printf("\nregUm = %f", model->regUm);
  printf("\nregG_k = %f", model->regG_k);

  //assign usermidps randomly between 0 to 5
  if (NULL == data->userMidps) {
    data->userMidps = (float*) malloc(sizeof(float)*data->nUsers);
    memset(data->userMidps, 0, sizeof(float)*data->nUsers);
  }
  for (u = 0; u < data->nUsers; u++) {
    data->userMidps[u] = (float) generateGaussianNoise(2.5, 2);
    //data->userMidps[u] = (float) (rand()%5);
    if (data->userMidps[u] < 0) {
      data->userMidps[u] = 0;
    } else if (data->userMidps[u] > 5) {
      data->userMidps[u] = 5;
    }
  }
  
  //initialize u_m
  model->u_m = (float *) malloc(sizeof(float)*params->nUsers);
  for (u = 0; u < params->nUsers; u++) {
    model->u_m[u] = (float) rand() / (float) (RAND_MAX);  
    //model->u_m[u] = data->userMidps[u]; 
  }

  //initialize steepness parameter
  //model->g_k = (float) rand()/ (float) (RAND_MAX);
  model->g_k = 1.0;

  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  //loadUserItemWtsFrmTrain(data);

  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  //transform set ratings via midP param
  //printf("\nscaling by maxrat: %f", MAX_RAT);
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    UserSets_transToSigm(userSet, data->userMidps);
    //UserSets_scaledTo01(userSet, MAX_RAT); 
  }
  
  //printf("\nData is as follow: ");
  //printf("\n------------------------------------------------------------------");
  //writeData(data); 
  
  //printf("\nModel objective: %f", model->_(objective)(model, data, NULL));
  
  model->_(train) (model, data, params, NULL, valTest); 

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "uFacAvgSigm.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "iFacAvgSigm.txt");

  free(model->u_m);
  model->_(free)(model);
}



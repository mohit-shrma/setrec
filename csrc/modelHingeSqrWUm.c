#include "modelHingeSqrWUm.h"


float hingeSqrLoss(float r_us, float r_us_est) {
  float loss = 0;
  if (r_us_est*r_us < 1.0) {
    loss = (1.0 - r_us*r_us_est)*(1.0 - r_us*r_us_est);
  }
  return loss;
}


void ModelHingeSqrWUm_gradCheck(void *self, int u, int *set, int setSz, float r_us) {
  int i, j, item;
  float commGradCoeff, r_us_est, avgRat;
  ModelHingeSqrWUm *model = self;
  float *sumItemLatFac   = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad           = (float*) malloc(sizeof(float)*model->_(facDim));
  float *iGrad           = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fac             = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float lossRight, lossLeft, gradE, umGrad; 

  //compute common grad coeffs and est score
  memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (j = 0; j < model->_(facDim); j++) {
      sumItemLatFac[j] += model->_(iFac)[item][j];
    }
  }

  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz; 
  r_us_est = avgRat - model->u_m[u];
  commGradCoeff = -1.0*r_us*2.0*(1.0 - r_us*r_us_est);

  //find gradient w.r.t. u
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    if (r_us*r_us_est <= 1.0) {
      uGrad[j] = commGradCoeff*sumItemLatFac[j]/setSz;
    }
    //add regularization
    //uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
  }
  
  //printf("\nr_us_est orig: %f", r_us_est);

  //perturb user with +E and -E and compute Loss 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(uFac)[u][j] + 0.0001;
  }
  avgRat = dotProd(fac, sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = avgRat - model->u_m[u];
  //printf("\nr_us_est +R: %f", r_us_est);
  lossRight = hingeSqrLoss(r_us, r_us_est);
  
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = model->_(uFac)[u][j] - 0.0001;
  }
  avgRat = dotProd(fac, sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = avgRat - model->u_m[u];
  //printf("\nr_us_est -L: %f", r_us_est);
  lossLeft = hingeSqrLoss(r_us, r_us_est);

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*uGrad[j]*0.0001;
  }
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\nu: %d r_us: %f r_us_est:%f lr: %f ll: %f diff: %f div: %f lDiff:%f gradE:%f", u, r_us, r_us_est, 
        lossRight, lossLeft, 
        lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  //find gradient w.r.t. one of the item in set
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  item = set[rand()%setSz];
  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz; 
  r_us_est = avgRat - model->u_m[u];
  for (j = 0; j < model->_(facDim); j++) {
    if (r_us*r_us_est <= 1.0) {
      iGrad[j] = commGradCoeff*model->_(uFac)[u][j]/setSz;
    }
  }

  //perturb item with +E and -E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] + 0.0001;
  }
  avgRat = dotProd(model->_(uFac)[u], fac, model->_(facDim))/setSz; 
  r_us_est = avgRat - model->u_m[u];
  lossRight = hingeSqrLoss(r_us, r_us_est);

  memset(fac, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] - 0.0001;
  }
  avgRat = dotProd(model->_(uFac)[u], fac, model->_(facDim))/setSz; 
  r_us_est = avgRat - model->u_m[u];
  lossLeft = hingeSqrLoss(r_us, r_us_est);

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*iGrad[j]*0.0001;
  }
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\ni: %d r_us: %f r_us_est:%f lr: %f ll: %f diff: %f div: %f lDiff:%f gradE:%f", item, r_us, r_us_est, 
        lossRight, lossLeft, 
        lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  //find gradient w.r.t. u_m
  umGrad = 0;
  if (r_us*r_us_est <= 1.0) {
    umGrad = -1.0*commGradCoeff;
  }

  //perturb u_m with +E and -E
  avgRat = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/setSz;
  r_us_est = avgRat - (model->u_m[u] + 0.0001);
  lossRight = hingeSqrLoss(r_us, r_us_est);

  r_us_est = avgRat - (model->u_m[u] - 0.0001);
  lossLeft = hingeSqrLoss(r_us, r_us_est);
  
  gradE = 2.0*umGrad*0.0001;
  
  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\num: %d diff: %f div: %f lDiff:%f gradE:%f", u, lossRight-lossLeft-gradE, 
      (lossRight-lossLeft)/gradE, lossRight-lossLeft, gradE);
  }

  free(sumItemLatFac);
  free(iGrad);
  free(uGrad);
  free(fac);
}


float ModelHingeSqrWUm_setScore(void *self, int u, int *set, int setSz, 
    float **sim) {
  
  int i, item;
  ModelHingeSqrWUm *model = self;
  float r_us_est = 0, diff;

  for (i = 0; i < setSz; i++) {
    item     = set[i];
    r_us_est += dotProd(model->_(uFac)[u], model->_(iFac)[item],
        model->_(facDim));
  }
  r_us_est = r_us_est/setSz;
  r_us_est = r_us_est - model->u_m[u];
  
  return r_us_est;
}


float ModelHingeSqrWUm_objective(void *self, Data *data, float **sim) {
  
  int u, i, s;
  int setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float se = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0, umRegErr = 0;
  ModelHingeSqrWUm *model = self;
  int nSets = 0;
  int uSets = 0;  
  
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
      diff = hingeSqrLoss(userSet->labels[s], userSetPref);
      se += diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    umRegErr += model->regUm*model->u_m[u]*model->u_m[u];
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);


  //printf("\nse: %f uRegErr: %f umRegErr: %f iRegErr: %f", se, uRegErr, 
  //    umRegErr, iRegErr);

  return (se + uRegErr + umRegErr + iRegErr);
}


void ModelHingeSqrWUm_trainRMSProp(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
  
  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet, bestIter;
  int *set;
  float r_us, r_us_est, prevObj, dev; 
  float temp, bestObj; 
  
  ModelHingeSqrWUm *model = self;
  ModelHingeSqrWUm *bestModel = NULL;
  bestModel = NEW(ModelHingeSqrWUm, "model that achieved lowest obj");
  bestModel->_(init) (bestModel, params->nUsers, params->nItems,
      params->facDim, params->regU, params->regI, params->learnRate);
  bestModel->u_m = (float*) malloc(sizeof(float)*params->nUsers);
  memset(bestModel->u_m, 0, sizeof(float)*params->nUsers);
  
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

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {
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

      assert(setSz <= MAX_SET_SZ);
      if (setSz == 1) {
        //no update if set contain one item
        continue;
      }

      //perform gradient checks
      if (iter % 100 == 0 && subIter % 100 == 0) {
        ModelHingeSqrWUm_gradCheck(model, u, set, setSz, r_us);
      }

      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      avgRat = dotProd(model->_(uFac)[u], sumItemLatFac,  model->_(facDim))/setSz;
      r_us_est = avgRat - model->u_m[u];

      //commGradCoeff = exp(-1.0*(dev))*pow(sigmoid(dev), 2);
      commGradCoeff = -1.0*r_us*2.0*(1.0 - r_us*r_us_est);
      
      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));  
      for (j = 0; j < model->_(facDim); j++) {
        if (r_us*r_us_est <= 1.0) {
          uGrad[j] = commGradCoeff*sumItemLatFac[j]/setSz;
        }
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
      memset(iGrad, 0, sizeof(float)*model->_(facDim));
      for (j = 0; j < model->_(facDim); j++) {
        if (r_us*r_us_est <= 1.0) {
          iGrad[j] = commGradCoeff*model->_(uFac)[u][j]/setSz;
        }
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
      umGrad = 0;
      if (r_us*r_us_est <= 1.0) {
        umGrad = commGradCoeff*-1.0;
      }
      umGrad += 2.0*model->u_m[u]*model->regUm;
      
      //accumulate gradients square
      umGradAcc[u] = params->rhoRMS*umGradAcc[u] + (1.0 - params->rhoRMS)*umGrad*umGrad;

      //update
      model->u_m[u] -= (model->_(learnRate)/sqrt(umGradAcc[u] + 0.0000001))*umGrad;
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      if (model->_(isTerminateClassModel)(model, bestModel, iter, &bestIter, &bestObj, 
          &prevObj, valTest, data)) {
        break;
      }
    }

  }

  valTest->setObj = bestObj;
  printf("\nNum Iter: %d best Obj: %.10e bestIter: %d", iter, bestObj, bestIter);
 
  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->testMat, 10); 
  printf("\nTest spearman: %f", valTest->testSpearman);

  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->valMat, 10);
  printf("\nVal spearman: %f", valTest->valSpearman);

  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationClassLoss) (bestModel, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainSetRMSE = model->_(trainClassLoss)(bestModel, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testClassLoss) (bestModel, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  free(bestModel->u_m);
  bestModel->_(free)(bestModel);

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


void ModelHingeSqrWUm_copy(void *self, void *dest) {
  
  int i, j;

  ModelHingeSqrWUm *frmModel = self;
  ModelHingeSqrWUm *toModel = dest;

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
}


Model ModelHingeSqrWUmProto = {
  .objective = ModelHingeSqrWUm_objective,
  .copy      = ModelHingeSqrWUm_copy,
  .setScore  = ModelHingeSqrWUm_setScore,
  .train     = ModelHingeSqrWUm_trainRMSProp
};


void modelHingeSqrWUm(Data *data, Params *params, ValTestRMSE *valTest) {
  
  int u;
  
  UserSets *userSet = NULL;
  ModelHingeSqrWUm *model = NEW(ModelHingeSqrWUm, "hinge loss set prediction");
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);
  model->rhoRMS = params->rhoRMS;
  model->regUm = params->epsRMS;

  printf("\nrhoRMS = %f", params->rhoRMS);
  printf("\nregUm = %f", model->regUm);

  //assign usermidps randomly between 0 to 5
  /*
  for (u = 0; u < data->nUsers; u++) {
    data->userMidps[u] = (float) generateGaussianNoise(2.5, 2);
    //data->userMidps[u] = (float) (rand()%5);
    if (data->userMidps[u] < 0) {
      data->userMidps[u] = 0;
    } else if (data->userMidps[u] > 5) {
      data->userMidps[u] = 5;
    }
  }
  */

  //initialize u_m
  model->u_m = (float *) malloc(sizeof(float)*params->nUsers);
  for (u = 0; u < params->nUsers; u++) {
    model->u_m[u] = (float) rand() / (float) (RAND_MAX);  
    //model->u_m[u] = data->userMidps[u]; 
  }

  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  loadUserItemWtsFrmTrain(data);

  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  //transform set ratings via midP param
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    UserSets_transToHingeBin(userSet, data->userMidps);
  }

  //printf("\nData is as follow: ");
  //printf("\n------------------------------------------------------------------");
  //writeData(data); 

  model->_(train) (model, data, params, NULL, valTest); 
 
  //write the learned u_m value
  //writeFloatVector(model->u_m, data->nUsers, "um_users.txt");

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "uFacAvgSigm.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "iFacAvgSigm.txt");

  free(model->u_m);
  model->_(free)(model);
}





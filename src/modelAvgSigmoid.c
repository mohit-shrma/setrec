#include "modelAvgSigmoid.h"


float sigmoid(x) {
 return 1.0/(1.0 + exp(-x));
}


float ModelAvgSigmoid_setScore(void *self, int u, int *set, int setSz, 
    float **sim) {
  
  int i, item;
  ModelAvgSigmoid *model = self;
  float r_us_est = 0;

  for (i = 0; i < setSz; i++) {
    item     = set[i];
    r_us_est += dotProd(model->_(uFac)[u], model->_(iFac)[item],
        model->_(facDim));
  }
  r_us_est = r_us_est/setSz;
  r_us_est = sigmoid(r_us_est - model->u_m[u]);
  
  return r_us_est;
}


float ModelAvgSigmoid_objective(void *self, Data *data, float **sim) {
  
  int u, i, s;
  UserSets *userSet = NULL;
  int *set = NULL;
  float se = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelAvgSigmoid *model = self;
  int nSets = 0;
  int uSets = 0;  
  
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
    for (s = 0; s < userSet->numSets; s++) {
      if (UserSets_isSetTestVal(self, s)) {
        continue
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
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  
  return (se + uRegErr + iRegErr);
}


void ModelAvgSigmoid_trainRMSProp(void *self, Data *data, Params *params, float **sim,
    ValTestRMSE *valTest) {
  
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

  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  float *umGradAcc    = (float*) malloc(sizeof(float)*params->nUsers);

  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  memset(uGradsAcc, 0, sizeof(float)*params->nUsers);
  
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
      if (UserSets_isSetTestVal(self, s)) {
        continue
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

      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      avgRat = dotProd(model->_(uFac)[u], sumItemLatFac,  model->_(facDim))/setSz;
      r_us_est = sigmoid(avgRat - model->u_m[u]);

      //TODO: gradient checking
      
      //commGradCoeff = exp(-1.0*(avgRat - model->u_m[u]))*pow(sigmoid(avgRat-model->u_m[u]), 2);
      commGradCoeff = exp(-1.0*(avgRat - model->u_m[u]))*r_us_est*r_us_est;
      commGradCoeff = 2 * (r_us_est - r_us) * commGradCoeff;

      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));  
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = commGradCoeff*sumItemLatFac[j]/setSz;
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
        iGrad[j] = commGradCoeff*model->_(uFac)[u][j]/setSz;
      }
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }
      }

      //compute user midp gradient
      umGrad = commGradCoeff*-1.0;
      
      //accumulate gradients square
      umGradAcc[u] = params->rhoRMS*umGradAcc[u][j] + (1.0 - params->rhoRMS)*umGrad*umGrad;

      //update
      //TODO: init
      model->u_m[u] -= (model->_(learnRate)/(sqrt(umGradAcc[u][j]) + 0.0000001))*umGrad;

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

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);
  
  //free up memory
  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


Model ModelAvgSigmoidProto = {
  .objective = ModelAvgSigmoid_objective,
  .setScore = ModelAvgSigmoid_setScore,
  .train = ModelAvgSigmoid_trainRMSProp
};


void modelAvgSigmoid(Data *data, Params *params, ValTestRMSE *valTest) {
  
  int u;
  UserSets *userSet = NULL;
  ModelAvgSigmoid *model = NEW(ModelAvgSigmoid, "sigmoid set prediction");
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);
  model->rhoRMS = params->rhoRMS;

  printf("\nrhoRMS = %f", params->rhoRMS);
  
  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  loadUserItemWtsFrmTrain(data);

  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  //transform set ratings via midP param
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    UserSets_transToBin(userSet, data->userMidps);
  }
  
  model->_(train) (model, data, params, NULL); 
  
  model->_(free)(model);
}



#include "modelAddSim.h"


void compAddSimUGrad(ModelAddSim *model, int user, int *set, int setSz, 
  float avgSimSet, float r_us, float *sumItemLatFac, float *uGrad) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  
  temp = 2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->_(uFac)[user], 
        sumItemLatFac, model->_(facDim)));
  temp = temp * (-1.0/setSz);  
  
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = sumItemLatFac[j]*temp;
  }

}


//TODO: verify if below computation correct
void compAddSimIGrad(ModelAddSim *model, int user, int item, int *set, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp, nPairs, comDiff;
  
  nPairs = (setSz * (setSz-1))/2;
  memset(iGrad, 0, sizeof(float)*model->_(facDim));

  comDiff = -2.0 * (r_us - (1.0/(1.0*setSz))*dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim)));
  
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = comDiff*((1.0/(1.0/setSz))*model->_(uFac)[user][j] + 
        model->simCoeff[user]*(1.0/nPairs)*(sumItemLatFac[j] - 
          model->_(iFac)[j])); 
  }

}


float compAddSimCoeffGrad(ModelAddSim *model, int user, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac) { 
  
  float grad;

  grad = -2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->_(uFac)[user], 
        sumItemLatFac, model->_(facDim)))*avgSimSet;  
  
  return grad;
}


Model ModelAddSimProto = {
  .objective = ModelAddSim_objective,
  .train     = ModelAddSim_train
};


float ModelAddSim_objective(void *self, Data *data, float **sim) {
  
}


//r_us = 1/|s| u^T sum_i{i \in s} <v_i> + lambdaSim_u*avgSetSim
float ModelAddSim_setScore(void *self, int user, float userSimCoeff, int *set,
    int setSz, float **sim) {
  
  int i, j, k; 
  int item;
  float *sumItemLatFac = NULL;
  float pref = 0.0, avgSimSet = 0.0;
  ModelAddSim *model = self;

  avgSimSet = model->_(setSimilarity)(self, set, setSz, NULL);

  sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->_(facDim); k++) {
      sumItemLatFac[k] += model->_(iFac)[item][k];
    }
  }

  pref = (1.0/setSz)*dotProd(model->uFac[user], itemsLatFac, 
      model->facDim) + userSimCoeff*avgSimSet; 

  free(sumItemLatFac);
  
  return pref;
}


float ModelAddSim_objective(void *self, Data *data, float **sim) {

  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelAddSim *model = self;

  int nSets = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    isTestValSet = 0;
    for (s = 0; s < userSet->numSets; s++) {
      
      //check if set in test sets
      for (i = 0; i < userSet->szTestSet; i++) {
        if (s == userSet->testSets[i]) {
          isTestValSet = 1;
          break;
        }
      }

      //check if set in validation sets
      for (i = 0; i < userSet->szValSet && !isTestValSet; i++) {
        if (s == userSet->valSets[i]) {
          isTestValSet = 1;
          break;
        }
      }
      
      if (isTestValSet) {
        continue;
      }
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      userSetPref = ModelAddSim_setScore(model, u, set, setSz, NULL);
 
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f", diff, userSetPref);
      rmse += diff*diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
    

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f ", (rmse+uRegErr+iRegErr),
      rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


void ModelAddSim_train(void *self, Data *data, Params *params, float **sim,
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us, prevVal; //actual set preference
  float avgSetSim, temp; 
  ModelAddSim *model   = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  prevVal = 0.0;

  model->_(objective)(model, data, sim);
  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
      userSet = data->userSets[u];
      isTestValSet = 0;
      //select a non-test non-val set for user
      s = rand() % userSet->numSets;
      //check if set in test sets
      for (i = 0; i < userSet->szTestSet; i++) {
        if (s == userSet->testSets[i]) {
          isTestValSet = 1;
          break;
        }
      }
      //check if set in validation sets
      for (i = 0; i < userSet->szValSet && !isTestValSet; i++) {
        if (s == userSet->valSets[i]) {
          isTestValSet = 1;
          break;
        }
      }
      
      if (isTestValSet) {
        continue;
      }
      
      set       = userSet->uSets[s];
      setSz     = userSet->uSetsSize[s];
      r_us      = userSet->labels[s];
      
      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }

      avgSetSim = model->_(setSimilarity)(self, set, setSz, sim);
      //avgSetSim = model->_(setSimilarity)(self, set, setSz, NULL);
      
      //update user and item latent factor
     
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }

      //compute user gradient
      computeUGrad(model, u, set, setSz, avgSetSim, r_us, sumItemLatFac, uGrad);
      //update user + reg
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGrad(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim));
      }
      
      //update user sim coeff grad
      model->simCoeff[u] = model->simCoeff[u] - model->_(learnRate)*compAddSimCoeffGrad(model, user, setSz, avgSetSim, r_us, sumItemLatFac); 
    }

    //update sim
    model->_(updateSim)(model, sim);
    
    //objective check
    if (iter % OBJ_ITER == 0) {
      printf("\nuserFacNorm:%f itemFacNorm:%f", model->_(userFacNorm)(self, data), model->_(itemFacNorm)(self, data));

      model->_(objective)(model, data, sim);
    }

    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest[0] = model->_(indivItemSetErr) (model, data->valSet);
      printf("\nIter:%d validation error: %f", iter, valTest[0]);
      if (iter > 0) {
        if (fabs(prevVal - valTest[0]) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest[0], fabs(prevVal - valTest[0]));
          break;
        }
      }
      prevVal = valTest[0];
    }
    
  }

  
  //get train error
  //printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));

  //get test eror
  valTest[1] = model->_(indivItemSetErr) (model, data->testSet);

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);

}


void ModelAddSim_init(void *self, Params *params) {
  int i;
  ModelAddSim *model = self;
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);
  model->simCoeff = (float*) malloc(sizeof(float)*params->nUsers);
  memset(model->simCoeff, 0, sizeof(float)*params->nUsers);
  for (i = 0; i < params->nUsers; i++) {
    model->simCoeff[i] = (float) rand() / (float) (RAND_MAX); 
  }
}


void ModelAddSim_free(void *self) {
  ModelAddSim *model = self;
  free(model->simCoeff);
  model->_(free)(model);
}


void modelAddSim(Data *data, Params *params, float *valTest) {

  float **sim = NULL;
  int i, j, k;

  //allocate storage for model
  ModelAddSim *model = NEW(ModelAddSim, "set prediction with added sim");
  ModelAddSim_init(model, params);
  
  //allocate sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  model->_(updateSim) (model, sim);

  //train model
  model->_(train)(model, data, params, sim, valTest);

  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }

  ModelAddSim_free(model);
}




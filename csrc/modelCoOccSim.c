#include "modelCoOccSim.h"


void computeUGradCoOccSim(ModelCoOccSim *model, int user, int *set, int setSz, 
    float r_us, float *sumItemLatFac, float *uGrad) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  
  temp = 2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->_(uFac)[user], 
        sumItemLatFac, model->_(facDim)));
  temp = temp * (-1.0/setSz);
  //printf("\ndotProd = %f norm sumItemLatFac = %f uFac = %f", dotProd(model->_(uFac)[user], 
  //      sumItemLatFac, model->_(facDim)), norm(sumItemLatFac, model->_(facDim)), norm(model->_(uFac)[user], model->_(facDim)));
  //printf("\n temp = %f setSz = %d avgSimSet = %f", temp, setSz, avgSimSet);
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = sumItemLatFac[j]*temp;
  }

}


void computeIGradCoOccSim(ModelCoOccSim *model, int user, int item, int *set, int setSz, 
    float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp,  comDiff;
  
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  
  comDiff = r_us - (1.0/(1.0*setSz))*dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim));
  temp = 2.0 * comDiff * (-1.0/(1.0*setSz));
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = temp*model->_(uFac)[user][j];
  }

}


float ModelCoOccSim_objective(void *self, Data *data, float **sim) {
  
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0, setSim = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelCoOccSim *model = self;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      isTestValSet = 0;
      
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
      userSetPref = model->_(setScore)(model, u, set, setSz, NULL);
      //set sim
      setSim = model->_(setSimilarity)(self, set, setSz, sim);   
 
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
      rmse += diff*diff*setSim;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  //printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr),
  //    rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


//TODO:check if coefficient norm update required
void ModelCoOccSim_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us, prevVal; //actual set preference
  float temp, avgSetSim; 
  ModelCoOccSim *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  prevVal = 0.0;

  printf("\nInitial Objective val: %f", model->_(objective)(model, data, sim));
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

      //dont update no. of items in set is 1
      if (setSz == 1) {
        continue;
      }

      //get pairwise average similarity of set
      avgSetSim = model->_(setSimilarity) (self, set, setSz, sim);

      //update user and latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      
      //compute user gradient
      computeUGradCoOccSim(model, u, set, setSz, r_us, sumItemLatFac, uGrad);
      //update user + reg
      for (k = 0; k < model->_(facDim); k++) {
        model->_(uFac)[u][k] -= model->_(learnRate)*avgSetSim*(uGrad[k] + model->_(regU)*model->_(uFac)[u][k]);
      }
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGradCoOccSim(model, u, item, set, setSz, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        for (k = 0; k < model->_(facDim); k++) {
          model->_(iFac)[item][k] -= model->_(learnRate)*avgSetSim*(iGrad[k] + model->_(regI)*model->_(iFac)[item][k]);
        }
      }

    }

    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest[0] = model->_(indivItemSetErr) (model, data->valSet);
      //printf("\nIter:%d validation error: %f", iter, valTest[0]);
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
   
    
    //objective check
    /*
    if (iter % OBJ_ITER == 0) {
      model->_(objective)(model, data, sim);
    }
    */
    
  }

  printf("\nFinal objective val: %f", model->_(objective)(model, data, sim));
  
  //get test error
  valTest[1] = model->_(indivItemSetErr) (model, data->testSet);
  
  //model->_(writeUserSetSim)(self, data, "userSetsWOSim.txt");

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);

}


Model ModelCoOccSimProto = {
  .objective = ModelCoOccSim_objective,
  .train = ModelCoOccSim_train
};


void modelCoOccSim(Data *data, Params *params, float *valTest) {
  
  int i, j;
  float **sim = NULL;

  //allocate storage for model
  ModelCoOccSim *modelCoOccSim = NEW(ModelCoOccSim, 
      "set pred using co occ jaccard similarity");
  modelCoOccSim->_(init) (modelCoOccSim, params->nUsers, params->nItems, 
      params->facDim, params->regU, params->regI, params->learnRate);  

  //allocate space for sim
  if (params->useSim) {
    sim = (float **) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  //compute jaccard similarities
  //Data_jaccSim(data, sim);   
  
  loadItemSims(params, sim);

  //writeUpperMat(sim, data->nItems, data->nItems, "mLensSimC2.txt");
  //modelCoOccSim->_(writeUserSetSim)(modelCoOccSim, data, sim, "mLensTagSetSim.txt");  

  //train model
  modelCoOccSim->_(train)(modelCoOccSim, data, params, sim, valTest);


  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }

  modelCoOccSim->_(free)(modelCoOccSim);
}




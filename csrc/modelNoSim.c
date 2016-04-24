#include "modelNoSim.h"


void computeUGrad2(ModelNoSim *model, int user, int *set, int setSz, 
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


void computeIGrad2(ModelNoSim *model, int user, int item, int *set, int setSz,
    float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp, comDiff;
  
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  
  comDiff = r_us - (1.0/(1.0*setSz))*dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim));
  temp = 2.0 * comDiff * (-1.0/(1.0*setSz));
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = temp*model->_(uFac)[user][j];
  }

}


Model ModelNoSimProto = {
  .objective        = ModelNoSim_objective,
  .train            = ModelNoSim_train
};



float ModelNoSim_objective(void *self, Data *data, float **sim) {

  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelNoSim *model = self;

  int nSets = 0;

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
 
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
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
  printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr),
      rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}



void coeffUpdate2(float *fac, float *grad, float reg, float learnRate, int facDim) {

  int k;
  float facNorm;

  for (k = 0; k < facDim; k++) {
    fac[k] -= learnRate*(grad[k] + reg*fac[k]); 
    /*
    if (fac[k] < 0) {
      fac[k] = 0;
    }
    */
  }

}



void coefficientNormUpdate2(float *fac, float *grad, float reg, float learnRate, int facDim) {

  int k;
  float facNorm;

  for (k = 0; k < facDim; k++) {
    fac[k] -= learnRate*(grad[k] + reg*fac[k]); 
  }

  //normalize factor
  facNorm = norm(fac, facDim);
  for (k = 0; k < facDim; k++) {
    fac[k] = fac[k]/facNorm;
  }

}


void ModelNoSim_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us, prevVal; //actual set preference
  float temp; 
  ModelNoSim *model         = self;
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

      //update user and item latent factor
     
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      
      //compute user gradient
      computeUGrad2(model, u, set, setSz, r_us, sumItemLatFac, uGrad);
      //update user + reg
      //printf("\nb4 u = %d norm uFac = %f uGrad = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)), norm(uGrad, model->_(facDim)));
      
      coeffUpdate2(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //printf("\naftr u = %d norm uFac = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)));
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGrad2(model, u, item, set, setSz, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        //printf("\nitem = %d norm iGrad = %f", item, norm(iGrad, model->_(facDim)));
        coeffUpdate2(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim));
      }

    }

    
    //objective check
    if (iter % OBJ_ITER == 0) {
      printf("\nuserFacNorm:%f itemFacNorm:%f", model->_(userFacNorm)(self, data), model->_(itemFacNorm)(self, data));

      model->_(objective)(model, data, NULL);
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



void modelNoSim(Data *data, Params *params, float *valTest) {
 
  //allocate storage for model
  ModelNoSim *modelNoSim = NEW(ModelNoSim, "set prediction model with sim");
  modelNoSim->_(init)(modelNoSim, params->nUsers, params->nItems, params->facDim, params->regU, 
    params->regI, params->learnRate);

  
  //train model 
  modelNoSim->_(train)(modelNoSim, data,  params, NULL, valTest);

  modelNoSim->_(free)(modelNoSim);
}




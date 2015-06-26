#include "modelHingeProdSim.h"


void computeUGradHingeProdSim(MHingeProdSim *model, int user, int *set, int setSz, 
  float avgSimSet, float r_us, float *sumItemLatFac, float *uGrad, float setPref) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->_(facDim));
 
  if (setPref*r_us < 1) {
    //compute gradient
    temp = (-1.0*r_us*avgSimSet)/setSz;
    for (j = 0; j < model->_(facDim); j++) {
      uGrad[j] = sumItemLatFac[j]*temp;
    }
  }

}


void computeIGradHingeProdSim(MHingeProdSim *model, int user, int item, int *set, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac, float *iGrad, float setPref) {
  
  int i, j;
  float temp, nPairs;
  
  nPairs = (setSz * (setSz-1))/2;
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  
  //NOTE: make sure passed fresh avgSimSet setPref
  if (setPref*r_us < 1) {
    //compute gradient
    temp = dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim)) * (1.0/nPairs);

    for (j = 0; j < model->_(facDim); j++) {
      iGrad[j] = temp * (sumItemLatFac[j] - model->_(iFac)[item][j]);
      iGrad[j] += model->_(uFac)[user][j]*avgSimSet;  
      iGrad[j] = iGrad[j]/((-1.0*r_us)/setSz);
    }
  }

}


void simNsetScore(void *self, int user, int *set, int setSz, 
    float **sim, float *avgSimSet, float *setScore) {

  int i, j, k;
  int item, nPairs;
  float *itemsLatFac = NULL;
  float pref = 0.0, avgSetSim = 0.0;
  MHingeProdSim *model = self;

  itemsLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(itemsLatFac, 0, sizeof(float)*model->_(facDim));

  nPairs = (setSz * (setSz - 1))/2;

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->_(facDim); k++) {
      itemsLatFac[k] += model->_(iFac)[item][k];
    }
    for (j = i+1; j < setSz; j++) {
      if (sim != NULL) {
        avgSetSim += sim[set[i]][set[j]];
      } else {
        avgSetSim += dotProd(model->_(iFac)[set[i]], model->_(iFac)[set[j]], 
            model->_(facDim));
      }
    }
  }
  
  if (0 == nPairs) {
    avgSetSim = 1.0;
  } else {
    avgSetSim = avgSetSim/nPairs;
  }

  pref = (1.0/setSz)*dotProd(model->_(uFac)[user], itemsLatFac, 
      model->_(facDim))*avgSetSim; 
  
  *avgSimSet = avgSetSim;
  *setScore = pref;

  free(itemsLatFac);
}


//r_us = (1/|s| u^T sum<i \n s>(v_i))sim(s)
float MHingeProdSim_setScore(void *self, int user, int *set, int setSz, 
    float **sim) {
  
  int i, j, k;
  int item, nPairs;
  float *itemsLatFac = NULL;
  float pref = 0.0, avgSetSim = 0.0;
  MHingeProdSim *model = self;

  itemsLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(itemsLatFac, 0, sizeof(float)*model->_(facDim));

  nPairs = (setSz * (setSz - 1))/2;

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->_(facDim); k++) {
      itemsLatFac[k] += model->_(iFac)[item][k];
    }
    for (j = i+1; j < setSz; j++) {
      if (sim != NULL) {
        avgSetSim += sim[set[i]][set[j]];
      } else {
        avgSetSim += dotProd(model->_(iFac)[set[i]], model->_(iFac)[set[j]], 
            model->_(facDim));
      }
    }
  }
  
  if (0 == nPairs) {
    avgSetSim = 1.0;
  } else {
    avgSetSim = avgSetSim/nPairs;
  }


  pref = (1.0/setSz)*dotProd(model->_(uFac)[user], itemsLatFac, 
      model->_(facDim))*avgSetSim; 

  free(itemsLatFac);
  
  return pref; 
}


//hing loss across all sets + regularization terms
float MHingeProdSim_objective(void *self, Data *data, float **sim) {

  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float loss = 0, hingeLoss = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  MHingeProdSim *model = self;

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
      userSetPref = model->_(setScore)(model, u, set, setSz, sim);
 
      if (1 <= userSetPref*userSet->labels[s]) {
        hingeLoss = 0;
      } else {
        hingeLoss = 1.0 - userSetPref*userSet->labels[s];  
      }
      
      loss += hingeLoss;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  printf("\nObj: %f loss: %f uRegErr: %f iRegErr: %f", (loss+uRegErr+iRegErr),
      loss, uRegErr, iRegErr);
  return (loss + uRegErr + iRegErr);
}


void MHingeProdSim_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item, isNan;
  int *set;
  float r_us, prevVal, obj, prevObj; //actual set preference
  float avgSetSim, setPref, temp; 
  MHingeProdSim *model         = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  prevVal = 0.0;
  isNan = 0;

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
      assert(setSz > 1);

      simNsetScore(model, u, set, setSz, NULL, &avgSetSim, &setPref);
      
      //update user and item latent factor
     
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      
      //compute user gradient
      computeUGradHingeProdSim(model, u, set, setSz, avgSetSim, r_us, 
          sumItemLatFac, uGrad, setPref);
    
      /*
      if (setPref*r_us > 1) {
        //dont update on correct classification
        continue;
      }
      */

      //update user + reg
      for (k = 0; k < model->_(facDim); k++) {
        model->_(uFac)[u][k] -= model->_(learnRate)*(uGrad[k] + model->_(regU)*model->_(uFac)[u][k]);
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGradHingeProdSim(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad, setPref);
        //update item + reg
        for (k = 0; k < model->_(facDim); k++) {
          model->_(iFac)[item][k] -= model->_(learnRate)*(iGrad[k] + model->_(regI)*model->_(iFac)[item][k]);
        }
        //coefficientNormUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
        //    model->_(learnRate), model->_(facDim));
      }

    }
    
    //update sim
    model->_(updateSim)(model, sim);
    
    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest[0] = model->_(validationClassLoss) (model, data, sim);
      //valTest[0] = model->_(hingeValidationErr) (model, data, sim);
      printf("\nIter:%d validation error: %f", iter, valTest[0]);
      if (VAL_CONV && iter > 0) {
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
    if (iter % OBJ_ITER == 0) {
      obj = model->_(objective)(model, data, sim);
      
      if (obj != obj) {
        printf("\nFound NaN");
        isNan = 1;
        break;
      }
      
      if (OBJ_CONV && iter > 0) {
        if (fabs(prevObj - obj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter, 
              prevObj, obj);
          break;
        }
      }
      prevObj = obj;
    }
    
  }
  
  //get train error
  //printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));

  obj = model->_(objective)(model, data, sim);
 
  //get validation error
  valTest[0] = model->_(validationClassLoss) (model, data, sim);
  //valTest[0] = model->_(validationErr) (model, data, sim);
  //get test eror
  valTest[1] = model->_(testClassLoss) (model, data, sim);
  //valTest[1] = model->_(testErr) (model, data, sim);

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


Model MHingeProdSimProto = {
  .objective  = MHingeProdSim_objective,
  .train      = MHingeProdSim_train,
  .setScore   = MHingeProdSim_setScore
};


void mHingeProdSim(Data *data, Params *params, float *valTest) {

  float **sim = NULL;
  int i, j, k;

  //allocate model storage
  MHingeProdSim *model = NEW(MHingeProdSim, "set prediction with sim using hinge loss"); 
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);
  model->_(validationClassLoss) (model, data, sim);

  //allocate space for sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }
  model->_(updateSim)(model, sim);

  //train model
  model->_(train)(model, data, params, sim, valTest);

  //free up allocated space
  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }

  model->_(free)(model);

}




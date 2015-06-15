
#include "modelSim.h"

//r_us = 1/|s| u^T sum_{i \in s}<v_i>
float ModelSim_setScore(void *self, int user, int *set, int setSz, float **sim) {
  
  int i, j, k;
  int item;
  float *itemsLatFac = NULL;
  float pref = 0.0;
  ModelSim *model = self;

  itemsLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(itemsLatFac, 0, sizeof(float)*model->_(facDim));

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->_(facDim); k++) {
      itemsLatFac[k] += model->_(iFac)[item][k];
    }
  }

  pref = (1.0/setSz)*dotProd(model->_(uFac)[user], itemsLatFac, 
      model->_(facDim)); 

  //printf("\nnorm(itemsLatFac) = %f, pref = %f ", norm(itemsLatFac, 
  //      model->_(facDim)), pref);

  free(itemsLatFac);
  
  return pref;
}


//TODO: put in common place
float setSimilarity(int *set, int setSz, void *self, float **sim) {
  float setSim = 0.0;
  int i, j, nPairs = 0;
  ModelSim *model = self;
  nPairs = (setSz * (setSz-1)) / 2;
  if (sim != NULL) {
    for (i = 0; i < setSz; i++) {
      for (j = i+1; j < setSz; j++) {
        setSim += sim[set[i]][set[j]];
      }
    }
  } else {
    for (i = 0; i < setSz; i++) {
      for (j = i+1; j < setSz; j++) {
        //printf("\n%f %f", model->_(iFac)[set[i]], model->_(iFac)[set[j]]);
        setSim += dotProd(model->_(iFac)[set[i]], model->_(iFac)[set[j]], 
            model->_(facDim));
      }
    }
  }
  //printf("\nsetSim = %f, nPairs = %d, setSz = %d", setSim, nPairs, setSz);
  setSim = setSim/nPairs;

  if (nPairs == 0) {
    setSim = 1.0;
  }

  return setSim;
}


void computeUGrad(ModelSim *model, int user, int *set, int setSz, 
  float avgSimSet, float r_us, float *sumItemLatFac, float *uGrad) {
  int i, j;
  float temp;

  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  
  temp = 2.0 * (r_us - (1.0/(setSz*1.0))*dotProd(model->_(uFac)[user], 
        sumItemLatFac, model->_(facDim)));
  temp = temp * (-1.0/setSz) * avgSimSet;
  //printf("\ndotProd = %f norm sumItemLatFac = %f uFac = %f", dotProd(model->_(uFac)[user], 
  //      sumItemLatFac, model->_(facDim)), norm(sumItemLatFac, model->_(facDim)), norm(model->_(uFac)[user], model->_(facDim)));
  //printf("\n temp = %f setSz = %d avgSimSet = %f", temp, setSz, avgSimSet);
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = sumItemLatFac[j]*temp;
  }

}


void computeIGrad(ModelSim *model, int user, int item, int *set, int setSz, 
    float avgSimSet, float r_us, float *sumItemLatFac, float *iGrad) {
  
  int i, j;
  float temp, nPairs, comDiff;
  
  nPairs = (setSz * (setSz-1))/2;
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  
  comDiff = r_us - (1.0/(1.0*setSz))*dotProd(model->_(uFac)[user], sumItemLatFac, model->_(facDim));
  temp = 2.0 * comDiff * (-1.0/(1.0*setSz)) * avgSimSet;
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = temp*model->_(uFac)[user][j];
  }

  temp = comDiff*comDiff*(1.0/nPairs);
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] += temp*(sumItemLatFac[j] - model->_(iFac)[item][j]);
  }

}


Model ModelSimProto = {
  .objective        = ModelSim_objective,
  .setScore         = ModelSim_setScore,
  .train            = ModelSim_train
};


float userSetLossU(void *self, UserSets *userSet, int setInd, float *uFac) {
  float loss, r_us, avgSimSet;
  int user, i, j;
  float *sumItemFac = NULL;
  ModelSim *model = self;
  
  loss = 0;
  user = userSet->userId;
  r_us = userSet->labels[setInd];
  avgSimSet = setSimilarity(userSet->uSets[setInd], userSet->uSetsSize[setInd],
      self, NULL); 

  sumItemFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(sumItemFac, 0, sizeof(float)*model->_(facDim));

  if (uFac == NULL) {
    uFac = model->_(uFac)[user];
  }
 
  //sum item's latent factor in set
  for (i = 0; i < userSet->uSetsSize[setInd]; i++) {
    for (j = 0; j < model->_(facDim); j++) {
      sumItemFac[j] += model->_(iFac)[userSet->uSets[setInd][i]][j];
    }
  }
  
  loss = (r_us - (1.0/userSet->uSetsSize[setInd])*dotProd(uFac,
        sumItemFac, model->_(facDim)) );
  loss = loss * loss * avgSimSet;

  free(sumItemFac);

  return loss;
}


float userSetLossI(void *self, UserSets *userSet, int setInd, int item, 
    float *iFac) {
  float loss, r_us, avgSimSet;
  int user, i, j, k, nPairs;
  float *sumItemFac = NULL;
  ModelSim *model = self;

  loss = 0;
  user = userSet->userId;
  r_us = userSet->labels[setInd];
  avgSimSet = 0;
  nPairs = (userSet->uSetsSize[setInd] * (userSet->uSetsSize[setInd] - 1))/2;
  sumItemFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(sumItemFac, 0, sizeof(float)*model->_(facDim));
  
  if (iFac ==  NULL) {
    iFac = model->_(iFac)[item];
  }

  //compute sum of item factors in set
  for (i = 0; i < userSet->uSetsSize[setInd]; i++) {
    //get item from set
    j = userSet->uSets[setInd][i]; 
    if (j != item) {
      for (k = 0; k < model->_(facDim); k++) {
        sumItemFac[k] += model->_(iFac)[j][k];
      }
    }
  }
  //add the item latent factor to set
  for (k = 0; k < model->_(facDim); k++) {
    sumItemFac[k] += iFac[k];
  }

  //compute average similarity of items in set
  for (i = 0; i < userSet->uSetsSize[setInd]; i++) {
    for (j = i+1; j < userSet->uSetsSize[setInd]; j++) {
      if (userSet->uSets[setInd][i] == item || userSet->uSets[setInd][j] == item) {
        continue;
      }
      avgSimSet += dotProd(model->_(iFac)[userSet->uSets[setInd][i]], 
          model->_(iFac)[userSet->uSets[setInd][j]], 
          model->_(facDim));
    }
  }
  for (i = 0; i < userSet->uSetsSize[setInd]; i++) {
    if (item != userSet->uSets[setInd][i]) {
      avgSimSet += dotProd(model->_(iFac)[userSet->uSets[setInd][i]],
          iFac, model->_(facDim));
    }
  } 
  avgSimSet = avgSimSet/nPairs;

  loss = (r_us - (1.0/userSet->uSetsSize[setInd])*dotProd(model->_(uFac)[user],
        sumItemFac, model->_(facDim)));
  loss = loss * loss * avgSimSet;

  free(sumItemFac);

  return loss;
}


void ModelSim_gradCheck(void *self, UserSets *userSet) {

  int i, j, k;
  int setInd, itemInd, item;
  float *uFac, *uFacTmp, *uFacTmp2;
  float *iFac, *iFacTmp, *iFacTmp2;
  float *uGrad, *iGrad, *delta, *sumItemLatFac;
  float userLoss, userLossTemp1, userLossTemp2, avgSimSet;
  float itemLossTemp1, itemLossTemp2;
  float diff, div;
  ModelSim *model = self;


  fflush(stdout);

  setInd = rand() % userSet->numSets;
  while(userSet->uSetsSize[setInd] <= 1) {
    setInd = rand() % userSet->numSets;
  }
  
  userLoss = userSetLossU(self, userSet, setInd, NULL);
  
  uFacTmp = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uFacTmp, 0, (sizeof(float)*model->_(facDim)));
  
  uFacTmp2 = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uFacTmp2, 0, (sizeof(float)*model->_(facDim)));
  
  uFac = model->_(uFac)[userSet->userId];

  uGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uGrad, 0, (sizeof(float)*model->_(facDim)));
  
  delta = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(delta, 0, (sizeof(float)*model->_(facDim)));
 
  sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(sumItemLatFac, 0, (sizeof(float)*model->_(facDim)));
  
  avgSimSet = setSimilarity(userSet->uSets[setInd], userSet->uSetsSize[setInd], 
      self, NULL);

  for (i = 0; i < userSet->uSetsSize[setInd]; i++) {
    for (k = 0; k < model->_(facDim); k++) {
      sumItemLatFac[k] += model->_(iFac)[userSet->uSets[setInd][i]][k];
    }
  }

  computeUGrad(model, userSet->userId, userSet->uSets[setInd], 
      userSet->uSetsSize[setInd],
      avgSimSet, userSet->labels[setInd], sumItemLatFac, uGrad);

  for (i = 0; i < 5; i++) {
    for (k = 0; k < model->_(facDim); k++) {
      //TODO: perturbation of 0.001 scale
      delta[k] = ((float)rand()/(float)(RAND_MAX)) * 0.001;
      uFacTmp[k] = uFac[k] + delta[k];
      uFacTmp2[k] = uFac[k] - delta[k];
    }
    userLossTemp1 = userSetLossU(self, userSet, setInd, uFacTmp);
    userLossTemp2 = userSetLossU(self, userSet, setInd, uFacTmp2);
    diff = userLossTemp1 - userLoss - dotProd(delta, uGrad, model->_(facDim));
    div = ((userLossTemp1 - userLossTemp2)/2.0)/dotProd(delta, uGrad, 
        model->_(facDim));
    printf("\nuser grad check diff = %f, div = %f",  diff, div);
    fflush(stdout);
  }

  
  itemInd = rand()%userSet->uSetsSize[setInd];
  item = userSet->uSets[setInd][itemInd];

  iGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iGrad, 0, (sizeof(float)*model->_(facDim)));

  iFac = model->_(iFac)[item];

  iFacTmp = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iFacTmp, 0, (sizeof(float)*model->_(facDim)));
  
  iFacTmp2 = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iFacTmp2, 0, (sizeof(float)*model->_(facDim)));
 
  computeIGrad(model, userSet->userId, item, userSet->uSets[setInd], 
      userSet->uSetsSize[setInd], avgSimSet, userSet->labels[setInd], 
      sumItemLatFac, iGrad);
  
  for (i = 0; i < 5; i++) {
    for (k = 0; k < model->_(facDim); k++) {
      delta[k] = ((float)rand()/(float)(RAND_MAX)) * 0.001;
      iFacTmp[k] = iFac[k] + delta[k];
      iFacTmp2[k] = iFac[k] - delta[k];
    }
    
    itemLossTemp1 = userSetLossI(model, userSet, setInd, item, iFacTmp);
    itemLossTemp2 = userSetLossI(model, userSet, setInd, item, iFacTmp2);
    
    diff = itemLossTemp1 - userLoss - dotProd(delta, iGrad, model->_(facDim));
    div = ((itemLossTemp1 - itemLossTemp2)/2.0)/dotProd(delta, iGrad, model->_(facDim));
    printf("\nitem gad check diff = %f div = %f", diff, div);
  }
  fflush(stdout);

  free(uFacTmp);
  free(uFacTmp2);
  free(iFacTmp);
  free(iFacTmp2);
  free(uGrad);
  free(iGrad);
  free(delta);
  free(sumItemLatFac);
}


//TODO: need sim matrix in computing objective
//different name to avoid ambiguous match in model.c
float ModelSim_objective(void *self, Data *data) {

  int u, i, s, item, setSz;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0, setSim = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelSim *model = self;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      //TODO
      userSetPref = model->_(setScore)(model, u, set, setSz, NULL);
      //set sim
      setSim = setSimilarity(set, setSz, model, NULL);   
 
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
      rmse = diff*diff*setSim;
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


void ModelSim_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us; //actual set preference
  float avgSetSim, temp; 
  ModelSim *model         = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));

  model->_(objective)(model, data);
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

      //TODO: dont update no. of items in set is 1
      if (setSz == 1) {
        continue;
      }

      //TODO: verify whether similarity with and without sim are same
      avgSetSim = setSimilarity(set, setSz, model, sim);
      //update user and latent factor
     
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
      //printf("\nb4 u = %d norm uFac = %f uGrad = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)), norm(uGrad, model->_(facDim)));
      for (k = 0; k < model->_(facDim); k++) {
        model->_(uFac)[u][k] -= model->_(learnRate)*(uGrad[k] + model->_(regU)*model->_(uFac)[u][k]);
      }
      //printf("\naftr u = %d norm uFac = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)));
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //get item gradient
        computeIGrad(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        //printf("\nitem = %d norm iGrad = %f", item, norm(iGrad, model->_(facDim)));
        for (k = 0; k < model->_(facDim); k++) {
          model->_(iFac)[item][k] -= model->_(learnRate)*(iGrad[k] + model->_(regI)*model->_(iFac)[item][k]);
        }
      }

    }

    //update sim
    model->_(updateSim)(model, sim);
    //objective check
    if (iter % 2 == 0) {
      model->_(objective)(model, data);
      //validation err
      valTest[0] = model->_(validationErr) (model, data, sim);
    }
    
  }

  //get test eror
  valTest[1] = model->_(testErr) (model, data, sim);

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


void modelSim(Data *data, Params *params, float *valTest) {
 
  float **sim  = NULL;

  int i, j, k;

  //init random with seed
  srand(params->seed);

  //allocate storage for model
  ModelSim *modelSim = NEW(ModelSim, "set prediction model with sim");
  modelSim->_(init)(modelSim, params->nUsers, params->nItems, params->facDim, params->regU, 
    params->regI, params->learnRate);

  //allocate space for sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }
  modelSim->_(updateSim)(modelSim, sim);

  //perform gradient check
  //for (i = 0; i < 100; i++) {
  //  ModelSim_gradCheck(modelSim, data->userSets[rand()%data->nUsers]);
  //}
  
  //train model 
  modelSim->_(train)(modelSim, data,  params, sim, valTest);

  //free up allocated space
  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }
  
  modelSim->_(free)(modelSim);
  
}







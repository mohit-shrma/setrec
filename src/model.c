#include "model.h"


void updateSim(float **sim, Model *model) {
  int i, j;
  for (i = 0; i < model->nItems; i++) {
    sim[i][i] = 1.0;
    for (j = i+1; j < model->nItems; j++) {
      sim[i][j] = sim[j][i] = dotProd(model->iFac[i], model->iFac[j], model->facDim);
    }
  }
}


float computeObjective(Data *data, Model *model) {
  
  int u, i, item;
  UserSets *userSet;
  float rmse = 0, diff = 0;
  float uRegErr = 0, iRegErr = 0;

  for (u = 0 ; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      //ignore if item not in train
      if (userSet->testValItems[item]) {
        continue;
      }
      diff = userSet->itemWtSets[i]->wt - dotProd(model->uFac[u], model->iFac[item], model->facDim);
      rmse += diff*diff;
    }
    uRegErr += dotProd(model->uFac[u], model->uFac[u], model->facDim);
  }
  uRegErr *= model->regU;

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->iFac[i], model->iFac[i], model->facDim);
  }
  iRegErr *= model->regI;
  printf("\nObj: %f RMSE: %f uRegErr: %f iRegErr: %f", (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr);
  return (rmse + uRegErr + iRegErr);
}


float *setScores(int *sets, int setsSz, UserSets *userSet) {
  
  int i, j;
  int item, setSz;
  ItemWtSets *itemWtSets;
  float *scores = (float*) malloc(sizeof(float)*setsSz);
  memset(scores, 0, sizeof(float)*setsSz);

  for (i = 0; i < setsSz; i++) {
    setSz = userSet->uSetsSize[sets[i]];
    for (j = 0; j < setSz; j++) {
      item = userSet->uSets[sets[i]][j];
      itemWtSets = UserSets_search(userSet, item);
      scores[i] += itemWtSets->wt;    
    }
  }
  return scores;
}


float setScoreLatFac(int *set, int setSz, Model *model, UserSet *userSet) {
  int i, j;
  int item;
  float score = 0;
  for (i = 0; i < setSz; i++) {
    item = set[i];
    score += dotProd(model->uFac[userSet->userId], model->iFac[item], model->facDim);
  }
  return score;
}


float** computeTestScores(Data *data) {
  int u;
  UserSets *userSet;
  float **testScores = (float**) malloc(sizeof(float*)*data->nUsers);
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    testScores[u] = setScores(userSet->testSets, userSet->szTestSet, userSet); 
  }
  return testScores;
}


float** computeValScores(Data *data) {
  int u;
  UserSets *userSet;
  float **valScores = (float**) malloc(sizeof(float*)*data->nUsers); 
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    valScores[u] = setScores(userSet->valSets, userSet->szValSet, userSet); 
  }
  return valScores;
}


float **testLabels(Data *data) {
  int u, i; 
  UserSets *userSet;
  float **labels = (float**) malloc(sizeof(float*)*data->nUsers);
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    labels[u] = (float*) malloc(sizeof(float)*userSet->szTestSet);
    for (i = 0; i < userSet->szTestSet; i++) {
      labels[u][i] = userSet->labels[userSet->testSets[i]];
    }
  }
  return labels;
}


float **valLabels(Data *data) {
  int u, i; 
  UserSets *userSet;
  float **labels = (float**) malloc(sizeof(float*)*data->nUsers);
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    labels[u] = (float*) malloc(sizeof(float)*userSet->szValSet);
    for (i = 0; i < userSet->szValSet; i++) {
      labels[u][i] = userSet->labels[userSet->valSets[i]];
    }
  }
  return labels;
}


float computeCorr(float **labels, float **predScores, int nUsers, 
    int setsPerUser) {

  int u, i, j;
  float corr;
  float *x, *y;
  int signLoss = 0;

  x = (float*)malloc(sizeof(float)*nUsers*setsPerUser);
  y = (float*)malloc(sizeof(float)*nUsers*setsPerUser);
  j = 0;

  for (u = 0; u < nUsers; u++) {
    for (i = 0; i < setsPerUser; i++) {
      x[j] = labels[u][i];
      y[j] = predScores[u][i];
      j++;
    }
  }

  corr = pearsonCorr(x, y, nUsers*setsPerUser);

  //writeFloatVector(x, nUsers*setsPerUser, "x.txt");
  //writeFloatVector(y, nUsers*setsPerUser, "y.txt");

  for (i = 0; i < nUsers*setsPerUser; i++) {
    if (x[i]*y[i] < 0) {
      signLoss++;
    } 
  }
  printf("\nsignLoss = %d", signLoss);

  free(x);
  free(y);

  return corr;
}


float validationErr(Model *model, Data *data) {
  float **valScores = NULL;
  float **labels = NULL;
  float corr = 0;
  int u;

  valScores = computeValScores(data);
  labels = valLabels(data);

  corr = computeCorr(labels, valScores, data->nUsers, data->userSets[0]->szValSet);
  
  //free up space
  for (u = 0; u < data->nUsers; u++) {
    free(valScores[u]);
    free(labels[u]);
  }
  free(valScores);
  free(labels);

  return corr;
}


float testErr(Model *model, Data *data) {
  float **testScores = NULL;
  float **labels = NULL;
  float corr = 0;
  int u;

  testScores = computeTestScores(data);
  labels = testLabels(data);
  
  corr = computeCorr(labels, testScores, data->nUsers, data->userSets[0]->szTestSet);
  
  //free up space
  for (u = 0; u < data->nUsers; u++) {
    free(testScores[u]);
    free(labels[u]);
  }
  free(testScores);
  free(labels);
  
  return corr;
}


void trainModel(Model *model, Data *data, Params *params, float **sim) {
  int iter, u, i, j, k;
  int numItems, item, itemInd;
  float uTv, diff, Wui;
  UserSets *userSet;

  int *usersAx, *itemsAx;
  
  usersAx = (int *) malloc(sizeof(int)*data->nUsers);
  memset(usersAx, 0, sizeof(int)*data->nUsers);
  itemsAx = (int*) malloc(sizeof(int)*data->nItems);
  memset(itemsAx, 0, sizeof(int)*data->nItems);

  printf("\nBaseline Val Err: %f", validationErr(model, data));
  printf("\nBaseline Test Err: %f", testErr(model, data));

  for (iter = 0; iter < params->maxIter; iter++) {
    //go over users
    //TODO: experiment with ordering i.e. item first then users
    for (u = 0; u < data->nUsers; u++) {
      
      userSet = data->userSets[u];

      //for each user select an item
      itemInd = rand() % userSet->nUserItems;
      i = userSet->itemWtSets[itemInd]->item; //item
      //ignore if item not in user train
      if (userSet->testValItems[i]) {
        continue;
      }
      
      usersAx[u] += 1;
      itemsAx[i] += 1;

      Wui = userSet->itemWtSets[itemInd]->wt; //user score on item

      //update model for u and i
      uTv = dotProd(model->uFac[u], model->iFac[i], model->facDim);
      
      //update user and item latent factor
      diff = Wui - uTv;
      for (k = 0; k < model->facDim; k++) {
        model->uFac[u][k] += model->learnRate*(diff*model->iFac[i][k] - model->regU*model->uFac[u][k]);
        model->iFac[i][k] += model->learnRate*(diff*model->uFac[u][k] - model->regI*model->iFac[i][k]);
      }

    }

    if (params->useSim) {
      //update sim
      updateSim(sim, model);
      
      //update W if using sim: decide here or above in loop
      UserSets_updWt(userSet, sim);
    }

    //objective check
    if (iter%100 == 0) {
      computeObjective(data, model);
      printf("\nValidation Err: %f", validationErr(model, data));
    }
  }

  printf("\nTestErr: %f", testErr(model, data));

  //writeIntVector(usersAx, data->nUsers, "usersAx.txt");
  //writeIntVector(itemsAx, data->nItems, "itemsAx.txt");
  
  free(usersAx);
  free(itemsAx);
}


void model(Data *data, Params *params) {
 
  Model *model = NULL;
  float **sim  = NULL;

  int i, j, k;

  //init random with seed
  srand(params->seed);

  //allocate storage for model
  model = (Model *) malloc(sizeof(Model));
  Model_init(model, params->nUsers, params->nItems, params->facDim, 
      params->regU, params->regI, params->learnRate);

  //allocate space for sim
  if (params->useSim) {
    sim = (float**) malloc(sizeof(float*)*data->nItems);
    for (i = 0; i < data->nItems; i++) {
      sim[i] = (float*) malloc(sizeof(float)*data->nItems);
      memset(sim[i], 0, sizeof(float)*data->nItems);
    }
  }

  //initialize weights based on user sets
  for (i = 0; i < data->nUsers; i++) {
    UserSets_initWt(data->userSets[i]);
  }
  
  //train model 
  trainModel(model, data, params, sim);

  //test model


  //free up allocated space
  if (params->useSim) {
    for (i = 0; i < data->nItems; i++) {
      free(sim[i]);
    }
    free(sim);
  }
  Model_free(model);
}




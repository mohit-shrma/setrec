#include "modelRand.h"

  
float ModelRand_validationErr(void *self, Data *data, float **sim) {
  
  int u, i, j, k, item;
  ModelRand *model       = self;
  int nValSets           = 0;
  float *valLabels       = NULL;
  float *valModelScores  = NULL;
  float rmse             = 0;
  ItemWtSets *itemWtSets = NULL;
  UserSets *userSet      = NULL;
  int rnd;
  float posPc, negPc;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nValSets += userSet->szValSet;
  }
   
  valLabels = (float*) malloc(sizeof(float)*nValSets);
  valModelScores = (float*) malloc(sizeof(float)*nValSets);
  memset(valLabels, 0, sizeof(float)*nValSets);
  memset(valModelScores, 0, sizeof(float)*nValSets);

  posPc = ((float)model->posCount)/((float)(model->posCount + model->negCount));
  negPc = 1.0 - posPc;

  j = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->szValSet; i++) {
      valLabels[j] = userSet->labels[userSet->valSets[i]];
      valModelScores[j] = (float) generateGaussianNoise(model->avgLabel, 1.0);
      /*
      rnd = rand()%100;
      if (rnd <= posPc*100) {
        valModelScores[j] = 1;
      } else {
        valModelScores[j] = -1;
      }
      */
      
      if (valModelScores[j] <= -1) {
        valModelScores[j] = -1;
      } else if (valModelScores[j] >= 1) {
        valModelScores[j] = 1;
      }
      
      j++;
    }
  }
  
  for (j = 0; j < nValSets; j++) {
    rmse += (valLabels[j] - valModelScores[j]) * (valLabels[j] - valModelScores[j]);
  }
  rmse = sqrt(rmse/nValSets);
  
  //printf("\n%s validation rmse = %f", model->_(description), rmse);

  free(valLabels);
  free(valModelScores);
  return rmse;
}


float ModelRand_testErr(void *self, Data *data, float **sim) {
    
  int u, i, j, k, item;
  ModelRand *model       = self;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float rmse             = 0;
  ItemWtSets *itemWtSets = NULL;
  UserSets *userSet      = NULL;
  int rnd;
  float posPc, negPc;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nTestSets += userSet->szTestSet;
  }
   
  testLabels = (float*) malloc(sizeof(float)*nTestSets);
  testModelScores = (float*) malloc(sizeof(float)*nTestSets);
  memset(testLabels, 0, sizeof(float)*nTestSets);
  memset(testModelScores, 0, sizeof(float)*nTestSets);

  posPc = ((float)model->posCount)/((float)(model->posCount + model->negCount));
  negPc = 1.0 - posPc;

  j = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->szTestSet; i++) {
      testLabels[j] = userSet->labels[userSet->testSets[i]];
      testModelScores[j] = (float) generateGaussianNoise(model->avgLabel, 1.0);
      /*rnd = rand()%100;
      if (rnd <= posPc*100) {
        testModelScores[j] = 1;
      } else {
        testModelScores[j] = -1;
      }*/
      
      if (testModelScores[j] <= -1) {
        testModelScores[j] = -1;
      } else if (testModelScores[j] >= 1) {
        testModelScores[j] = 1;
      }
      
      j++;
    }
  }
  

  for (j = 0; j < nTestSets; j++) {
    rmse += (testLabels[j] - testModelScores[j]) * (testLabels[j] - testModelScores[j]);
  }
  rmse = sqrt(rmse/nTestSets);
  
  //printf("\n%s test rmse = %f", model->_(description), rmse);


  free(testLabels);
  free(testModelScores);
  return rmse;

}


void ModelRand_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {
  
  ModelRand *model = self;
  UserSets *userSet = NULL;
  int u, i, j, s;
  int nSets = 0;
  
  model->posCount = 0;
  model->negCount = 0;
  model->avgLabel = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      model->avgLabel += userSet->labels[s];
      if (userSet->labels[s] > 0) {
        model->posCount++;
      } else {
        model->negCount++;
      }
    }
    nSets += userSet->numSets;
  }
  model->avgLabel = model->avgLabel/nSets;

  /*
  printf("\navgLabel: %f posCount: %d negCount: %d", model->avgLabel, 
      model->posCount, model->negCount);
  */

  //compute validation error
  valTest[0] = model->_(validationErr) (model, data, sim);

  //compute test error
  valTest[1] = model->_(testErr) (model, data, sim);
}


Model ModelRandProto = {
  .train = ModelRand_train,
  .validationErr = ModelRand_validationErr,
  .testErr = ModelRand_testErr
};


void modelRand(Data *data, Params *params, float *valTest) {
  ModelRand *modelRand = NEW(ModelRand, "random prediction model");
  modelRand->_(init)(modelRand, params->nUsers, params->nItems, params->facDim, params->regU, 
    params->regI, params->learnRate);
  modelRand->_(train)(modelRand, data, params, NULL, valTest);
  modelRand->_(free)(modelRand);
}



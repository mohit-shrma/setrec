#include "modelBaseline.h"

void ModelBase_train(void *self, Data *data, Params *params, float **sim, float *valTest) {
  
  int u, i, j, k, setInd;
  UserSets *userSet = NULL;
  ItemWtSets *itemWtSets = NULL;
  int nTestValSets;

  ModelBase *model = self;
  int found = 0;
  
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    //assign score to all items for user
    for (i = 0; i < userSet->nUserItems; i++) {
      itemWtSets = userSet->itemWtSets[i];
      
      //init score
      itemWtSets->wt = 0;
      nTestValSets   = 0;

      //go through the sets where item appears
      for (j = 0; j < itemWtSets->szItemSets; j++) {
        setInd = itemWtSets->itemSets[j];
        
        found = 0;
        //ignore if setInd in test or validation sets
        for (k = 0; k < userSet->szValSet; k++) {
          if (setInd == userSet->valSets[k]) {
            nTestValSets++;
            found = 1;
          }
        }

        if (found) {
          continue;
        }

        found = 0;
        for (k = 0; k < userSet->szTestSet; k++) {
          if (setInd == userSet->testSets[k]) {
            nTestValSets++;
            found = 1;
          }
        }
        if (found) {
          continue;
        }

        itemWtSets->wt += userSet->labels[setInd] / userSet->uSetsSize[setInd];
      }

      if (itemWtSets->szItemSets > nTestValSets) {
        itemWtSets->wt = itemWtSets->wt / (itemWtSets->szItemSets - nTestValSets);
      }

    }
  }

  //compute validation error
  valTest[0] = model->_(validationErr) (model, data, sim); 
  
  //compute test error
  valTest[1] = model->_(testErr) (model, data, sim);

}


float ModelBase_validationErr(void *self, Data *data, float **sim) {
  
  int u, i, j, k, item;
  ModelBase *model       = self;
  int nValSets           = 0;
  float *valLabels       = NULL;
  float *valModelScores  = NULL;
  float rmse             = 0;
  ItemWtSets *itemWtSets = NULL;
  UserSets *userSet      = NULL;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nValSets += userSet->szValSet;
  }
   
  valLabels = (float*) malloc(sizeof(float)*nValSets);
  valModelScores = (float*) malloc(sizeof(float)*nValSets);
  memset(valLabels, 0, sizeof(float)*nValSets);
  memset(valModelScores, 0, sizeof(float)*nValSets);

  j = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->szValSet; i++) {
      valLabels[j] = userSet->labels[userSet->valSets[i]];
      valModelScores[j] = 0;
      for (k = 0; k < userSet->uSetsSize[userSet->valSets[i]]; k++) {
        item = userSet->uSets[userSet->valSets[i]][k];
        itemWtSets = UserSets_search(userSet, item);
        valModelScores[j] += itemWtSets->wt;
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


float ModelBase_testErr(void *self, Data *data, float **sim) {
  
  int u, i, j, k, item;
  ModelBase *model       = self;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float rmse             = 0;
  ItemWtSets *itemWtSets = NULL;
  UserSets *userSet      = NULL;
  int *testSetInd = NULL;


  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nTestSets += userSet->szTestSet;
  }
   
  testLabels = (float*) malloc(sizeof(float)*nTestSets);
  testModelScores = (float*) malloc(sizeof(float)*nTestSets);
  memset(testLabels, 0, sizeof(float)*nTestSets);
  memset(testModelScores, 0, sizeof(float)*nTestSets);

  testSetInd = (int *) malloc(sizeof(int)*nTestSets);
  memset(testSetInd, 0, sizeof(int)*nTestSets);

  j = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->szTestSet; i++) {
      testSetInd[j] = userSet->testSets[i];
      testLabels[j] = userSet->labels[userSet->testSets[i]];
      testModelScores[j] = 0;
      for (k = 0; k < userSet->uSetsSize[userSet->testSets[i]]; k++) {
        item = userSet->uSets[userSet->testSets[i]][k];
        itemWtSets = UserSets_search(userSet, item);
        testModelScores[j] += itemWtSets->wt;
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
  free(testSetInd);
  return rmse;
}


Model ModelBaseProto = {
  .train = ModelBase_train,
  .validationErr = ModelBase_validationErr,
  .testErr = ModelBase_testErr
};


void modelBase(Data *data, Params *params, float *valTest) {
 
  ModelBase *modelBase = NEW(ModelBase, "second baseline prediction model");
  modelBase->_(init)(modelBase, params->nUsers, params->nItems, params->facDim, params->regU, 
    params->regI, params->learnRate);
  //train or compuite baselines
  modelBase->_(train)(modelBase, data, params, NULL, valTest);
  modelBase->_(free)(modelBase);
}



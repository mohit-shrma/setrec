#include "model.h"


void Model_init(void *selfRef, int nUsers, int nItems, int facDim, float regU, 
    float regI, float learnRate) {
  
  int i, j;
  Model *self = selfRef;
  self->nUsers    = nUsers;
  self->nItems    = nItems;
  self->facDim    = facDim;
  self->regU      = regU;
  self->regI      = regI;
  self->learnRate = learnRate;

  self->uFac = (float**) malloc(sizeof(float*)*nUsers);
  for (i = 0; i < nUsers; i++) {
    self->uFac[i] = (float*) malloc(sizeof(float)*facDim);
    memset(self->uFac[i], 0, sizeof(float)*facDim);
    for (j = 0; j < facDim; j++) {
      self->uFac[i][j] = (float)rand() / (float)(RAND_MAX);
    }
  }

  self->iFac = (float**) malloc(sizeof(float*)*nItems);
  for (i = 0; i < nItems; i++) {
    self->iFac[i] = (float*) malloc(sizeof(float)*facDim);
    memset(self->iFac[i], 0, sizeof(float)*facDim);
    for (j = 0; j < facDim; j++) {
      self->iFac[i][j] = (float)rand() / (float)(RAND_MAX);
    }
  }

}


void Model_reset(void *selfRef) {
  
  int i, j;
  Model *self = selfRef;

  for (i = 0; i < self->nUsers; i++) {
    memset(self->uFac[i], 0, sizeof(float)*self->facDim);
    for (j = 0; j < self->facDim; j++) {
      self->uFac[i][j] = (float)rand() / (float)(RAND_MAX);
    }
  }

  for (i = 0; i < self->nItems; i++) {
    memset(self->iFac[i], 0, sizeof(float)*self->facDim);
    for (j = 0; j < self->facDim; j++) {
      self->iFac[i][j] = (float)rand() / (float)(RAND_MAX);
    }
  }

}


void Model_describe(void *self) {
  Model *model = self;
  printf("\n%s", model->description);
}


void Model_updateSim(void *self, float **sim) {
  int i, j;
  Model *model = self;
  for (i = 0; i < model->nItems; i++) {
    sim[i][i] = 1.0;
    for (j = i+1; j < model->nItems; j++) {
      sim[i][j] = sim[j][i] = dotProd(model->iFac[i], model->iFac[j], model->facDim);
    }
  }
}


float Model_objective(void *self, Data *data) {
  printf("\nModel specific objective computation.");
  return -1.0;
}


float Model_setScore(void *self, int user, int *set, int setSz, float **sim) {
  printf("\nModel specific set score.");
  return -1;
}


void Model_train(void *self, Data *data, Params *params, float **Sim, float *valTest) {
  printf("\nModel specific training procedure");
}


void Model_free(void *selfRef) {
  int i;
  Model *self = selfRef;

  for (i = 0; i < self->nUsers; i++) {
    free(self->uFac[i]);
  }
  free(self->uFac);

  for (i = 0; i < self->nItems; i++) {
    free(self->iFac[i]);
  }
  free(self->iFac);
  free(self->description);

  free(self);
}


float Model_validationErr(void *self, Data *data, float **sim) {
  int u, i, j;
  
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nValSets          = 0;
  float *valLabels      = NULL;
  float *valModelScores = NULL;
  float rmse            = 0;

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
      valModelScores[j] =  model->setScore(model, u, userSet->uSets[userSet->valSets[i]], 
          userSet->uSetsSize[userSet->valSets[i]], sim);
      valLabels[j] = userSet->labels[userSet->valSets[i]];
      j++;
    }
  }

  for (j = 0; j < nValSets; j++) {
    rmse += (valLabels[j] - valModelScores[j]) * (valLabels[j] - valModelScores[j]);
  }
  rmse = sqrt(rmse/nValSets);
  
  //printf("\n%s validation rmse = %f", model->description, rmse);

  free(valLabels);
  free(valModelScores);
  return rmse;
}


float Model_testErr(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float rmse            = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nTestSets += userSet->szTestSet;
  }

  testLabels = (float*) malloc(sizeof(float)*nTestSets);
  testModelScores = (float*) malloc(sizeof(float)*nTestSets);
  memset(testLabels, 0, sizeof(float)*nTestSets);
  memset(testModelScores, 0, sizeof(float)*nTestSets);

  j = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->szTestSet; i++) {
      testModelScores[j] =  model->setScore(model, u, userSet->uSets[userSet->testSets[i]], 
          userSet->uSetsSize[userSet->testSets[i]], sim);
      testLabels[j] = userSet->labels[userSet->testSets[i]];
      j++;
    }
  }

  for (j = 0; j < nTestSets; j++) {
    rmse += (testLabels[j] - testModelScores[j]) * (testLabels[j] - testModelScores[j]);
  }
  rmse = sqrt(rmse/nTestSets);
  
  //printf("\n%s test rmse = %f", model->description, rmse);

  free(testLabels);
  free(testModelScores);
  return rmse;
}


float Model_trainErr(void *self, Data *data, float **sim) {

  int u, i, j, s, isTestValSet, nTrainSets, setSz;
  Model *model = self;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff;
  
  nTrainSets = 0;
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
      
      diff = (userSet->labels[s] - model->setScore(model, u, set, setSz, sim));
      rmse += diff*diff;
      nTrainSets++;
    }
  }
  
  rmse = sqrt(rmse/nTrainSets);
  
  return rmse;
}



float Model_userFacNorm(void *self, Data *data) {
  Model *model = self;
  return matNorm(model->uFac, model->nUsers, model->facDim);
}


float Model_itemFacNorm(void *self, Data *data) {
  Model *model = self;
  return matNorm(model->iFac, model->nItems, model->facDim);
}


float Model_setSimilarity(void *self, int *set, int setSz, float **sim) {
  
  float setSim = 0.0;
  int i, j, nPairs = 0;
  Model *model = self;

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
        setSim += dotProd(model->iFac[set[i]], model->iFac[set[j]], 
            model->facDim);
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


void Model_writeUserSetSim(void *self, Data *data, char *fName) {
    
  Model *model = self;
  int u, i, s;
  UserSets * userSet = NULL;
  int *set = NULL;
  FILE *fp = NULL;
  float sim = 0.0;

  fp = fopen(fName, "w");
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];    
      sim = model->setSimilarity(self, set, userSet->uSetsSize[s], NULL);
      fprintf(fp, "\n%d %d %d %f", u, s, userSet->uSetsSize[s], sim);
    }
  }
  fclose(fp);
 
}


void *Model_new(size_t size, Model proto, char *description) {
  
  //set up default methods if they are not set up
  if (!proto.init) proto.init                   = Model_init;
  if (!proto.describe) proto.describe           = Model_describe;
  if (!proto.updateSim) proto.updateSim         = Model_updateSim;
  if (!proto.objective) proto.objective         = Model_objective;
  if (!proto.setScore) proto.setScore           = Model_setScore;
  if (!proto.train) proto.train                 = Model_train;
  if (!proto.free) proto.free                   = Model_free;
  if (!proto.validationErr) proto.validationErr = Model_validationErr;
  if (!proto.testErr) proto.testErr             = Model_testErr;
  if (!proto.trainErr) proto.trainErr           = Model_trainErr;
  if (!proto.reset) proto.reset                 = Model_reset;
  if (!proto.userFacNorm) proto.userFacNorm     = Model_userFacNorm;
  if (!proto.itemFacNorm) proto.itemFacNorm     = Model_itemFacNorm;
  if (!proto.setSimilarity) proto.setSimilarity = Model_setSimilarity;
  if (!proto.writeUserSetSim) proto.writeUserSetSim = Model_writeUserSetSim;

  //struct of one size
  Model *model = calloc(1, size);
  //copy from proto to model or point a different pointer to cast it
  *model = proto;

  //copy the description
  model->description = strdup(description);

  return model;
}


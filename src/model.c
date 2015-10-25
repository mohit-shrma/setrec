#include "model.h"
#include <string.h>

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


float Model_objective(void *self, Data *data, float **sim) {
  printf("\nModel specific objective computation.");
  return -1.0;
}


//r_us = 1/|s| u^T sum_{i \in s}<v_i>
float Model_setScore(void *self, int user, int *set, int setSz, float **sim) {
 
  int i, j, k;
  int item;
  float *itemsLatFac = NULL;
  float pref = 0.0;
  Model *model = self;

  itemsLatFac = (float*) malloc(sizeof(float)*model->facDim);
  memset(itemsLatFac, 0, sizeof(float)*model->facDim);

  for (i = 0; i < setSz; i++) {
    item = set[i];
    for (k = 0; k < model->facDim; k++) {
      itemsLatFac[k] += model->iFac[item][k];
    }
  }

  pref = (1.0/setSz)*dotProd(model->uFac[user], itemsLatFac, 
      model->facDim); 

  free(itemsLatFac);
  
  return pref;
}


void Model_train(void *self, Data *data, Params *params, float **Sim, ValTestRMSE *valTest) {
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


float Model_hingeValErr(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model  = self;
  UserSets *userSet     = NULL;
  int nValSets          = 0;
  float *valLabels      = NULL;
  float *valModelScores = NULL;
  float loss       = 0;

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
    if (valLabels[j]*valModelScores[j] < 1) {
      loss += 1.0 - valLabels[j]*valModelScores[j];
    }
  }
  
  //printf("\n%s validation rmse = %f", model->description, rmse);

  free(valLabels);
  free(valModelScores);
  return loss;
}


float Model_validationClassLoss(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nValSets          = 0;
  float *valLabels      = NULL;
  float *valModelScores = NULL;
  float classLoss        = 0;

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
    if (valLabels[j]*valModelScores[j] <= 0) {
      classLoss += 1;
    }
  }
  
  //printf("\n%s validation rmse = %f", model->description, rmse);

  free(valLabels);
  free(valModelScores);
  return classLoss;
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
      /*
      int *set = userSet->uSets[userSet->testSets[i]];
      int setSz = userSet->uSetsSize[userSet->testSets[i]];
      int k;
      printf("\nQ:%f:%f: ", testModelScores[j], testLabels[j]);
      for (k = 0; k < setSz; k++) {
        printf("%d ", set[k]);
      }
      */
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


float Model_hingeTestErr(void *self, Data *data, float **sim) {
    
  int u, i, j;
  Model *model  = self;
  UserSets *userSet     = NULL;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float loss        = 0;

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
      testModelScores[j] =  model->setScore(model, u, 
          userSet->uSets[userSet->testSets[i]], 
          userSet->uSetsSize[userSet->testSets[i]], sim);
      testLabels[j] = userSet->labels[userSet->testSets[i]];
      j++;
    }
  }

  for (j = 0; j < nTestSets; j++) {
    if (testLabels[j]*testModelScores[j] < 1) {
      loss += 1.0 - testLabels[j]*testModelScores[j];
    }
  }
  
  //printf("\n%s test rmse = %f", model->description, rmse);

  free(testLabels);
  free(testModelScores);
  return loss;

}


float Model_testClassLoss(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float classLoss        = 0;

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
    if (testLabels[j]*testModelScores[j] <= 0) {
      classLoss += 1;
    }
  }
  
  //printf("\n%s test rmse = %f", model->description, rmse);

  free(testLabels);
  free(testModelScores);
  return classLoss;
}


float Model_trainClassLoss(void *self, Data *data, float **sim) {

  int u, i, j, s, isTestValSet, setSz;
  Model *model = self;
  UserSets *userSet = NULL;
  int *set = NULL;
  float classLoss = 0, setScore;
  
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
      setScore = model->setScore(model, u, set, setSz, sim); 
      if (setScore*userSet->labels[s] <= 0) {
        classLoss += 1;
      }
    }
  }
  
  return classLoss;
}


float Model_validationClass01Loss(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nValSets          = 0;
  float *valLabels      = NULL;
  float *valModelScores = NULL;
  float classLoss        = 0;

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
      if (valModelScores[j] > 0.5) {
        valModelScores[j] = 1.0;
      } else {
        valModelScores[j] = 0.0;
      }
      valLabels[j] = userSet->labels[userSet->valSets[i]];
      j++;
    }
  }

  for (j = 0; j < nValSets; j++) {
    classLoss += fabs(valModelScores[j] - valLabels[j]);
  }
  
  //printf("\n%s validation rmse = %f", model->description, rmse);

  free(valLabels);
  free(valModelScores);
  return classLoss/nValSets;
}


float Model_testClass01Loss(void *self, Data *data, float **sim) {
  
  int u, i, j;
  Model *model          = self;
  UserSets *userSet     = NULL;
  int nTestSets          = 0;
  float *testLabels      = NULL;
  float *testModelScores = NULL;
  float classLoss        = 0;

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
      if (testModelScores[j] > 0.5) {
        testModelScores[j] = 1.0;
      } else {
        testModelScores[j] = 0.0;
      }
      testLabels[j] = userSet->labels[userSet->testSets[i]];
      j++;
    }
  }

  for (j = 0; j < nTestSets; j++) {
      classLoss += fabs(testModelScores[j] - testLabels[j]);
  }
  
  //printf("\n%s test rmse = %f", model->description, rmse);

  free(testLabels);
  free(testModelScores);
  return classLoss/nTestSets;
}


float Model_trainClass01Loss(void *self, Data *data, float **sim) {

  int u, i, j, s, isTestValSet, setSz;
  Model *model = self;
  UserSets *userSet = NULL;
  int *set = NULL;
  float classLoss = 0, setScore;
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
      setScore = model->setScore(model, u, set, setSz, sim); 
      
      if (setScore > 0.5) {
        setScore = 1.0;
      } else {
        setScore = 0;
      }
      classLoss += fabs(setScore - userSet->labels[s]);
      nSets++;
    }
  }
  
  return classLoss/nSets;
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

  if (0 == nPairs) {
    setSim = 1.0;
  } else {
    setSim = setSim/nPairs;
  }
  
  //printf("\nsetSim = %f, nPairs = %d, setSz = %d", setSim, nPairs, setSz);

  return setSim;

}


void Model_writeUserSetSim(void *self, Data *data, float **sim, char *fName) {
    
  Model *model = self;
  int u, i, s, setSz;
  UserSets * userSet = NULL;
  int *set = NULL;
  FILE *fp = NULL;
  float simSet = 0.0;

  fp = fopen(fName, "w");
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (s = 0; s < userSet->numSets; s++) {
      set = userSet->uSets[s];    
      setSz = userSet->uSetsSize[s];
      simSet = model->setSimilarity(self, set, userSet->uSetsSize[s], sim);
      fprintf(fp, "\n%d %d %d %f", u, s, setSz, simSet);
    }
  }
  fclose(fp);
 
}


float Model_indivTrainSetsErr(void *self, Data *data) {
  
  int u, i, item;
  UserSets *userSet = NULL;
  Model *model = self;
  float rmse = 0, diff; 
  int nItems = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      //NOTE: following assumes that their actual ratings are loaded in
      //itemWtSets->wt
      diff = (userSet->itemWtSets[i]->wt - 
          dotProd(model->uFac[u], model->iFac[item], model->facDim));
      rmse += diff*diff;
      nItems++;
    }
  }
  
  rmse = sqrt(rmse/(nItems*1.0)); 

  return rmse;
}


float Model_indivItemCSRErr(void *self, gk_csr_t *mat, char *opFName) {
  int u, ii, i, jj, j, item;
  float diff = 0, itemRat, estItemRat;
  int nnz = 0;
  Model *model = self;
  FILE *fp = NULL;
  
  if (opFName) {
    fp = fopen(opFName, "w");
  }

  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      itemRat = mat->rowval[ii];
      estItemRat = dotProd(model->uFac[u], model->iFac[item], model->facDim);
      if (fp) {
        fprintf(fp, "%d,%d,%f,%f\n", u, item, itemRat, estItemRat);
      }

      diff += (itemRat - estItemRat)*(itemRat - estItemRat);
      nnz++;
    }
  }
  if (fp) {
    fclose(fp);
  }

  return sqrt(diff/nnz);

}


float Model_hitRate(void *self, gk_csr_t *trainMat, gk_csr_t *testMat) {
  
  int u, i, j, ii, jj;
  int testItem;
 
  int nItems            = trainMat->ncols;
  float nHits           = 0;
  Model *model          = self;
  gk_fkv_t *itemPrefInd = gk_fkvmalloc(nItems, "hitrate on test");
  int *isItemRated      = (int*) malloc(sizeof(int)*nItems);

  for (u = 0; u < trainMat->nrows; u++) {
    
    //get test item
    testItem = testMat->rowind[testMat->rowptr[u]];

    //get items rated by user in train
    memset(isItemRated, 0, sizeof(int)*nItems);
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      if (trainMat->rowval[ii] > 0) {
        isItemRated[trainMat->rowind[ii]] = 1;
      }
    }

    //go over all items and predict rating
    for (i = 0; i < nItems; i++) {
      if (isItemRated[i]) {
        itemPrefInd[i].key = 0;
      } else {
        itemPrefInd[i].key = dotProd(model->uFac[u], model->iFac[i], model->facDim);
      }
      itemPrefInd[i].val = i;
    }

    //put top-N largest in beginning of array
    gk_dfkvkselect(nItems, 10, itemPrefInd);
    
    for (j = 0; j < 10; j++) {
      if (itemPrefInd[j].val == testItem) {
        //hit
        nHits += 1.0;
      }
    }

  }

  gk_free((void **)&itemPrefInd, LTERM);
  free(isItemRated);

  return (nHits/trainMat->nrows);
}


float Model_itemFeatScore(void *self, int u, int item, gk_csr_t *featMat) {
  return 0;
}


float Model_coldHitRateTr(void *self, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *itemFeatMat, int *testItemIds, 
    int nTestItems, int N) {
  
  int u, i, j, k;
  int nUserTestItems, nUserTopNItems, item;
  float recallU, recallSum, recallAvg;
  Model *model                = self;
  int nUsers                  = model->nUsers;
  gk_fkv_t *testItemsRatingKV = gk_fkvmalloc(nTestItems, "testItemsRating");
  bool *isTestUsers           = (bool*) malloc(sizeof(bool)*nUsers);
  int nRelUsers               = 0;
  FILE *fp = NULL;

  //fp = fopen("topItemRat.txt", "w");
  
  //get number of test users and test users
  memset(isTestUsers, false, sizeof(bool)*nUsers);
  nRelUsers = 0;
  for (u = 0; u < model->nUsers; u++) {
    if (trainMat->rowptr[u+1] - trainMat->rowptr[u] == 0) {
      //user has no training items
      continue;
    }
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] > 0) {
        isTestUsers[u] = true;
        nRelUsers++;
        break;
      }
    }
  }
  
  recallSum = 0;
  //compute ratings for test users on test items and keep top-N
  for (u = 0; u < model->nUsers; u++) {
    if (!isTestUsers[u]) {
      continue;
    }
    
    for (i = 0; i < nTestItems; i++) {
      item = testItemIds[i];
      //compute user rating on item
      testItemsRatingKV[i].key = model->itemFeatScore(model, u, item, 
          itemFeatMat);
      testItemsRatingKV[i].val = item;
    }

    //get top-N in the beginning
    gk_dfkvkselect(nTestItems, N, testItemsRatingKV); 
    //sort top-N items in decreasing order
    gk_fkvsortd(N, testItemsRatingKV);
    
    if (fp) {
      for (j = 0; j < N; j++) {
        fprintf(fp, "%zu %f ", testItemsRatingKV[j].val, 
            testItemsRatingKV[j].key);
      }
      fprintf(fp, "\n");
    }

    //compute hitrate for u
    nUserTestItems = 0;
    nUserTopNItems = 0;
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] <= 0) {
        continue;
      }
      item = testMat->rowind[j];
      nUserTestItems++;

      //check if item present in predicted Top-N items
      for (k = 0; k < N; k++) {
        if (testItemsRatingKV[k].val == item) {
          nUserTopNItems++;
        }
      }
    }
   
    //TODO: discuss it with George
    if (N > nUserTestItems) {
      recallU = ((float)nUserTopNItems)/((float)nUserTestItems);
    } else {
      recallU = ((float)nUserTopNItems)/((float)N);
    }
    recallSum += recallU;
  }
 
  //printf("\nrecallSum: %f nRelUsers: %d", recallSum, nRelUsers);

  recallAvg = recallSum/nRelUsers;


  if (fp) {
    fclose(fp);
  }

  free(isTestUsers);
  gk_free((void**)&testItemsRatingKV, LTERM);

  return recallAvg;
}


float Model_coldHitRateTrParBak(void *self, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *itemFeatMat, int *testItemIds, 
    int nTestItems, int N) {
  
  int i, j, k, u;
  int nUserTestItems, nUserTopNItems, item;
  float recallU, recallSum, recallAvg;
  Model *model                = self;
  const int nUsers                  = model->nUsers;
  gk_fkv_t *testItemsRatingKV = NULL;
  bool *isTestUsers           = (bool*) malloc(sizeof(bool)*nUsers);
  int nRelUsers               = 0;
  float *uRecalls = (float*) malloc((sizeof(float)*nUsers));

  //get number of test users and test users
  memset(isTestUsers, false, sizeof(bool)*nUsers);
  nRelUsers = 0;
  for (u = 0; u < model->nUsers; u++) {
    if (trainMat->rowptr[u+1] - trainMat->rowptr[u] == 0) {
      //user has no training items
      continue;
    }
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] > 0) {
        isTestUsers[u] = true;
        nRelUsers++;
        break;
      }
    }
  }
  
  memset(uRecalls, 0, sizeof(float)*nUsers);

  printf("\nnUsers: %d, nRelUsers: %d", nUsers, nRelUsers);

  //compute ratings for test users on test items and keep top-N
  //
  recallU = 0;
#pragma omp parallel default(none) shared(recallSum)
  {
    int t;
  recallSum = 0;
#pragma omp for reduction(+:recallSum) 
for (t = 0; t < nUsers; t++) {
    recallSum += 1.0;
}
}
  printf("\nav recallSum: %f nRelUsers: %d", recallSum, nRelUsers);
/*
for (u = 0; u < nUsers; u++) {
    recallU = 0;
    if (isTestUsers[u]) {
      testItemsRatingKV = gk_fkvmalloc(nTestItems, "testItemsRating");
   
      for (i = 0; i < nTestItems; i++) {
        item = testItemIds[i];
        //compute user rating on item
        testItemsRatingKV[i].key = model->itemFeatScore(model, u, item, 
            itemFeatMat);
        testItemsRatingKV[i].val = item;
      }

      //get top-N in the beginning
      gk_dfkvkselect(nTestItems, N, testItemsRatingKV); 
      //sort top-N items in decreasing order
      gk_fkvsortd(N, testItemsRatingKV);
      
      //compute hitrate for u
      nUserTestItems = 0;
      nUserTopNItems = 0;
      for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
        if (testMat->rowval[j] <= 0.0) {
          continue;
        }
        item = testMat->rowind[j];
        nUserTestItems++;

        //check if item present in predicted Top-N items
        for (k = 0; k < N; k++) {
          if (testItemsRatingKV[k].val == item) {
            nUserTopNItems++;
          }
        }
      }
     
      gk_free((void**)&testItemsRatingKV, LTERM);
      
      //TODO: discuss it with George
      if (N > nUserTestItems && nUserTestItems > 0) {
        recallU = ((float)nUserTopNItems)/((float)nUserTestItems);
      } else {
        recallU = ((float)nUserTopNItems)/N;
      }
      recallU = 10.0;
    
    }
     
    //uRecalls[u] = 1.0;//recallU;
    //recallSum += recallU;
    recallSum += 1.0;//uRecalls[u];
}
*/
  /*
  FILE *fp = fopen("parURecalls.txt", "w");
  for (u = 0; u < nUsers; u++) {
    fprintf(fp, "%f\n", uRecalls[u]);
  }
  fclose(fp);
  */

  
  recallAvg = recallSum/nRelUsers;

  recallSum = 0;
  for (u = 0; u < model->nUsers; u++) {
    recallSum += uRecalls[u];
  }

  printf("\nrecallSum arr: %f", recallSum);

  free(isTestUsers);
  free(uRecalls);
  return recallAvg;
}


float Model_coldHitRateTrPar(void *self, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *itemFeatMat, int *testItemIds, 
    const int nTestItems, const int N) {
  
  int i, j, k, u;
  float recallSum, recallAvg;
  Model *model                = self;
  const int nUsers                  = model->nUsers;
  bool *isTestUsers           = (bool*) malloc(sizeof(bool)*nUsers);
  int nRelUsers               = 0;
  float *uRecalls = (float*) malloc((sizeof(float)*nUsers));

  //get number of test users and test users
  memset(isTestUsers, false, sizeof(bool)*nUsers);
  nRelUsers = 0;
  for (u = 0; u < model->nUsers; u++) {
    if (trainMat->rowptr[u+1] - trainMat->rowptr[u] == 0) {
      //user has no training items
      continue;
    }
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] > 0) {
        isTestUsers[u] = true;
        nRelUsers++;
        break;
      }
    }
  }
  
  memset(uRecalls, 0, sizeof(float)*nUsers);

  printf("\nnUsers: %d, nRelUsers: %d", nUsers, nRelUsers);

  //compute ratings for test users on test items and keep top-N
  //
#pragma omp parallel default(none) shared(recallSum, isTestUsers, testItemIds, model, testMat, itemFeatMat)
{
    int t, w, x, y, nUserTopNItems, nUserTestItems, item;
    float recallU; 
    gk_fkv_t *testItemsRatingKV = NULL;
    recallSum = 0;
#pragma omp for reduction(+:recallSum) 
    for (t = 0; t < nUsers; t++) {
      recallU = 0;
      if (isTestUsers[t]) {
        testItemsRatingKV = gk_fkvmalloc(nTestItems, "testItemsRating");
     
        for (w = 0; w < nTestItems; w++) {
          item = testItemIds[w];
          //compute user rating on item
          testItemsRatingKV[w].key = model->itemFeatScore(model, t, item, 
              itemFeatMat);
          testItemsRatingKV[w].val = item;
        }

        //get top-N in the beginning
        gk_dfkvkselect(nTestItems, N, testItemsRatingKV); 
        //sort top-N items in decreasing order
        gk_fkvsortd(N, testItemsRatingKV);
        
        //compute hitrate for t
        nUserTestItems = 0;
        nUserTopNItems = 0;
        for (x = testMat->rowptr[t]; x < testMat->rowptr[t+1]; x++) {
          if (testMat->rowval[x] <= 0.0) {
            continue;
          }
          item = testMat->rowind[x];
          nUserTestItems++;

          //check if item present in predicted Top-N items
          for (y = 0; y < N; y++) {
            if (testItemsRatingKV[y].val == item) {
              nUserTopNItems++;
            }
          }
        }
       
        gk_free((void**)&testItemsRatingKV, LTERM);
        
        //TODO: discuss it with George
        if (N > nUserTestItems && nUserTestItems > 0) {
          recallU = ((float)nUserTopNItems)/((float)nUserTestItems);
        } else {
          recallU = ((float)nUserTopNItems)/N;
        }
        //recallU = 10.0;
      
      }
      //uRecalls[u] = 1.0;//recallU;
      recallSum += recallU;
      //recallSum += 1.0;//uRecalls[u];
    }
}

  recallAvg = recallSum/nRelUsers;
  printf("\nav recallSum: %f nRelUsers: %d", recallSum, nRelUsers);

  recallSum = 0;
  for (u = 0; u < model->nUsers; u++) {
    recallSum += uRecalls[u];
  }

  printf("\nrecallSum arr: %f", recallSum);

  free(isTestUsers);
  free(uRecalls);
  return recallAvg;
}


/*
float Model_coldHitRateTr(void *self, gk_csr_t *trainMat, 
    gk_csr_t *testMat, gk_csr_t *itemFeatMat, int *testItemIds, 
    int nTestItems, int N) {
  
  int u, i, j, k;
  int nUserTestItems, nUserTopNItems, item;
  float recallU, recallSum, recallAvg;
  Model *model                = self;
  int nUsers                  = model->nUsers;
  gk_fkv_t *testItemsRatingKV = NULL;
  bool *isTestUsers           = (bool*) malloc(sizeof(bool)*nUsers);
  int nRelUsers               = 0;
  float *uRecalls = (float*) malloc((sizeof(float)*nUsers));

  //get number of test users and test users
  memset(isTestUsers, false, sizeof(bool)*nUsers);
  nRelUsers = 0;
  for (u = 0; u < model->nUsers; u++) {
    if (trainMat->rowptr[u+1] - trainMat->rowptr[u] == 0) {
      //user has no training items
      continue;
    }
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] > 0) {
        isTestUsers[u] = true;
        nRelUsers++;
        break;
      }
    }
  }
  
  recallSum = 0;

  memset(uRecalls, 0, sizeof(float)*nUsers);


  //compute ratings for test users on test items and keep top-N
  //
//#pragma omp parallel for private(u, i, j, k, item, testItemsRatingKV, nUserTestItems, nUserTopNItems, recallU) reduction(+:recallSum) 
  for (u = 0; u < model->nUsers; u++) {
    if (!isTestUsers[u]) {
      continue;
    }
   
    testItemsRatingKV = gk_fkvmalloc(nTestItems, "testItemsRating");
 
    for (i = 0; i < nTestItems; i++) {
      item = testItemIds[i];
      //compute user rating on item
      testItemsRatingKV[i].key = model->itemFeatScore(model, u, item, 
          itemFeatMat);
      testItemsRatingKV[i].val = item;
    }

    //get top-N in the beginning
    gk_dfkvkselect(nTestItems, N, testItemsRatingKV); 
    //sort top-N items in decreasing order
    gk_fkvsortd(N, testItemsRatingKV);
    
    //compute hitrate for u
    nUserTestItems = 0;
    nUserTopNItems = 0;
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] <= 0) {
        continue;
      }
      item = testMat->rowind[j];
      nUserTestItems++;

      //check if item present in predicted Top-N items
      for (k = 0; k < N; k++) {
        if (testItemsRatingKV[k].val == item) {
          nUserTopNItems++;
        }
      }
    }
   
    gk_free((void**)&testItemsRatingKV, LTERM);
    
    //TODO: discuss it with George
    if (N > nUserTestItems) {
      recallU = ((float)nUserTopNItems)/((float)nUserTestItems);
    } else {
      recallU = ((float)nUserTopNItems)/((float)N);
    }
    recallSum += recallU;
    uRecalls[u] = recallU;
  }

  FILE *fp = fopen("parURecalls.txt", "w");
  for (u = 0; u < nUsers; u++) {
    fprintf(fp, "%f\n", uRecalls[u]);
  }
  fclose(fp);
  
  printf("\nrecallSum: %f nRelUsers: %d", recallSum, nRelUsers);
  
  recallAvg = recallSum/nRelUsers;

  free(isTestUsers);
  free(uRecalls);
  return recallAvg;
}
*/


float Model_coldHitRate(void *self, UserSets **userSets, 
    gk_csr_t *testMat, gk_csr_t *itemFeatMat, int *testItemIds, 
    int nTestItems, int N) {
  
  int u, i, j, k;
  int nUserTestItems, nUserTopNItems, item;
  float recallU, recallSum, recallAvg;
  Model *model                = self;
  int nUsers                  = model->nUsers;
  gk_fkv_t *testItemsRatingKV = gk_fkvmalloc(nTestItems, "testItemsRating");
  bool *isTestUsers           = (bool*) malloc(sizeof(bool)*nUsers);
  int nRelUsers               = 0;
  FILE *fp = NULL;

  fp = fopen("topItemRat.txt", "w");

  //get number of test users and test users
  memset(isTestUsers, false, sizeof(bool)*nUsers);
  nRelUsers = 0;
  for (u =0; u < model->nUsers; u++) {
    if (userSets[u]->numSets == 0) {
      continue;
    }
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] > 0) {
        isTestUsers[u] = true;
        nRelUsers++;
        break;
      }
    }
  }
  
  recallSum = 0;
  //compute ratings for test users on test items and keep top-N
  for (u = 0; u < model->nUsers; u++) {
    if (!isTestUsers[u]) {
      continue;
    }
    
    for (i = 0; i < nTestItems; i++) {
      item = testItemIds[i];
      //compute user rating on item
      testItemsRatingKV[i].key = model->itemFeatScore(model, u, item, 
          itemFeatMat);
      testItemsRatingKV[i].val = item;
    }

    //get top-N in the beginning
    gk_dfkvkselect(nTestItems, N, testItemsRatingKV); 
    //sort top-N items in decreasing order
    gk_fkvsortd(N, testItemsRatingKV);
    
    if (fp) {
      for (j = 0; j < N; j++) {
        fprintf(fp, "%zu %f ", testItemsRatingKV[j].val, 
            testItemsRatingKV[j].key);
      }
      fprintf(fp, "\n");
    }

    //compute hitrate for u
    nUserTestItems = 0;
    nUserTopNItems = 0;
    for (j = testMat->rowptr[u]; j < testMat->rowptr[u+1]; j++) {
      if (testMat->rowval[j] <= 0) {
        continue;
      }
      item = testMat->rowind[j];
      nUserTestItems++;

      //check if item present in predicted Top-N items
      for (k = 0; k < N; k++) {
        if (testItemsRatingKV[k].val == item) {
          nUserTopNItems++;
        }
      }
    }
   
    //TODO: discuss it with George
    if (N > nUserTestItems) {
      recallU = ((float)nUserTopNItems)/((float)nUserTestItems);
    } else {
      recallU = ((float)nUserTopNItems)/((float)N);
    }
    recallSum += recallU;
  }
  
  recallAvg = recallSum/nRelUsers;

  if (fp) {
    fclose(fp);
  }

  free(isTestUsers);
  gk_free((void**)&testItemsRatingKV, LTERM);

  return recallAvg;
}


float Model_hitRateOrigTopN(void *self, gk_csr_t *trainMat,
    float **origUFac, float **origIFac, int N) {
  
  int u, i, j, ii, jj;
  int nItems            = trainMat->ncols;
  float nHits           = 0;
  Model *model          = self;
  
  if (origUFac == NULL || origIFac == NULL) {
    return 0;
  }
  
  gk_fkv_t *origItemPrefInd = gk_fkvmalloc(nItems, "preferences of the original");
  gk_fkv_t *modelItemPrefInd = gk_fkvmalloc(nItems, "preferences of the model");
  int *isItemRated      = (int*) malloc(sizeof(int)*nItems);

  for (u = 0; u < trainMat->nrows; u++) {
    
    //get items rated by user in train
    memset(isItemRated, 0, sizeof(int)*nItems);
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      if (trainMat->rowval[ii] > 0) {
        isItemRated[trainMat->rowind[ii]] = 1;
      }
    }

    //go over all items and predict rating
    for (i = 0; i < nItems; i++) {
      if (isItemRated[i]) {
        origItemPrefInd[i].key = 0;
        modelItemPrefInd[i].key = 0;
      } else {
        modelItemPrefInd[i].key = dotProd(model->uFac[u], model->iFac[i], model->facDim);
        origItemPrefInd[i].key = dotProd(origUFac[u], origIFac[i], model->facDim); 
      }
      modelItemPrefInd[i].val = i;
      origItemPrefInd[i].val = i;
    }

    //put top-N largest in beginning of array
    gk_dfkvkselect(nItems, N, origItemPrefInd);
    gk_dfkvkselect(nItems, N, modelItemPrefInd);

    for (i = 0; i < N; i++) {
      //check if origItemPrefInd[i].val is in modelItemPrefInd[0:N-1]
      for (j = 0; j < N; j++) {
        if (origItemPrefInd[i].val == modelItemPrefInd[j].val) {
          nHits += 1.0;
        }
      }
    }

  }

  gk_free((void **)&origItemPrefInd, LTERM);
  gk_free((void **)&modelItemPrefInd, LTERM);
  free(isItemRated);

  //divide by no. of users to get avg hit per user
  return (nHits/trainMat->nrows); 
}


double Model_spearmanRankCorrN(void *self, gk_csr_t *testMat, int N) {
  
  int u, i, j, ii, item;
  Model *model          = self;
  double *actualRat     = (double*) malloc(sizeof(double)*N);
  double *predRat       = (double*) malloc(sizeof(double)*N);
  double *spearmanWork  = (double*) malloc(sizeof(double)*2*N);
  double uSpearMan, avgSpearMan, gr80Spearman, less0Spearman;
  int gr80 = 0, less0 = 0, nanCount = 0;

  uSpearMan     = 0;
  avgSpearMan   = 0;
  gr80Spearman  = 0;
  less0Spearman = 0;
 
  FILE *fp = NULL;
  //FILE *fp = fopen("testPred.txt", "w");

  for (u = 0; u < testMat->nrows; u++) {
    memset(actualRat, 0, sizeof(double)*N);
    memset(predRat, 0, sizeof(double)*N);
    memset(spearmanWork, 0, sizeof(double)*2*N);
    j = 0;
    //get actual and predicted rating for test items
    for (ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1] && j < N; 
                                  ii++, j++) {
      item         = testMat->rowind[ii];
      actualRat[j] = testMat->rowval[ii];
      predRat[j]   = dotProd(model->uFac[u], model->iFac[item],
                              model->facDim);
      if (fp) {
        fprintf(fp, "%f ", predRat[j]);
      }
    }

    if (fp) {
      fprintf(fp, "\n");
    }
    uSpearMan = gsl_stats_spearman(actualRat, 1, predRat, 1, N, spearmanWork);
   
    //nan check
    if (uSpearMan != uSpearMan) {
      nanCount++;    
      continue;
    }

    if (uSpearMan > 0) {
      gr80++;
      gr80Spearman += uSpearMan;
    } else{
      less0++;
      less0Spearman += uSpearMan;
    }
    avgSpearMan += uSpearMan;
 
  }
  
  if (fp != NULL) {
    fclose(fp);
  }

  printf("\nnanCount: %d gr80: %d less0:%d gr80+less0:%d", nanCount, 
      gr80, less0, gr80+less0);
  
  avgSpearMan = avgSpearMan/(gr80+less0);
  gr80Spearman = gr80Spearman/gr80;
  less0Spearman = less0Spearman/less0;
 
  free(actualRat);
  free(predRat);
  free(spearmanWork);
  
  return avgSpearMan; 
}


int Model_isTerminateModel(void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data) {
  
  int ret               = 0;
  Model *model          = self;
  Model *bestModel      = bestM;
  if (data->valMat) {
    valTest->valItemsRMSE = model->indivItemCSRErr(model, data->valMat, NULL);
    valTest->valSpearman  = -1;//model->spearmanRankCorrN(model, data->valMat, 10);
  } else {
    valTest->valItemsRMSE = -1;
    valTest->valSpearman = -1;
  }
  valTest->setObj = model->objective(model, data, NULL);
 
  printf("\nIter: %d Obj: %.10e valRMSE: %f", iter, 
      valTest->setObj, valTest->valItemsRMSE);
  //printf("\nIter: %d Obj: %.10e valSpearman: %f valRMSE: %f", iter, 
  //    valTest->setObj, valTest->valSpearman, valTest->valItemsRMSE);
  
  if (iter > 0) {
    if (valTest->setObj < *bestObj) {
      model->copy(model, bestModel);
      *bestObj = valTest->setObj;
      *bestIter = iter;
    }

    if (iter - *bestIter >= 500) {
      //after 500 more iterations cant beat best objective achieved
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e currIter:%d currObj: %.10e",
          *bestIter, *bestObj, iter, valTest->setObj);
      ret = 1;
    }

    if (fabs(*prevObj - valTest->setObj) < EPS) {
      //exit train procedure
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
          *prevObj, valTest->setObj);
      ret = 1;
    }
  }
  
  if (iter == 0) {
    *bestObj = valTest->setObj;
    *bestIter = iter;
  }
  
  *prevObj = valTest->setObj;
  return ret;
}


int Model_isTerminateColdModel(void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data, 
    int *testItemIds, int nTestItems) {
  
  int ret               = 0;
  Model *model          = self;
  Model *bestModel      = bestM;
  if (data->valMat) {
    //valTest->valItemsRMSE = model->indivItemCSRErr(model, data->valMat, NULL);
    valTest->valSpearman  = model->coldHitRate(model, data->userSets, 
        data->valMat, data->itemFeatMat, testItemIds, nTestItems, 10);
  } else {
    valTest->valItemsRMSE = -1;
    valTest->valSpearman = -1;
  }
  valTest->setObj = model->objective(model, data, NULL);
 
  printf("\nIter: %d Obj: %.10e valSpearman: %f valRMSE: %f", iter, 
      valTest->setObj, valTest->valSpearman, valTest->valItemsRMSE);
  
  if (iter > 0) {
    if (valTest->setObj < *bestObj) {
      model->copy(model, bestModel);
      *bestObj = valTest->setObj;
      *bestIter = iter;
    }

    if (iter - *bestIter >= 500) {
      //after 500 more iterations cant beat best objective achieved
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e currIter:%d currObj: %.10e",
          *bestIter, *bestObj, iter, valTest->setObj);
      ret = 1;
    }

    if (fabs(*prevObj - valTest->setObj) < EPS) {
      //exit train procedure
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
          *prevObj, valTest->setObj);
      ret = 1;
    }
  }
  
  if (iter == 0) {
    *bestObj = valTest->setObj;
    *bestIter = iter;
  }
  
  *prevObj = valTest->setObj;
  return ret;
}


int Model_isTerminateClassModel(void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data) {
  
  int ret               = 0;
  Model *model          = self;
  Model *bestModel      = bestM;
  valTest->valSetRMSE   = model->validationClassLoss(model, data, NULL);
  valTest->setObj       = model->objective(model, data, NULL);
  valTest->valSpearman  = model->spearmanRankCorrN(model, data->valMat, 10);
 
  printf("\nIter: %d Obj: %.10e valSpearman: %f valClassLoss: %f", iter, 
      valTest->setObj, valTest->valSpearman, valTest->valSetRMSE);
  
  if (iter > 0) {
    if (valTest->setObj < *bestObj) {
      model->copy(model, bestModel);
      *bestObj = valTest->setObj;
      *bestIter = iter;
    }

    if (iter - *bestIter >= 500) {
      //after 500 more iterations cant beat best objective achieved
      printf("\nNOT CONVERGED: bestIter:%d bestObj: %.10e currIter:%d currObj: %.10e",
          *bestIter, *bestObj, iter, valTest->setObj);
      ret = 1;
    }

    if (fabs(*prevObj - valTest->setObj) < EPS) {
      //exit train procedure
      printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
          *prevObj, valTest->setObj);
      ret = 1;
    }
  }
  
  fflush(stdout);

  if (iter == 0) {
    *bestObj = valTest->setObj;
    *bestIter = iter;
  }
  
  *prevObj = valTest->setObj;
  return ret;
}


float Model_cmpLoadedFac(void *self, Data *data) {
  int u, i, j;
  Model *model = self;
  float uSumSqFacDiff = 0.0;
  float iSumSqFacDiff = 0.0;
  
  for (u = 0; u < data->nUsers; u++) {
    for (j = 0; j < model->facDim; j++) {
      uSumSqFacDiff += pow(model->uFac[u][j] - data->uFac[u][j], 2);    
    }
  }
 
  for (i = 0; i < data->nItems; i++) {
    for (j = 0; j < model->facDim; j++) {
      iSumSqFacDiff += pow(model->iFac[i][j] - data->iFac[i][j], 2);
    }
  }

  printf("\nuFacNormDiff = %f, iFacNormDiff = %f", sqrt(uSumSqFacDiff), 
      sqrt(iSumSqFacDiff));
  
  return (uSumSqFacDiff + iSumSqFacDiff) / 2.0;
}


void Model_copy(void *self, void *dest) {
  
  int i, j;

  Model *frmModel = self;
  Model *toModel = dest;

  //copy to model
  toModel->nUsers    = frmModel->nUsers;
  toModel->nItems    = frmModel->nItems;
  toModel->regU      = frmModel->regU;
  toModel->regI      = frmModel->regI;
  toModel->facDim    = frmModel->facDim;
  toModel->learnRate = frmModel->learnRate;
  
  //TODO: copy model description

  for (i = 0; i < frmModel->nUsers; i++) {
    memcpy(toModel->uFac[i], frmModel->uFac[i], sizeof(float)*frmModel->facDim);
  }
  
  for (i = 0; i < frmModel->nItems; i++) {
    memcpy(toModel->iFac[i], frmModel->iFac[i], sizeof(float)*frmModel->facDim);
  }

}


void *Model_new(size_t size, Model proto, const char *description) {
  
  //set up default methods if they are not set up
  if (!proto.init) proto.init                               = Model_init;
  if (!proto.describe) proto.describe                       = Model_describe;
  if (!proto.updateSim) proto.updateSim                     = Model_updateSim;
  if (!proto.objective) proto.objective                     = Model_objective;
  if (!proto.setScore) proto.setScore                       = Model_setScore;
  if (!proto.train) proto.train                             = Model_train;
  if (!proto.free) proto.free                               = Model_free;
  if (!proto.validationErr) proto.validationErr             = Model_validationErr;
  if (!proto.validationClassLoss) proto.validationClassLoss = Model_validationClassLoss;
  if (!proto.validationClass01Loss) proto.validationClass01Loss = Model_validationClass01Loss;
  if (!proto.testErr) proto.testErr                         = Model_testErr;
  if (!proto.hingeTestErr) proto.hingeTestErr               = Model_hingeTestErr;
  if (!proto.hingeValidationErr) proto.hingeValidationErr   = Model_hingeValErr;
  if (!proto.testClassLoss) proto.testClassLoss             = Model_testClassLoss;
  if (!proto.testClass01Loss) proto.testClass01Loss         = Model_testClass01Loss;
  if (!proto.trainClassLoss) proto.trainClassLoss           = Model_trainClassLoss;
  if (!proto.trainClass01Loss) proto.trainClass01Loss       = Model_trainClass01Loss;
  if (!proto.trainErr) proto.trainErr                       = Model_trainErr;
  if (!proto.reset) proto.reset                             = Model_reset;
  if (!proto.userFacNorm) proto.userFacNorm                 = Model_userFacNorm;
  if (!proto.itemFacNorm) proto.itemFacNorm                 = Model_itemFacNorm;
  if (!proto.setSimilarity) proto.setSimilarity             = Model_setSimilarity;
  if (!proto.writeUserSetSim) proto.writeUserSetSim         = Model_writeUserSetSim;
  if (!proto.hitRate) proto.hitRate                         = Model_hitRate;
  if (!proto.hitRateOrigTopN) proto.hitRateOrigTopN         = Model_hitRateOrigTopN;
  if (!proto.spearmanRankCorrN) proto.spearmanRankCorrN     = Model_spearmanRankCorrN; 
  if (!proto.indivTrainSetsErr) proto.indivTrainSetsErr     = Model_indivTrainSetsErr;
  if (!proto.cmpLoadedFac) proto.cmpLoadedFac               = Model_cmpLoadedFac;
  if (!proto.indivItemCSRErr) proto.indivItemCSRErr         = Model_indivItemCSRErr;
  if (!proto.copy) proto.copy                               = Model_copy;
  if (!proto.isTerminateModel) proto.isTerminateModel       = Model_isTerminateModel;
  if (!proto.isTerminateColdModel) proto.isTerminateColdModel       = Model_isTerminateColdModel;
  if (!proto.isTerminateClassModel) proto.isTerminateClassModel = Model_isTerminateClassModel;
  if (!proto.coldHitRate) proto.coldHitRate                  = Model_coldHitRate;
  if (!proto.coldHitRateTr) proto.coldHitRateTr              = Model_coldHitRateTr;
  if (!proto.coldHitRateTrPar) proto.coldHitRateTrPar              = Model_coldHitRateTrPar;
  if (!proto.itemFeatScore) proto.itemFeatScore              = Model_itemFeatScore; 
  //struct of one size
  Model *model = calloc(1, size);
  //copy from proto to model or point a different pointer to cast it
  *model = proto;

  //copy the description
  model->description = strdup(description);

  return model;
}



#include "datastruct.h"


void UserSets_init(UserSets * const self, int user, int numSets, int nItems,
  int nUserItems) {
  
  int i;
  
  self->userId = user;
  self->numSets = numSets;
  self->nUserItems = nUserItems;

  self->uSets = (int **) malloc(sizeof(int*)*numSets);
  memset(self->uSets, 0, sizeof(int*)*numSets);

  self->labels = (float*) malloc(sizeof(float)*numSets);
  memset(self->labels, 0, sizeof(float)*numSets);
  
  self->uSetsSize = (int*) malloc(sizeof(int) * numSets);
  memset(self->uSetsSize, 0, sizeof(int)*numSets);
  
  //map of item to sets, NOTE: this exists for all items
  self->itemSets = (int **) malloc(sizeof(int*)*nItems);
  memset(self->itemSets, 0, sizeof(int*)*nItems);
  self->itemSetsSize = (int *) malloc(sizeof(int)*nItems);
  memset(self->itemSetsSize, 0, sizeof(int)*nItems);
  
  self->items = (int*) malloc(sizeof(int)*nUserItems);
  memset(self->items, 0, sizeof(int)*nUserItems);
  
  self->itemWts = (float*) malloc(sizeof(float)*nUserItems);
  memset(self->itemWts, 0, sizeof(float)*nUserItems);
}


void UserSets_initWt(UserSets *self) {
  
  int i, j, k;
  int item, setInd;
  float score;

  for (i = 0; i < self->nUserItems; i++) {
    item = self->items[i];
    score = 0;
    //go over sets of item
    for (j = 0; j < self->itemSetsSize[item]; j++) {
      //jth set in which item belongs
      setInd = self->itemSets[item][j];
      score += self->labels[setInd]/self->uSetsSize[setInd];
    }
    score = score/self->itemSetsSize[item];
    self->itemWts[i] = score;
  }

}


void UserSets_updWt(UserSets *self, float **sim) {
  
  int i, j, k;
  int item, setInd;
  float score, simSet;

  for (i = 0; i < self->nUserItems; i++) {
    item = self->items[i];
    score = 0;
    //go over sets of item
    for (j = 0; j < self->itemSetsSize[item]; j++) {
      //jth set in which item belongs
      setInd = self->itemSets[item][j];
      simSet = 0;
      for (k = 0; k < self->uSetsSize[setInd]; k++) {
        //aggregate sim with all items in set except the item itself
        if (self->uSets[setInd][k] != item) {
          simSet += sim[item][self->uSets[setInd][k]]; 
        }
      }
      score += (self->labels[setInd]*simSet)/self->uSetsSize[setInd];
    }
    score = score/self->itemSetsSize[item];
    self->itemWts[i] = score;
  }

}


void UserSets_free(UserSets * const self) {
  int i, item;
 
  //free arrays of arrays
  for (i = 0; i < self->numSets; i++) {
    free(self->uSets[i]);
  }
  free(self->uSets);

  for (i = 0; i < self->nUserItems; i++) {
    item = self->items[i];
    free(self->itemSets[item]);
  }
  free(self->itemSets);
  
  free(self->uSetsSize);
  free(self->labels);
  free(self->items);
  free(self->itemSetsSize);
  free(self->itemWts);
  free(self); 
}


void Data_init(Data *self, int nUsers, int nItems) {
  
  int i;
  self->nUsers = nUsers;
  self->nItems = nItems;

  self->userSets = (UserSets**) malloc(sizeof(UserSets*)*nUsers);
  for (i = 0; i < nUsers; i++) {
    self->userSets[i] = (UserSets*) malloc(sizeof(UserSets));
  }
  
}


void Data_free(Data *self) {
  int i;

  for (i = 0; i < self->nUsers; i++) {
    UserSets_free(self->userSets[i]);
  }
  
  free(self);
}


void Model_init(Model *self, int nUsers, int nItems, int facDim, float regU, 
    float regI, float learnRate) {
  
  int i, j;
  
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


void Model_free(Model *self) {
  int i;
  
  for (i = 0; i < self->nUsers; i++) {
    free(self->uFac[i]);
  }
  free(self->uFac);

  for (i = 0; i < self->nItems; i++) {
    free(self->iFac[i]);
  }
  free(self->iFac);


  free(self);
}


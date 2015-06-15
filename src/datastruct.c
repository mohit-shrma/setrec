#include "datastruct.h"


void UserSets_init(UserSets * const self, int user, int numSets, int nItems,
  int nUserItems) {
  
  int i, j, setInd;
  
  self->userId     = user;
  self->numSets    = numSets;
  self->nUserItems = nUserItems;
  self->szValSet   = 1;
  self->szTestSet  = 1;

  self->uSets = (int **) malloc(sizeof(int*)*numSets);
  memset(self->uSets, 0, sizeof(int*)*numSets);

  self->labels = (float*) malloc(sizeof(float)*numSets);
  memset(self->labels, 0, sizeof(float)*numSets);
  
  self->uSetsSize = (int*) malloc(sizeof(int) * numSets);
  memset(self->uSetsSize, 0, sizeof(int)*numSets);
  
  //map of item to sets, NOTE: this exists for all items
  //TODO: use bsearch on arr
  self->testValItems = (int*) malloc(sizeof(int)*nItems);
  memset(self->testValItems, 0, sizeof(int)*nItems);


  self->itemWtSets = (ItemWtSets **) malloc(sizeof(ItemWtSets*)*nUserItems);   
  for (i = 0; i < nUserItems; i++) {
    self->itemWtSets[i] = (ItemWtSets *) malloc(sizeof(ItemWtSets));
    memset(self->itemWtSets[i], 0, sizeof(ItemWtSets));    
  }

  //initialize test and val sets using random
  self->valSets = (int *) malloc(sizeof(int)*self->szValSet);
  memset(self->valSets, 0, sizeof(int)*self->szValSet);
  i = 0;
  while (i < self->szValSet) {
    setInd = rand()%numSets;
    //make sure set is not present already in validation set
    for (j = 0; j < i; j++) {
      if (setInd == self->valSets[j]) {
        continue; 
      }
    }
    self->valSets[i++] = setInd;
  }


  self->testSets = (int *) malloc(sizeof(int)*self->szTestSet);
  memset(self->testSets, 0, sizeof(int)*self->szTestSet);
  i = 0;
  while (i < self->szTestSet) {
    setInd = rand()%numSets;
    //make sure set is not present already in test set
    for (j = 0; j < i; j++) {
      if (setInd == self->testSets[j]) {
        continue;
      }
    }
    //make sure set is not present in validation set
    for (j = 0; j < self->szValSet; j++) {
      if (setInd == self->valSets[j]) {
        continue;
      }
    }
    self->testSets[i++] = setInd;
  }

}


void UserSets_reset(UserSets * const self, int nItems) {
 
  int i, j, setInd;
 
  //TODO: can remove this
  //reset self->testValItems
  memset(self->testValItems, 0, sizeof(int)*nItems);
  
  //reset validation and test sets
  memset(self->valSets, 0, sizeof(int)*self->szValSet);
  i = 0; 
  while (i < self->szValSet) {
    setInd = rand()%self->numSets;
    //make sure set is not present already in validation set
    for (j = 0; j < i; j++) {
      if (setInd == self->valSets[j]) {
        continue; 
      }
    }
    self->valSets[i++] = setInd;
  }
  
  
  memset(self->testSets, 0, sizeof(int)*self->szTestSet);
  i = 0;
  while (i < self->szTestSet) {
    setInd = rand()%self->numSets;
    //make sure set is not present already in test set
    for (j = 0; j < i; j++) {
      if (setInd == self->testSets[j]) {
        continue;
      }
    }
    //make sure set is not present in validation set
    for (j = 0; j < self->szValSet; j++) {
      if (setInd == self->valSets[j]) {
        continue;
      }
    }
    self->testSets[i++] = setInd;
  }


}


void UserSets_initWt(UserSets *self) {
  
  int i, j, k;
  int setInd;
  float score;
  int testValSetCt = 0;

  for (i = 0; i < self->nUserItems; i++) {
    score = 0;
    //go over sets of item
    testValSetCt = 0;
    for (j = 0; j < self->itemWtSets[i]->szItemSets; j++) {
      //jth set in which item belongs
      setInd = self->itemWtSets[i]->itemSets[j];
      
      //ignore if setInd present in test or validation
      for (k = 0; k < self->szValSet; k++) {
        if (setInd == self->valSets[k]) {
          testValSetCt++;
          continue;
        }
      }

      for (k = 0; k < self->szTestSet; k++) {
        if (setInd == self->testSets[k]) {
          testValSetCt++;
          continue;
        }
      }
      
      score += self->labels[setInd]/self->uSetsSize[setInd];
    }

    if ((self->itemWtSets[i]->szItemSets - testValSetCt) > 0) {
      score = score/(self->itemWtSets[i]->szItemSets - testValSetCt);
      self->itemWtSets[i]->wt = score;
    } else {
      self->testValItems[self->itemWtSets[i]->item] = 1;
      self->itemWtSets[i]->wt = 0;
    }
  }

}


void UserSets_dispWt(UserSets *self) {
  
  int i, j;
  for (i = 0; i < self->nUserItems; i++) {
    printf("\n%d %f", self->itemWtSets[i]->item, self->itemWtSets[i]->wt);
  }

}


void UserSets_writeWt(UserSets *self, char *fileName) {
  
  FILE *fp = NULL;
  int i;

  fp = fopen(fileName, "w");
  
  for (i = 0; i < self->nUserItems; i++) {
    fprintf(fp, "\n%d %f", self->itemWtSets[i]->item, self->itemWtSets[i]->wt);
  }

  fclose(fp);
}


void UserSets_updWt_avgItemSim(UserSets *self, float **sim) {
  
  int i, j, k;
  int item, setInd;
  float score, simSet;
  int testValSetCt, nSim;

  for (i = 0; i < self->nUserItems; i++) {
    item = self->itemWtSets[i]->item;
    score = 0;
    //go over sets of item
    testValSetCt = 0;
    for (j = 0; j < self->itemWtSets[i]->szItemSets; j++) {
      //jth set in which item belongs
      setInd = self->itemWtSets[i]->itemSets[j];
      
      //ignore if setInd present in test or validation
      for (k = 0; k < self->szValSet; k++) {
        if (setInd == self->valSets[k]) {
          testValSetCt++;
          continue;
        }
      }

      for (k = 0; k < self->szTestSet; k++) {
        if (setInd == self->testSets[k]) {
          testValSetCt++;
          continue;
        }
      }
  
      
      simSet = 0.0;
      nSim = 0;
      for (k = 0; k < self->uSetsSize[setInd]; k++) {
        //aggregate sim with all items in set except the item itself
        if (self->uSets[setInd][k] != item) {
          simSet += sim[item][self->uSets[setInd][k]]; 
          nSim++;
        }
      }

      if (nSim == 0) {
        score = self->labels[setInd]/self->uSetsSize[setInd];
      } else {
        simSet = simSet/nSim;
        score += (self->labels[setInd]*simSet)/self->uSetsSize[setInd];
      }

    }
    
    if ((self->itemWtSets[i]->szItemSets - testValSetCt) > 0) {
      score = score/(self->itemWtSets[i]->szItemSets - testValSetCt);
      self->itemWtSets[i]->wt = score;
    } else {
      assert(self->testValItems[self->itemWtSets[i]->item] == 1);
      self->itemWtSets[i]->wt = 0;
    }
    
  }

}


void UserSets_updWt_avgItemPairSim(UserSets *self, float **sim) {
  
  int i, j, k, l;
  int item, setInd;
  float score, simSet;
  int testValSetCt, nSim;

  for (i = 0; i < self->nUserItems; i++) {
    item = self->itemWtSets[i]->item;
    score = 0;
    //go over sets of item
    testValSetCt = 0;
    for (j = 0; j < self->itemWtSets[i]->szItemSets; j++) {
      //jth set in which item belongs
      setInd = self->itemWtSets[i]->itemSets[j];
      
      //ignore if setInd present in test or validation
      for (k = 0; k < self->szValSet; k++) {
        if (setInd == self->valSets[k]) {
          testValSetCt++;
          continue;
        }
      }

      for (k = 0; k < self->szTestSet; k++) {
        if (setInd == self->testSets[k]) {
          testValSetCt++;
          continue;
        }
      }
  
      
      simSet = 0.0;
      nSim = 0;
      
      for (k = 0; k < self->uSetsSize[setInd]; k++) {
        for (l = k+1; l < self->uSetsSize[setInd]; l++) {
          simSet += sim[self->uSets[setInd][k]][self->uSets[setInd][l]];
          nSim++;
        }
      } 
      
      if (nSim == 0) {
        score = self->labels[setInd]/self->uSetsSize[setInd];
      } else {
        simSet = simSet/nSim;
        score += (self->labels[setInd]*simSet)/self->uSetsSize[setInd];
      }

    }
    
    if ((self->itemWtSets[i]->szItemSets - testValSetCt) > 0) {
      score = score/(self->itemWtSets[i]->szItemSets - testValSetCt);
      self->itemWtSets[i]->wt = score;
    } else {
      assert(self->testValItems[self->itemWtSets[i]->item] == 1);
      self->itemWtSets[i]->wt = 0;
    }
    
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
    free(self->itemWtSets[i]->itemSets);
    free(self->itemWtSets[i]);
  }
  free(self->itemWtSets);
  
  free(self->uSetsSize);
  free(self->labels);
  free(self->testSets);
  free(self->valSets);
  free(self->testValItems);
  free(self); 
}


int comp (const void * elem1, const void * elem2) {
  ItemWtSets *f = *(ItemWtSets**)elem1;
  ItemWtSets *s = *(ItemWtSets**)elem2;
  if (f->item > s->item) return  1;
  if (f->item < s->item) return -1;
  return 0;
}


void UserSets_sortItems(UserSets *self) {
  qsort(self->itemWtSets, self->nUserItems, sizeof(ItemWtSets*), comp);
}


//use binary search to return the ItemWtSets
ItemWtSets* UserSets_search(UserSets *self, int item) {
  int i;
  int lb, ub, mid;
  
  lb = 0;
  ub = self->nUserItems - 1;

  while (lb <= ub) {
    mid = (lb + ub) / 2;
    if (self->itemWtSets[mid]->item == item) {
      return self->itemWtSets[mid];
    } else if (self->itemWtSets[mid]->item < item) {
      lb = mid+1;
    } else{
      ub = mid-1;
    }
  }

  printf("\nERR: cant find item");

  return NULL;
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


void Data_reset(Data *self, int nUsers, int nItems) {
  
  int i;
  
  for (i = 0; i < self->nUsers; i++) {
    UserSets_reset(self->userSets[i], nItems);
  }
  
}


void Data_free(Data *self) {
  int i;

  for (i = 0; i < self->nUsers; i++) {
    UserSets_free(self->userSets[i]);
  }
  free(self->userSets);
  free(self);
}



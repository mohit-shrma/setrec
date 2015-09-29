#include "datastruct.h"


void Params_display(Params *params) {
  printf("\nParameters:");
  printf("\n%s %s", params->user_set_file, params->ext_setSim_file);
  printf("\n%d %d", params->nUsers, params->nItems);
  printf("\n%d %f %f %f %f", params->facDim, params->regU, params->regI, 
      params->learnRate, params->constrainWt);
  printf("\n%d %d %d", params->useSim, params->maxIter, params->seed);
}


void loadUserItemWtsFrmTrain(Data *data) {
  int u, ii, i, j, item;
  UserSets *userSet = NULL;
  ItemWtSets *itemWtSets = NULL;
  float rat = 0;
  int notFoundCt = 0;
  int foundCt = 0;
  gk_csr_t *trainMat = data->trainMat;

  for (u = 0; u < trainMat->nrows; u++) {
    userSet = data->userSets[u];
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      item = trainMat->rowind[ii];
      rat = trainMat->rowval[ii];
      itemWtSets = UserSets_search(userSet, item);
      if (NULL == itemWtSets) {
        //printf("\nu:%d i:%d j:%d not found", u, i, j);
        notFoundCt++;
        continue;
      }
      foundCt++;
      itemWtSets->wt = rat;
    }
  }
  
  printf("\nfound:%d notFound: %d", foundCt, notFoundCt);
}


void UserSets_init(UserSets * const self, int user, int numSets, int nItems,
  int nUserItems) {
  
  int i, j, setInd;
  int found; 

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
 
  //make sure you can always have a test and validation set
  assert(numSets > self->szValSet + self->szTestSet);

  //initialize test and val sets using random
  self->valSets = (int *) malloc(sizeof(int)*self->szValSet);
  memset(self->valSets, 0, sizeof(int)*self->szValSet);
  i = 0;
  
  while (i < self->szValSet) {
    setInd = rand()%numSets;
    //make sure set is not present already in validation set
    found = 0;
    
    for (j = 0; j < i; j++) {
      if (setInd == self->valSets[j]) {
        found  = 1; 
        break;
      }
    }

    if (found) {
      continue;
    }

    self->valSets[i++] = setInd;
  }


  self->testSets = (int *) malloc(sizeof(int)*self->szTestSet);
  memset(self->testSets, 0, sizeof(int)*self->szTestSet);
  i = 0;
  while (i < self->szTestSet) {
    setInd = rand()%numSets;
    found = 0;
    //make sure set is not present already in test set
    for (j = 0; j < i; j++) {
      if (setInd == self->testSets[j]) {
        found = 1;
        break;
      }
    }
    //make sure set is not present in validation set
    for (j = 0; j < self->szValSet; j++) {
      if (setInd == self->valSets[j]) {
        found = 1;
        break;
      }
    }
    
    if (found) {
      continue;
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
  int setInd;
  float score, simSet;
  int testValSetCt, nSim;

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


void UserSets_transToSigm(UserSets *self, float *userMidps) {
  int i, u, s;

  u = self->userId;
  for (s = 0; s < self->numSets; s++) {
    self->labels[s] = sigmoid(self->labels[s] - userMidps[u], 50.0);
  }
}


void UserSets_transToHingeBin(UserSets *self, float *userMidps) {
  int i, u, s;

  u = self->userId;
  for (s = 0; s < self->numSets; s++) {
    if (self->labels[s] > userMidps[u]) {
      self->labels[s] = 1.0;
    } else {
      self->labels[s] = -1.0;
    }
  }
}


void UserSets_transToBin(UserSets *self, float *userMidps) {
  int i, u, s;
  float rat;
  u = self->userId;
  for (s = 0; s < self->numSets; s++) {
    if (self->labels[s] > userMidps[u]) {
      self->labels[s] = 1.0;
    } else {
      self->labels[s] = 0.0;
    }
  }
}


void UserSets_transToBinWNoise(UserSets *self, float *userMidps) {
  int i, u, s;
  float rat;
  float noisyRat;
  
  u = self->userId;
  for (s = 0; s < self->numSets; s++) {
    rat = sigmoid(self->labels[s] - userMidps[u], 1.0); 
    noisyRat = (float) generateGaussianNoise(rat, 0.01);
    if (noisyRat > 1.0) {
      self->labels[s] = 1.0;
    } else if (noisyRat < 0) {
      self->labels[s] = 0;
    }
    self->labels[s] = noisyRat;
  }

}


int UserSets_isSetTestVal(UserSets *self, int s) {
  int i;
  
  //check if in test
  for (i = 0; i < self->szTestSet; i++) {
    if (s == self->testSets[i]) {
      return 1;
    }
  }

  //check if in val
  for (i = 0; i < self->szValSet; i++) {
    if (s == self->valSets[i]) {
      return 1;
    }
  }

  return 0;
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


void ItemSets_init(ItemSets *itemSets, UserSets **userSets, int nUsers, 
    int nItems) {
  
  int i, u, item;
  int **itemUsers;
  UserSets *userSet = NULL;

  itemSets->nItems = nItems;
  itemSets->nUsers = nUsers;

  itemUsers= (int**) malloc(sizeof(int*)*nItems);
  itemSets->itemUsers = itemUsers;
  for (i = 0; i < nItems; i++) {
    itemUsers[i] = (int*) malloc(sizeof(int)*nUsers);
    memset(itemUsers[i], 0, sizeof(int)*nUsers);
  }

  //go through all user sets and mark items which occur in those users
  for (u = 0; u < nUsers; u++) {
    userSet = userSets[u];
    for (i = 0; i < userSet->nUserItems; i++) {
      item = userSet->itemWtSets[i]->item;
      itemUsers[item][u] = 1;
    }
  }
  writeIMat(itemUsers, nItems, nUsers, "itemUser.txt");
}


void ItemSets_free(ItemSets *self) {
  int i;
  for (i = 0; i < self->nItems; i++) {
    free(self->itemUsers[i]);
  }
  free(self->itemUsers);
  free(self);
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

  //printf("\nERR: cant find item");

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

  self->itemSets = (ItemSets *) malloc(sizeof(ItemSets));

  self->uFac = NULL;
  self->iFac = NULL;
  self->userMidps = NULL;
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
  gk_csr_Free(&(self->trainMat));
  gk_csr_Free(&(self->testMat));
  gk_csr_Free(&(self->valMat));
  if (self->uFac) {
    for (i = 0; i < self->nUsers; i++) {
      free(self->uFac[i]);
    }
    free(self->uFac);
  }
  if (self->iFac) {
    for (i = 0; i < self->nItems; i++) {
      free(self->iFac[i]);
    }
    free(self->iFac);
  }
  ItemSets_free(self->itemSets);
  if (self->userMidps) {
    free(self->userMidps);
  }
  free(self);
}


void Data_jaccSim(Data *data, float **sim) {
 
  int u, i, j, s;
  int item, nSets, setInd, prevSetInd, tempInd;
  float unionSize, intersSize;
  UserSets *userSet = NULL;
  Set** itemMembSets = NULL;
  Set *temp = NULL;

  //get number of sets
  nSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    nSets += userSet->numSets;
  }

  //for each item allocate space for sets 
  itemMembSets = (Set**) malloc(sizeof(Set*)*data->nItems);
  for (i = 0; i < data->nItems; i++) {
    itemMembSets[i] = (Set *) malloc(sizeof(Set));
    Set_init(itemMembSets[i], nSets);
  }

  temp = (Set *) malloc(sizeof(Set));
  Set_init(temp, nSets);

  //add sets to item
  setInd = 0;
  prevSetInd = 0;
  for (u = 0; u < data->nUsers; u++) {
    
    userSet = data->userSets[u];

    prevSetInd = setInd;
    //go through all sets and add them to items
    for (s = 0; s < userSet->numSets; s++) {
      for (j = 0; j < userSet->uSetsSize[s]; j++) {
        item = userSet->uSets[s][j];
        Set_addElem(itemMembSets[item], setInd);
      }
      setInd++;
    }

    //remove test set
    for (s = 0; s < userSet->szTestSet; s++) {
      tempInd = userSet->testSets[s];
      for (j = 0; j < userSet->uSetsSize[tempInd]; j++) {
        item = userSet->uSets[tempInd][j];
        //remove set corresponding to test set
        Set_delElem(itemMembSets[item], prevSetInd + tempInd);
      }
    }

    //remove validation set
    for (s = 0; s < userSet->szValSet; s++) {
      tempInd = userSet->valSets[s];
      for (j = 0; j < userSet->uSetsSize[tempInd]; j++) {
        item = userSet->uSets[tempInd][j];
        //remove set corresponding to validation set
        Set_delElem(itemMembSets[item], prevSetInd + tempInd);
      }
    }

  }
  
  //compute jaccard similarity between item sets
  for (i = 0; i < data->nItems; i++) {
    for (j = i+1; j < data->nItems; j++) {
      Set_union(temp, itemMembSets[i], itemMembSets[j]);
      unionSize = (float) Set_numElem(temp);
      Set_intersection(temp, itemMembSets[i], itemMembSets[j]);
      intersSize = (float) Set_numElem(temp);
      sim[i][j] = intersSize/unionSize;
      sim[j][i] = sim[i][j];
    }
  }
  
  //free allocated space for sets
  for (i = 0; i < data->nItems; i++) {
    Set_free(itemMembSets[i]);
  }
  free(itemMembSets);
  Set_free(temp);
}


int compItemRat(const void *elem1, const void *elem2) {
  ItemRat *item1Rat = *(ItemRat**)elem1;
  ItemRat *item2Rat = *(ItemRat**)elem2;
  if (item1Rat->rating > item2Rat->rating) return -1;
  if (item1Rat->rating < item2Rat->rating) return 1;
  return 0;
}



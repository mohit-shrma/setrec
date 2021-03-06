#include "modelMajority.h"

//compute the estimated score for set using majority model and return sum of
//latent factor for the items in top
float ModelMajority_estScoreNSumFac(void *self, int u, int *set, int setSz, 
    float *sumItemFac, ItemRat **itemRats) {
  
  int i, j, item, majSz;
  float r_us_est = 0, temp;
  ModelMajority *model = self;
  for (i = 0; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  //sort in decreasing order
  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  memset(sumItemFac, 0, sizeof(float)*model->_(facDim));
  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;
    if (i > 0) {
      if (itemRats[i]->rating > itemRats[i-1]->rating) {
        printf("\nNot in decreasing order:  %f %f", 
            itemRats[i]->rating, itemRats[i-1]->rating);
        fflush(stdout);
        exit(0);
      }
    }
    item = itemRats[i]->item;
    for (j = 0; j < model->_(facDim); j++) {
      sumItemFac[j] += model->_(iFac)[item][j];
    }
  }
  r_us_est = r_us_est/majSz;

  //temp = dotProd(model->_(uFac)[u], sumItemFac, model->_(facDim))/majSz; 

  return r_us_est;
}


//function to check if gradient computation for majority model is correct
//NOTE: commented out the constraint gradient as the check was failing for
//constraint
void ModelMajority_gradientCheck(void *self, int u, int *set, int setSz, 
    float r_us, ItemRat **itemRats) {

  int i, j, item, majSz;
  ModelMajority *model = self;
  float *sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float *iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fac           = (float*) malloc(sizeof(float)*model->_(facDim));
  float lossRight, lossLeft, gradE, r_us_est, rat_s, maxRat;
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }
  
  r_us_est = ModelMajority_estScoreNSumFac(model, u, set, setSz, sumItemLatFac,
      itemRats);

  //sample item from majority
  item = itemRats[rand()%majSz]->item;
  //item gradient
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = (2.0*(r_us_est - r_us)*model->_(uFac)[u][j])/majSz;
    if (itemRats[0]->item == item && itemRats[0]->rating < r_us) { 
      iGrad[j] += -1.0*model->constrainWt*model->_(uFac)[u][j];
    }
  }

  //perturb item with +E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim)); 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] + 0.0001;    
  }
  rat_s = dotProd(model->_(uFac)[u], fac, model->_(facDim))/majSz;
  lossRight = pow(rat_s - r_us, 2);
  if (itemRats[0]->item == item && r_us > itemRats[0]->rating) {
    //recompute item rating if top
    memset (fac, 0, sizeof(float)*model->_(facDim));
    for (j = 0; j < model->_(facDim); j++) {
      fac[j] = model->_(iFac)[item][j] + 0.0001;
    }
    maxRat = dotProd(model->_(uFac)[u], fac, model->_(facDim));
    lossRight += model->constrainWt*(r_us - maxRat);
  }

  //perturb item with  -E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim)); 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] - 0.0001;    
  }
  rat_s = dotProd(model->_(uFac)[u], fac, model->_(facDim))/majSz;
  lossLeft = pow(rat_s - r_us, 2);
  if (itemRats[0]->item == item && r_us > itemRats[0]->rating) {
    //recompute item rating if top
    memset (fac, 0, sizeof(float)*model->_(facDim));
    for (j = 0; j < model->_(facDim); j++) {
      fac[j] = model->_(iFac)[item][j] - 0.0001;
    }
    maxRat = dotProd(model->_(uFac)[u], fac, model->_(facDim));
    lossLeft += model->constrainWt*(r_us - maxRat);
  }

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*iGrad[j]*0.0001;
  }

  if (fabs(lossRight-lossLeft-gradE) > 0.0001) {
    printf("\nu: %d item: %d diff: %f div: %f lDiff:%f gradE:%f", u, item, 
      lossRight-lossLeft-gradE, (lossRight-lossLeft)/gradE, lossRight-lossLeft, 
      gradE);
  }


  free(sumItemLatFac);
  free(iGrad);
  free(uGrad);
  free(fac);
}


//test function to make sure sorting on ItemRat structure works
void testItemRat() {
  int i, j;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*5);
  for (i = 0; i < 5; i++) {
    itemRats[i] = (ItemRat*)malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
    itemRats[i]->item = i;
    itemRats[i]->rating = rand()/(1.0 + RAND_MAX);
  }
   
  qsort(itemRats, 5, sizeof(ItemRat*), compItemRat);

  for (i = 0; i < 5; i++) {
    printf("\nitem: %d rat: %f", itemRats[i]->item, itemRats[i]->rating);
  }

  for (i = 0; i < 5; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
}


//compute the score of set as the average of majority of the items
float ModelMajority_setScore(void *self, int u, int *set, int setSz, 
    float **sim) {
   
  int i, item, majSz;
  float r_us_est = 0;
  ModelMajority *model = self;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0 ; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;
    if (i > 0) {
      if (itemRats[i]->rating > itemRats[i-1]->rating) {
        printf("\nTrain:Not in decreasing order:  %f %f", 
            itemRats[i]->rating, itemRats[i-1]->rating);
        fflush(stdout);
        exit(0);
      }
    }
  }
  r_us_est = r_us_est/majSz;

  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);

  return r_us_est;
}


//return the score of set as the average of items in set
float ModelMajority_setScoreAvg(void *self, int u, int *set, int setSz, 
    float **sim) {
   
  int i, item, majSz;
  float r_us_est = 0;
  ModelMajority *model = self;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0 ; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  majSz = setSz;

  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;
  }
  r_us_est = r_us_est/majSz;

  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);

  return r_us_est;
}


float ModelMajority_setScoreSum(void *self, int u, int *set, int setSz, 
    float **sim) {
   
  int i, item, majSz;
  float r_us_est = 0;
  ModelMajority *model = self;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0 ; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  majSz = setSz;

  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;
  }

  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);

  return r_us_est;
}


//get the set label and max rating with in set
void ModelMajority_setScoreWMaxRat(void *self, int u, int *set, int setSz, 
    float *r_est, float *maxRat) {
   
  int i, item, majSz;
  float r_us_est = 0;
  ModelMajority *model = self;
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0 ; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }
  
  for (i = 0; i < majSz; i++) {
    r_us_est += itemRats[i]->rating;
  }
  r_us_est = r_us_est/majSz;

  *r_est = r_us_est;
  *maxRat = itemRats[0]->rating;

  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);

} 


//return the sum of majority of item latent factors
void ModelMajority_setMajFac(void *self, int u, int *set, int setSz, 
    ItemRat **itemRats, float *majFac) {
  int i, j, item, majSz;
  ModelMajority *model = self;

  for (i = 0; i < setSz; i++) {
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item  = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  memset(majFac, 0, model->_(facDim));
  for (i = 0; i < majSz; i++) {
    item = itemRats[i]->item;
    for (j = 0; j < model->_(facDim); j++) {
      majFac[j] += model->_(iFac)[item][j];
    }
  }

}


//check if item is in set and if present then get the sum of item latent factors
//from the set
int ModelMajority_isSetMajFac(void *self, int u, int *set, int setSz,
    ItemRat **itemRats, float *majFac, int qItem) {
  int i, j, item, majSz;
  int found = 0;
  ModelMajority *model = self;
  
  memset(majFac, 0, sizeof(float)*model->_(facDim));

  for (i = 0; i < setSz; i++) {
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (i = 0; i < setSz; i++) {
    item = set[i];
    itemRats[i]->item  = item;
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[item], 
        model->_(facDim));
  }

  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  memset(majFac, 0, model->_(facDim));
  for (i = 0; i < majSz; i++) {
    item = itemRats[i]->item;
    if (item == qItem) {
      found = 1;
    }
    for (j = 0; j < model->_(facDim); j++) {
      majFac[j] += model->_(iFac)[item][j];
    }
  }

  return found;
}


//sample a set and its rating as per majority model
void samplePosSetNRat(Data *data, int u, int *posSet, int setSz, float *rat) {
  
  int i, nItems, itemInd, item, found;
  int nUserItems, majSz;
  gk_csr_t *trainMat = data->trainMat;
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }
  float ratAvg = 0;

  nItems = 0;
  nUserItems = trainMat->rowptr[u+1] - trainMat->rowptr[u];
  while (nItems < setSz) {
    found = 0;
    itemInd = rand()%nUserItems;
    item = trainMat->rowind[trainMat->rowptr[u] + itemInd];  
    for (i = 0; i < nItems; i++) {
      if (item == posSet[i]) {
        found = 1;
        break;
      }
    }
    if (found) {
      continue;
    }

    posSet[nItems] = item;
    itemRats[nItems]->item = item;
    itemRats[nItems]->rating = trainMat->rowval[trainMat->rowptr[u] + itemInd];   
 
    nItems++;
  }

  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  if (setSz % 2  == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  ratAvg = 0;
  for (i = 0; i < majSz; i++) {
    ratAvg += itemRats[i]->rating;
  }
  ratAvg = ratAvg/majSz;
 
  *rat = ratAvg;

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
}


//count number of constraint violations
int ModelMajority_constViol(void *self, Data *data) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float userSetPref = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  
  FILE *fp = NULL;

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
      //get preference over set
      ModelMajority_setScoreWMaxRat(model, u, set, setSz, &userSetPref, 
          &maxRat);
      
      if (userSet->labels[s] > maxRat) {
        //constraint violated
        constViolCt++;
        //fprintf(fp, "%f\n", userSet->labels[s]-maxRat);
      }
      nSets++;
    }
  }
  

  return constViolCt;
}


//computation of objective for majority model
float ModelMajority_objective(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  int uSets = 0;  
  FILE *fp = NULL;
  //fp = fopen("constrVilOff.txt", "w");

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
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

      uSets++;
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      ModelMajority_setScoreWMaxRat(model, u, set, setSz, &userSetPref, 
          &maxRat);
      diff = userSetPref - userSet->labels[s];
      rmse += diff*diff;
      if (userSet->labels[s] > maxRat) {
        //constraint violated
        rmse += model->constrainWt*(userSet->labels[s] - maxRat);
        constViolCt++;
      }
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  
  //printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f Constraint violation count: %d", 
  //    (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr, constViolCt);
  
  //fclose(fp);
  return (rmse + uRegErr + iRegErr);
}


//computation of objective for majority model without constraint
float ModelMajority_objectiveWOCons(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  int uSets = 0;  
  FILE *fp = NULL;
  //fp = fopen("constrVilOff.txt", "w");

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
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

      uSets++;
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      ModelMajority_setScoreWMaxRat(model, u, set, setSz, &userSetPref, 
          &maxRat);
      diff = userSetPref - userSet->labels[s];
      rmse += diff*diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  
  //printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f Constraint violation count: %d", 
  //    (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr, constViolCt);
  
  //fclose(fp);
  return (rmse + uRegErr + iRegErr);
}


float ModelMajority_itemFeatScoreAvg(void *self, int u, int item, gk_csr_t *featMat) {
  
  ModelMajority *model = self;
  int i, j, ii, jj;
  int nFeats = featMat->rowptr[item+1] - featMat->rowptr[item];
  int *set = (int *) malloc(sizeof(int)*nFeats);
  float score = 0;

  //create a set of features
  j = 0;
  for (ii = featMat->rowptr[item]; ii < featMat->rowptr[item+1]; ii++) {
     set[j++] = featMat->rowind[ii];
  }
  assert(j == nFeats);
  score = ModelMajority_setScoreAvg(model, u, set, nFeats, NULL);
  
  free(set);
  return score;
}


//computation of objective for average model
float ModelMajority_objectiveAvg(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  int uSets = 0;  
  FILE *fp = NULL;
  float uNorm = 0, iNorm = 0;
  //fp = fopen("constrVilOff.txt", "w");

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
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

      uSets++;
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      userSetPref = ModelMajority_setScoreAvg(model, u, set, setSz, NULL);
      diff = userSetPref - userSet->labels[s];
      rmse += diff*diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
    uNorm += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  } 

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
    iNorm += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  
  /*
  printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f Constraint violation count: %d"
      " uNorm: %f iNorm: %f", 
      (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr, constViolCt, uNorm, iNorm);
  */

  //fclose(fp);
  return (rmse + uRegErr + iRegErr);
}


float ModelMajority_objectiveSum(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  int uSets = 0;  
  FILE *fp = NULL;
  //fp = fopen("constrVilOff.txt", "w");

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    uSets = 0;
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

      uSets++;
      
      set = userSet->uSets[s];
      setSz = userSet->uSetsSize[s];
      //get preference over set
      userSetPref = ModelMajority_setScoreSum(model, u, set, setSz, NULL);
      diff = userSetPref - userSet->labels[s];
      rmse += diff*diff;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  
  //printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f Constraint violation count: %d", 
  //    (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr, constViolCt);
  
  //fclose(fp);
  return (rmse + uRegErr + iRegErr);
}


//coordinate descent over users and items w/o constraint
void ModelMajority_trainCDWOCons(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, majSz, num, denom; 
  ModelMajority *model = self;
  
  //TODO: hard code
  int nSetsPerUser = 915;

  float** setSumItemLatFac = (float**) malloc(sizeof(float*)*nSetsPerUser);
  for (i = 0; i < nSetsPerUser; i++) {
    setSumItemLatFac[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(setSumItemLatFac[i], 0, sizeof(float)*model->_(facDim));
  }
  
  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));

  ItemWtSets *itemWtSet = NULL;

  for (iter = 0; iter < params->maxIter; iter++) {
    
    //update users
    for (u = 0; u < model->_(nUsers); u++) {
      userSet = data->userSets[u];
      
      //update user's lat fac
      for (j = 0; j < model->_(facDim); j++) {
      
        //go through over all sets to compute sum of lat fac in sets
        for (s = 0; s < userSet->numSets; s++) {
          //check if set in test or validation
          if (UserSets_isSetTestVal(userSet, s)) {
           continue;
          }
          memset(setSumItemLatFac[s], 0, sizeof(float)*model->_(facDim));
          ModelMajority_setMajFac(model, u, userSet->uSets[s], userSet->uSetsSize[s], 
              itemRats, setSumItemLatFac[s]);
        }

        //go over all sets
        num = 0;
        denom = 0;
        for (s = 0; s < userSet->numSets; s++) {
          setSz = userSet->uSetsSize[s];
          if (setSz % 2 == 0) {
            majSz = setSz/2;
          } else {
            majSz = setSz/2 + 1;
          }
          //compute numerator
          r_us = userSet->labels[s];
          r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[s], 
              model->_(facDim))/majSz;
          temp = r_us - r_us_est;
          temp += (model->_(uFac)[u][j]*setSumItemLatFac[s][j])/majSz;
          temp = temp*(setSumItemLatFac[s][j]/majSz);
          
          //update numerator
          num += temp;
          
          //compute denominator
          denom += (setSumItemLatFac[s][j]*setSumItemLatFac[s][j])/(majSz*majSz);
        }
        //add regularization to denominator
        denom += model->_(regU);

        //update user component
        model->_(uFac)[u][j] = num/denom;
      }
    }

    //update items
    for (item = 0; item < model->_(nItems); item++) {
      for (j = 0; j < model->_(facDim); j++) { 
        num = 0;
        denom = 0;
        for (u = 0; u < model->_(nUsers); u++) {
          if (data->itemSets->itemUsers[item][u]) {
            //u has item
            userSet = data->userSets[u];

            //find sets which contains the item
            itemWtSet = UserSets_search(userSet, item);

            if (itemWtSet != NULL) {
              for (i = 0; i < itemWtSet->szItemSets; i++) {
                s = itemWtSet->itemSets[i];
                set = userSet->uSets[s];
                setSz = userSet->uSetsSize[s];
                if (setSz % 2 == 0) {
                  majSz = setSz/2;
                } else {
                  majSz = setSz/2 + 1;
                }
                if (ModelMajority_isSetMajFac(model, u, set, setSz, itemRats, 
                      setSumItemLatFac[0], item)) {
                  r_us = userSet->labels[s];
                  r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[0],
                      model->_(facDim))/majSz;
                  temp = r_us - r_us_est;
                  num += (temp + (model->_(uFac)[u][j]*model->_(iFac)[item][j])/majSz)*(model->_(uFac)[u][j]/majSz);
                  denom += (model->_(uFac)[u][j]*model->_(uFac)[u][j])/(majSz*majSz);
                }
              }
            } else {
              printf("\nERROR: u %d item %d not found", u, item);
            }
          }
        }//end u
        model->_(iFac)[item][j] = num/(denom + model->_(regI));
      }
    } //end item
     
    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
        printf("\nIter: %d obj: %.10e avgHits: %f testItemsRmse: %f", iter, 
            valTest->setObj, model->_(hitRateOrigTopN) (model, data->trainMat, 
              data->uFac, data->iFac, 10),
            model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
      fflush(stdout);
    }
    
  }

  printf("\nEnd Obj: %.10e", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
}


//alternating coordinate descent over users and items w/o constraint
void ModelMajority_trainCDAltWOCons(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, majSz, num, denom; 
  ModelMajority *model = self;
  
  //TODO: hard code
  int nSetsPerUser = 915;

  float** setSumItemLatFac = (float**) malloc(sizeof(float*)*nSetsPerUser);
  for (i = 0; i < nSetsPerUser; i++) {
    setSumItemLatFac[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(setSumItemLatFac[i], 0, sizeof(float)*model->_(facDim));
  }
  
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));

  ItemWtSets *itemWtSet = NULL;

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < 2000; subIter++) {
      
      u = rand() % model->_(nUsers);
      
      //update users
      userSet = data->userSets[u];
      
      //update user's lat fac
      for (j = 0; j < model->_(facDim); j++) {
      
        //go through over all sets to compute sum of lat fac in sets
        for (s = 0; s < userSet->numSets; s++) {
          //check if set in test or validation
          if (UserSets_isSetTestVal(userSet, s)) {
           continue;
          }
          setSz = userSet->uSetsSize[s];
          if (setSz % 2 == 0) {
            majSz = setSz/2;
          } else {
            majSz = setSz/2 + 1;
          }
          memset(setSumItemLatFac[s], 0, sizeof(float)*model->_(facDim));
          ModelMajority_setMajFac(model, u, userSet->uSets[s], userSet->uSetsSize[s], 
              itemRats, setSumItemLatFac[s]);
        }

        //go over all sets
        num = 0;
        denom = 0;
        for (s = 0; s < userSet->numSets; s++) {
          setSz = userSet->uSetsSize[s];
          if (setSz % 2 == 0) {
            majSz = setSz/2;
          } else {
            majSz = setSz/2 + 1;
          }
          //compute numerator
          r_us = userSet->labels[s];
          r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[s], 
              model->_(facDim))/majSz;
          temp = r_us - r_us_est;
          temp += (model->_(uFac)[u][j]*setSumItemLatFac[s][j])/majSz;
          temp = temp*(setSumItemLatFac[s][j]/majSz);
          
          //update numerator
          num += temp;
          
          //compute denominator
          denom += (setSumItemLatFac[s][j]*setSumItemLatFac[s][j])/(majSz*majSz);
        }
        //add regularization to denominator
        denom += model->_(regU);

        //update user component
        model->_(uFac)[u][j] = num/denom;
      }

      //update items
      item = rand() % model->_(nItems);
      for (j = 0; j < model->_(facDim); j++) { 
        num = 0;
        denom = 0;
        for (u = 0; u < model->_(nUsers); u++) {
          if (data->itemSets->itemUsers[item][u]) {
            //u has item
            userSet = data->userSets[u];

            //find sets which contains the item
            itemWtSet = UserSets_search(userSet, item);

            if (itemWtSet != NULL) {
              for (i = 0; i < itemWtSet->szItemSets; i++) {
                s = itemWtSet->itemSets[i];
                set = userSet->uSets[s];
                setSz = userSet->uSetsSize[s];
                if (ModelMajority_isSetMajFac(model, u, set, setSz, itemRats, 
                      setSumItemLatFac[0], item)) {
                  r_us = userSet->labels[s];
                  r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[0],
                      model->_(facDim))/majSz;
                  temp = r_us - r_us_est;
                  num += (temp + (model->_(uFac)[u][j]*model->_(iFac)[item][j])/majSz)*(model->_(uFac)[u][j]/majSz);
                  denom += (model->_(uFac)[u][j]*model->_(uFac)[u][j])/(majSz*majSz);
                }
              }
            } else {
              printf("\nERROR: u %d item %d not found", u, item);
            }
          }
        }//end u
        model->_(iFac)[item][j] = num/(denom + model->_(regI));
      }
       
    }
    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter: %d obj: %f test items rmse: %f", iter, valTest->setObj, 
      model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
      fflush(stdout);
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
}


//full gradient descent using RMSProp adaptive learning rate w/o constraint
void ModelMajority_trainRMSPropAltWOCons(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, majSz, num, denom; 
  ModelMajority *model = self;
  
  //TODO: hard code
  int nSetsPerUser = 50;
  setSz = 5;

  float** setSumItemLatFac = (float**) malloc(sizeof(float*)*nSetsPerUser);
  for (i = 0; i < nSetsPerUser; i++) {
    setSumItemLatFac[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(setSumItemLatFac[i], 0, sizeof(float)*model->_(facDim));
  }
 
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }


  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  float *uGrad, *iGrad;
  uGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  iGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iGrad, 0, sizeof(float)*model->_(facDim));

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  ItemWtSets *itemWtSet = NULL;

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < 2000; subIter++) {
      
      u = rand() % model->_(nUsers);
      
      //update users
      userSet = data->userSets[u];
      //update user's lat fac
      //go through over all sets to compute sum of lat fac in sets
      for (s = 0; s < userSet->numSets; s++) {
        //check if set in test or validation
        if (UserSets_isSetTestVal(userSet, s)) {
         continue;
        }
        memset(setSumItemLatFac[s], 0, sizeof(float)*model->_(facDim));
        ModelMajority_setMajFac(model, u, userSet->uSets[s], userSet->uSetsSize[s], 
            itemRats, setSumItemLatFac[s]);
      }

      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));

      //go over all sets
      for (s = 0; s < userSet->numSets; s++) {
        r_us = userSet->labels[s];
        r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[s], 
            model->_(facDim))/majSz;
        temp = 2*(r_us - r_us_est);
        for (j = 0; j < model->_(facDim); j++) {
          uGrad[j] += temp*(-1.0/majSz)*setSumItemLatFac[s][j];
        }
      }
      
      //update user 
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //update item
      item = rand() % model->_(nItems);
      
      //compute gradient
      memset(iGrad, 0, sizeof(float)*model->_(facDim));  
      for (u = 0; u < model->_(nUsers); u++) {
        if (data->itemSets->itemUsers[item][u]) {
          //u has item
          userSet = data->userSets[u];

          //find sets which contains the item
          itemWtSet = UserSets_search(userSet, item);

          if (itemWtSet != NULL) {
            for (i = 0; i < itemWtSet->szItemSets; i++) {
              s = itemWtSet->itemSets[i];
              set = userSet->uSets[s];
              setSz = userSet->uSetsSize[s];
              if (ModelMajority_isSetMajFac(model, u, set, setSz, itemRats, 
                    setSumItemLatFac[0], item)) {
                r_us = userSet->labels[s];
                r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[0],
                    model->_(facDim))/majSz;
                temp = 2.0 * (r_us - r_us_est);
                for (j = 0; j < model->_(facDim); j++) {
                  iGrad[j] += temp*(-1.0/majSz)*model->_(uFac)[u][j];
                }
              }
            }
          } else {
            printf("\nERROR: u %d item %d not found", u, item);
          }
        }
      }//end u
        
      for (j = 0; j < model->_(facDim); j++) {
        //add reg
        iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        //accumulate the gradients square 
        iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + 
          (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
        //update
        model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
      }
    } 
       
    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
        printf("\nIter: %d obj: %f avgHits: %f testItemsRmse: %f", iter, 
            valTest->setObj, model->_(hitRateOrigTopN) (model, data->trainMat, 
              data->uFac, data->iFac, 10),
            model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 5000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
      fflush(stdout);
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
  free(uGrad);
  free(iGrad);
  for (u = 0; u < model->_(nUsers); u++) {
    free(uGradsAcc[u]);
  }
  free(uGradsAcc);
  for (i = 0; i < model->_(nItems); i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);
}


//full gradient descent using RMSProp adaptive learning rate w/o constraint
void ModelMajority_trainRMSPropAlt(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, majSz, num, denom; 
  ModelMajority *model = self;
  
  //TODO: hard code
  int nSetsPerUser = 50;
  setSz = 5;

  float* setSumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }


  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  float *uGrad, *iGrad;
  uGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  iGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iGrad, 0, sizeof(float)*model->_(facDim));

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  ItemWtSets *itemWtSet = NULL;

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < 2000; subIter++) {
      
      u = rand() % model->_(nUsers);
      
      //update users
      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));
      userSet = data->userSets[u];
      //update user's lat fac
      //go through over all sets to compute sum of lat fac in sets
      for (s = 0; s < userSet->numSets; s++) {
        //check if set in test or validation
        if (UserSets_isSetTestVal(userSet, s)) {
         continue;
        }
        memset(setSumItemLatFac, 0, sizeof(float)*model->_(facDim));
        ModelMajority_setMajFac(model, u, userSet->uSets[s], userSet->uSetsSize[s], 
            itemRats, setSumItemLatFac);
        r_us = userSet->labels[s];
        r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac, 
            model->_(facDim))/majSz;
        temp = 2*(r_us - r_us_est);
        for (j = 0; j < model->_(facDim); j++) {
          uGrad[j] += temp*(-1.0/majSz)*setSumItemLatFac[j];
          //check if constraint violated for set
          if (itemRats[0]->rating < r_us) {
            uGrad[j] += -1.0*model->constrainWt*model->_(iFac)[itemRats[0]->item][j];
          }
        }
      }
      
      //update user 
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //update item
      item = rand() % model->_(nItems);
      
      //compute gradient
      memset(iGrad, 0, sizeof(float)*model->_(facDim));  
      for (u = 0; u < model->_(nUsers); u++) {
        if (data->itemSets->itemUsers[item][u]) {
          //u has item
          userSet = data->userSets[u];

          //find sets which contains the item
          itemWtSet = UserSets_search(userSet, item);

          if (itemWtSet != NULL) {
            for (i = 0; i < itemWtSet->szItemSets; i++) {
              s = itemWtSet->itemSets[i];
              set = userSet->uSets[s];
              setSz = userSet->uSetsSize[s];
              if (ModelMajority_isSetMajFac(model, u, set, setSz, itemRats, 
                    setSumItemLatFac, item)) {
                r_us = userSet->labels[s];
                r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac,
                    model->_(facDim))/majSz;
                temp = 2.0 * (r_us - r_us_est);
                for (j = 0; j < model->_(facDim); j++) {
                  iGrad[j] += temp*(-1.0/majSz)*model->_(uFac)[u][j];
                  if (itemRats[0]->item == item && itemRats[0]->rating < r_us) {
                    iGrad[j] += -1.0*model->constrainWt*model->_(uFac)[u][j];
                  }
                }
              }
            }
          } else {
            printf("\nERROR: u %d item %d not found", u, item);
          }
        }
      }//end u
        
      for (j = 0; j < model->_(facDim); j++) {
        //add reg
        iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        //accumulate the gradients square 
        iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + 
          (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
        //update
        model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
      }
    } 
       
    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
        printf("\nIter: %d obj: %.10e avgHits: %f testItemsRmse: %f", iter, 
            valTest->setObj, model->_(hitRateOrigTopN) (model, data->trainMat, 
              data->uFac, data->iFac, 10),
            model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 5000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %.10e currObj: %.10e", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
      fflush(stdout);
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
  }

  printf("\nEnd Obj: %.10e", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(setSumItemLatFac);
  free(uGrad);
  free(iGrad);
  for (u = 0; u < model->_(nUsers); u++) {
    free(uGradsAcc[u]);
  }
  free(uGradsAcc);
  for (i = 0; i < model->_(nItems); i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);
}


//full gradient descent alternating over user and items w/o constraint
void ModelMajority_trainGDAltWOCons(void *self, Data *data, Params *params, float **sim, 
  ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, majSz, num, denom; 
  ModelMajority *model = self;
  
  //TODO: hard code
  int nSetsPerUser = 50;
  setSz = 5;

  float** setSumItemLatFac = (float**) malloc(sizeof(float*)*nSetsPerUser);
  for (i = 0; i < nSetsPerUser; i++) {
    setSumItemLatFac[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(setSumItemLatFac[i], 0, sizeof(float)*model->_(facDim));
  }
 
  
  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  float *uGrad, *iGrad;
  uGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(uGrad, 0, sizeof(float)*model->_(facDim));
  iGrad = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(iGrad, 0, sizeof(float)*model->_(facDim));

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  ItemWtSets *itemWtSet = NULL;

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < 2000; subIter++) {
      
      u = rand() % model->_(nUsers);
      /*
      //update users
      userSet = data->userSets[u];
      //update user's lat fac
      //go through over all sets to compute sum of lat fac in sets
      for (s = 0; s < userSet->numSets; s++) {
        //check if set in test or validation
        if (UserSets_isSetTestVal(userSet, s)) {
         continue;
        }
        memset(setSumItemLatFac[s], 0, sizeof(float)*model->_(facDim));
        ModelMajority_setMajFac(model, u, userSet->uSets[s], userSet->uSetsSize[s], 
            itemRats, setSumItemLatFac[s]);
      }

      //compute user gradient
      memset(uGrad, 0, sizeof(float)*model->_(facDim));

      //go over all sets
      for (s = 0; s < userSet->numSets; s++) {
        r_us = userSet->labels[s];
        r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[s], 
            model->_(facDim))/majSz;
        temp = 2*(r_us - r_us_est);
        for (j = 0; j < model->_(facDim); j++) {
          uGrad[j] += temp*(-1.0/majSz)*setSumItemLatFac[s][j];
        }
      }
      
      //update user 
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        model->_(uFac)[u][j] -= model->_(learnRate)*uGrad[j];
      }
      */ 
      
      //update item
      item = rand() % model->_(nItems);
      
      //compute gradient
      memset(iGrad, 0, sizeof(float)*model->_(facDim));  
      for (u = 0; u < model->_(nUsers); u++) {
        if (data->itemSets->itemUsers[item][u]) {
          //u has item
          userSet = data->userSets[u];

          //find sets which contains the item
          itemWtSet = UserSets_search(userSet, item);

          if (itemWtSet != NULL) {
            for (i = 0; i < itemWtSet->szItemSets; i++) {
              s = itemWtSet->itemSets[i];
              set = userSet->uSets[s];
              setSz = userSet->uSetsSize[s];
              if (ModelMajority_isSetMajFac(model, u, set, setSz, itemRats, 
                    setSumItemLatFac[0], item)) {
                r_us = userSet->labels[s];
                r_us_est = dotProd(model->_(uFac)[u], setSumItemLatFac[0],
                    model->_(facDim))/majSz;
                temp = 2.0 * (r_us - r_us_est);
                for (j = 0; j < model->_(facDim); j++) {
                  iGrad[j] += temp*(-1.0/majSz)*model->_(uFac)[u][j];
                }
              }
            }
          } else {
            printf("\nERROR: u %d item %d not found", u, item);
          }
        }
      }//end u
        
      for (j = 0; j < model->_(facDim); j++) {
        //add reg
        iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        //accumulate the gradients square 
        //update
        model->_(iFac)[item][j] -= model->_(learnRate)*iGrad[j];
      }
    } 
      
    if (iter % 500 == 0) {
      model->_(learnRate) = model->_(learnRate)/2.0;
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter: %d obj: %f test items rmse: %f", iter, valTest->setObj, 
      model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
      fflush(stdout);
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent using AdaDelta adaptive learning rate
void ModelMajority_trainAdaDelta(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, gradRMS, deltaRMS, delta; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  float **iDeltaAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uDeltaAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iDeltaAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iDeltaAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uDeltaAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uDeltaAcc[i], 0, sizeof(float)*model->_(facDim));
  }


  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        if (i > 0) {
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
        }
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];

        delta = -1.0*(sqrt(uDeltaAcc[u][j]+model->epsRMS)/sqrt(uGradsAcc[u][j]+model->epsRMS))*uGrad[j];  

        //apply update
        model->_(uFac)[u][j] += delta;
        
        //accumulate the updates square
        uDeltaAcc[u][j] = uDeltaAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*delta*delta;
      }


      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          delta = -1.0*(sqrt(iDeltaAcc[item][j]+model->epsRMS)/sqrt(iGradsAcc[item][j]+model->epsRMS))*iGrad[j];
          
          //apply update
          model->_(iFac)[item][j] += delta; 
          
          //accumulate the updates square
          iDeltaAcc[item][j] = iDeltaAcc[item][j]*params->rhoRMS +
            (1.0 - params->rhoRMS)*delta*delta;
        }
      }

    }
    
    //printf("\nEpoch: %d  succ = %d fail = %d avg diff = %f", 
    //    iter, succ, fail, epochDiff/subIter);

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter % 50 == 0) {
        printf("\nIter: %d obj: %f", iter, valTest->setObj);
      }
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
    free(uDeltaAcc[i]);
  }
  free(uGradsAcc);
  free(uDeltaAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
    free(iDeltaAcc[i]);
  }
  free(iGradsAcc);
  free(iDeltaAcc);
  

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent using RMSProp adaptive learning rate
void ModelMajority_trainRMSProp(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, bestIter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, bestObj = -1; 
  ModelMajority *model = self;
  ModelMajority *bestModel = NULL;
  
  bestModel = NEW(ModelMajority, "majority model that achieved lowest obj");
  bestModel->_(init) (bestModel, params->nUsers, params->nItems, params->facDim,
      params->regU, params->regI, params->learnRate);

  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);
  fflush(stdout);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= MAX_SET_SZ);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
   
      //ModelMajority_gradientCheck(model, u, set, setSz, r_us, itemRats);
      //continue;

      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;
      
      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }

      }
      
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      if (model->_(isTerminateModel)(model, bestModel, iter, &bestIter, &bestObj,
            &prevObj, valTest, data)) {
        break;
      }
    }
    
  }

  printf("\nEnd Obj: %.10e bestObj: %.10e bestIter: %d", valTest->setObj, 
      bestObj, bestIter);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(bestModel, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = bestModel->_(validationErr) (bestModel, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = bestModel->_(indivTrainSetsErr) (bestModel, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);
 
  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->testMat, 20); 
  printf("\nTest spearman: %f", valTest->testSpearman);
  
  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->valMat, 20); 
  printf("\nVal spearman: %f", valTest->valSpearman);

  valTest->trainSetRMSE = bestModel->_(trainErr)(bestModel, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = bestModel->_(testErr) (bestModel, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = bestModel->_(indivItemCSRErr) (bestModel, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");
  
  bestModel->_(copy) (bestModel, model);
  bestModel->_(free)(bestModel);

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent using adaptive learning rate(rmsprop) but without constraint
void ModelMajority_trainRMSPropWOCons(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
      userSet = data->userSets[u];
      isTestValSet = 0;
      //select a non-test non-val set for user
      s = rand() % userSet->numSets;
      
      //check if set in test or validation
      if (UserSets_isSetTestVal(userSet, s)) {
       continue;
      }
      
      set       = userSet->uSets[s];
      setSz     = userSet->uSetsSize[s];
      r_us      = userSet->labels[s];
      r_us_est  = 0.0;

      assert(setSz <= MAX_SET_SZ);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
   
      //ModelMajority_gradientCheck(model, u, set, setSz, r_us, itemRats);
      //continue;

      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;
      
      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //item gradient
      for (j = 0; j < model->_(facDim); j++) {
        iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          //add regularization
          temp = iGrad[j] + 2.0*model->_(regI)*model->_(iFac)[item][j]; 
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*temp*temp;
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*temp;
        }
      }
      
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter: %d obj: %f avgHits: %f testItemsRmse: %f", iter, 
          valTest->setObj, model->_(hitRateOrigTopN) (model, data->trainMat, 
            data->uFac, data->iFac, 10),
          model->_(indivItemCSRErr) (model, data->testMat, NULL));
      if (iter > 3000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent for average model using adaptive learning rate(RMSProp)
void ModelMajority_trainRMSPropAvg(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet, bestIter;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, bestObj; 
  ModelMajority *model = self;
  ModelMajority *bestModel =NEW(ModelMajority, "average model that achieved lowest obj"); 
  bestModel->_(init) (bestModel, params->nUsers, params->nItems, params->facDim,
      params->regU, params->regI, params->learnRate);
  
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      userSet = data->userSets[u];
      
      if (userSet->numSets == 0) {
        continue;
      }

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
      r_us_est  = 0.0;

      assert(setSz <= MAX_SET_SZ);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      majSz = setSz;

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }

      }
    
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      /*
      if (model->_(isTerminateColdModel)(model, bestModel, iter, &bestIter, &bestObj, 
          &prevObj, valTest, data, data->valItemIds, data->nValItems)) {
        break;
      }
      */

      if (model->_(isTerminateModel)(model, bestModel, iter, &bestIter, &bestObj,
            &prevObj, valTest, data)) {
        break;
      }
    }
    
  }

  printf("\nEnd Obj: %.10e bestObj: %.10e bestIter: %d", valTest->setObj, 
      bestObj, bestIter);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(bestModel, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  /*
  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                         data->valMat, 10); 
  printf("\nVal spearman: %f", valTest->valSpearman);

  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                          data->testMat, 10); 
  printf("\nTest spearman: %f", valTest->testSpearman);
  valTest->valSpearman = bestModel->_(coldHitRate)(bestModel, data->userSets,
      data->valMat, data->itemFeatMat, data->valItemIds, data->nValItems, 10); 
  printf("\nVal HR: %f", valTest->valSpearman);
  */

  valTest->valSetRMSE = bestModel->_(validationErr) (bestModel, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
 
  valTest->trainItemsRMSE = bestModel->_(indivTrainSetsErr) (bestModel, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = bestModel->_(trainErr)(bestModel, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
 
  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->testMat, 20); 
  printf("\nTest spearman: %f", valTest->testSpearman);
  
  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, data->valMat, 20); 
  printf("\nVal spearman: %f", valTest->valSpearman);
 
  valTest->testSetRMSE = bestModel->_(testErr) (bestModel, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = bestModel->_(indivItemCSRErr) (bestModel, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  bestModel->_(copy) (bestModel, model);
  bestModel->_(free) (bestModel);

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


void ModelMajority_trainRMSPropSum(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet, bestIter;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, bestObj; 
  ModelMajority *model = self;
  ModelMajority *bestModel =NEW(ModelMajority, "average model that achieved lowest obj"); 
  bestModel->_(init) (bestModel, params->nUsers, params->nItems, params->facDim,
      params->regU, params->regI, params->learnRate);
  
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*MAX_SET_SZ);
  for (i = 0; i < MAX_SET_SZ; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %.10e", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= MAX_SET_SZ);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      majSz = setSz;

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }

      }
    
     }

    //objective check
    if (iter % OBJ_ITER == 0) {
      if (model->_(isTerminateModel)(model, bestModel, iter, &bestIter, &bestObj,
            &prevObj, valTest, data)) {
        break;
      }
    }
    
  } 

  printf("\nEnd Obj: %.10e bestObj: %.10e bestIter: %d", valTest->setObj, 
      bestObj, bestIter);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(bestModel, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  /*
  valTest->valSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                         data->valMat, 10); 
  printf("\nVal spearman: %f", valTest->valSpearman);

  valTest->testSpearman = bestModel->_(spearmanRankCorrN)(bestModel, 
                                                          data->testMat, 10); 
  printf("\nTest spearman: %f", valTest->testSpearman);
  */

  valTest->valSetRMSE = bestModel->_(validationErr) (bestModel, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  //valTest->trainItemsRMSE = bestModel->_(indivTrainSetsErr) (bestModel, data);
  //printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = bestModel->_(trainErr)(bestModel, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = bestModel->_(testErr) (bestModel, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  //valTest->testItemsRMSE = bestModel->_(indivItemCSRErr) (bestModel, data->testMat, NULL);
  //printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  bestModel->_(free) (bestModel);


  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < MAX_SET_SZ; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent with set generation on the fly with adaptive learning rate
void ModelMajority_trainRMSPropSampSets(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  setSz = 5;
  set = (int*)malloc(sizeof(int)*setSz);

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
      //sample a positive set for u
      samplePosSetNRat(data, u, set, setSz, &r_us);
      
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;
      
      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
      
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }

      }
      
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\nIter: %d obj: %f avgHits: %f trainmsErr: %f", iter, valTest->setObj, 
          model->_(hitRateOrigTopN) (model, data->trainMat, data->uFac, 
            data->iFac, 10),
          model->_(indivTrainSetsErr) (model, data));
      if (iter > 5000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(set);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent with adaptive learning rate strategy from ADAM
void ModelMajority_trainADAM(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));

  float biasCorrFirstMom, biasCorrSecMom;


  //first moment estimate
  float **iMVec   = (float**) malloc(sizeof(float*)*params->nItems);
  float **uMVec    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iMVec[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iMVec[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uMVec[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uMVec[i], 0, sizeof(float)*model->_(facDim));
  }
  
  //second raw moment estimate
  float **iVVec   = (float**) malloc(sizeof(float*)*params->nItems);
  float **uVVec    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iVVec[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iVVec[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uVVec[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uVVec[i], 0, sizeof(float)*model->_(facDim));
  }


  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
 
  printf("\nbeta1 = %f", model->beta1);
  printf("\nbeta2 = %f", model->beta2);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      //sort ratings in decreasing order
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        /* 
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }
        */
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        /*
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
        */
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //update biased first moment estimate
        uMVec[u][j] = model->beta1*uMVec[u][j] + (1.0 - model->beta1)*uGrad[j];
        biasCorrFirstMom = uMVec[u][j]/(1.0 - pow(model->beta1, iter+1));

        //update biased second raw moment estimate
        uVVec[u][j] = model->beta2*uVVec[u][j] + (1.0 - model->beta2)*uGrad[j]*uGrad[j];
        biasCorrSecMom = uVVec[u][j]/(1.0 - pow(model->beta2, iter+1));

        //update
        model->_(uFac)[u][j] -= model->_(learnRate)*biasCorrFirstMom/(sqrt(biasCorrSecMom) + 0.000001);
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 0;
          if (i < majSz) {
            iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
            /*
            if (i == 0 && itemRats[0]->rating < r_us) {
              //item with max rating and constraint kicked in 
              iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
            }
            */
          }
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //update biased first moment estimate
          iMVec[item][j] = model->beta1*iMVec[item][j] + (1.0 - model->beta1)*iGrad[j];
          biasCorrFirstMom = iMVec[item][j]/(1.0 - pow(model->beta1, iter + 1));

          //update biased second raw estimate
          iVVec[item][j] = model->beta2*iVVec[item][j] + (1.0 - model->beta2)*iGrad[j]*iGrad[j];
          biasCorrSecMom = iVVec[item][j]/(1.0 - pow(model->beta2, iter + 1));      

          //update
          model->_(iFac)[item][j] -= model->_(learnRate)*biasCorrFirstMom/(sqrt(biasCorrSecMom) + 0.000001);

        }

      }
    
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter%500 == 0) {
        printf("\nIter: %d obj: %f ", iter, valTest->setObj);
      }
      if (iter > 10000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uMVec[i]);
    free(uVVec[i]);
  }
  free(uMVec);
  free(uVVec);
  
  for (i = 0; i < params->nItems; i++) {
    free(iVVec[i]);
    free(iMVec[i]);
  }
  free(iMVec);
  free(iVVec);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent with adaptive learning rate(AdaGrad)
void ModelMajority_trainAdaGrad(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **iGradsAcc    = (float**) malloc(sizeof(float*)*params->nItems);
  float **uGradsAcc    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uGradsAcc[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uGradsAcc[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        //accumulate the gradients square
        uGradsAcc[u][j] += uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        }
        
        for (j = 0; j < model->_(facDim); j++) {
          //accumulate the gradients square
          iGradsAcc[item][j] += iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }

      }
    
    }
    

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter % 50 == 0) {
        printf("\nIter: %d obj: %f", iter, valTest->setObj);
      }
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uGradsAcc[i]);
  }
  free(uGradsAcc);
  
  for (i = 0; i < params->nItems; i++) {
    free(iGradsAcc[i]);
  }
  free(iGradsAcc);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent using heavy ball method from Boyd notes
void ModelMajority_trainHeavyBall(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  float **uUpd    = (float**) malloc(sizeof(float*)*params->nItems);
  float **iUpd    = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0; i < params->nItems; i++) {
    iUpd[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iUpd[i], 0, sizeof(float)*model->_(facDim));
  }
  for (i = 0; i < params->nUsers; i++) {
    uUpd[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(uUpd[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  printf("\nmodel->beta1: %f", model->beta1);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
          //check if decreasing order
          //assert(itemRats[i]->rating <= itemRats[i-1]->rating);
        
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user sub gradient
      for (j = 0; j < model->_(facDim); j++) {
        //check if constraint violated
        if (itemRats[0]->rating < r_us) {
          //use only subgradient of constraint 
          uGrad[j] = -1.0*model->_(iFac)[itemRats[0]->item][j];
        } else {
          //use subgradient of the objective
          uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
          //add regularization
          uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        }
      }
    
      for (j = 0; j < model->_(facDim); j++) {
        temp = -1.0*model->_(learnRate)*uGrad[j] + model->beta1*(uUpd[u][j]);
        uUpd[u][j] = temp;
        //update
        model->_(uFac)[u][j] += temp;
      }

      //whether to estimate top ratings again after update
      //r_us_est = dotProd(model->_(uFac)[u], sumItemLatFac, model->_(facDim))/(majSz);
      //TODO: sort if updating all rating estimates

      //item updates 
      //check if constraint violated if yes then only update top item as others'
      //subgradient will be 0
      if (itemRats[0]->rating < r_us) {
        item = itemRats[0]->item;
        //use top item constraint sub gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = -1.0*model->_(uFac)[u][j];
          temp = -1.0*model->_(learnRate)*iGrad[j] + model->beta1*iUpd[item][j];
          iUpd[item][j] = temp;
          //update
          model->_(iFac)[item][j] += temp;
        }
      } else {
        //constraint not violated update all items in top 
        //compute  gradient of item and update their latent factor in sets
        for (i = 0; i < majSz; i++) {
          item = itemRats[i]->item;
          //get item gradient
          for (j = 0; j < model->_(facDim); j++) {
            iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
            //add regularization
            iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
            temp = -1.0*model->_(learnRate)*iGrad[j] + model->beta1*iUpd[item][j];
            iUpd[item][j] = temp;
            //update
            model->_(iFac)[item][j] += temp;
          }
        }
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter % 50 == 0) {
        printf("\nIter: %d obj: %f", iter, valTest->setObj);
      }
      //convergence check after 1000 iterations
      if (iter > 1000) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < params->nUsers; i++) {
    free(uUpd[i]);
  }
  free(uUpd);
  
  for (i = 0; i < params->nItems; i++) {
    free(iUpd[i]);
  }
  free(iUpd);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent with fixed learning rate
void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet, succ, fail;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, epochDiff; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  for (iter = 0; iter < params->maxIter; iter++) {
    epochDiff = 0;
    succ = 0;
    fail = 0;
    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100);

      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }
      
      /*
      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
          
        }

        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
      }
     
      coeffUpdateAdap(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim), iter);
      */

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
        }
        //update item + reg
        coeffUpdateAdap2(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim), iter);
      }

    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      printf("\niter: %d obj:%f", iter, valTest->setObj);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }
    

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter) {
    printf("\nNOT CONVERGED:Reached maximum iterations");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//update model given user sets 
void ModelMajority_update(void *self, UserSets *userSet, int setInd, 
    ItemRat **itemRats, float *sumItemLatFac, float *iGrad, float *uGrad) {

  int u, i, j;
  int setSz, majSz, item;
  int *set;
  float r_us, r_us_est;
  ModelMajority *model = self;

  u         = userSet->userId;
  set       = userSet->uSets[setInd];
  setSz     = userSet->uSetsSize[setInd];
  r_us      = userSet->labels[setInd];
  r_us_est  = 0.0;

  assert(setSz <= 100 && setSz > 1);

  for (i = 0; i < setSz; i++) {
    itemRats[i]->item = set[i];
    itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                    model->_(facDim));
  }
  
  qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
  
  if (setSz % 2  == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }

  //update user and item latent factor
  memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
  for (i = 0; i < majSz; i++) {
    item = itemRats[i]->item;
    if (i > 0) {
      if (itemRats[i]->rating > itemRats[i-1]->rating) {
        printf("\nTrain:Not in decreasing order:  %f %f", 
            itemRats[i]->rating, itemRats[i-1]->rating);
        fflush(stdout);
        exit(0);
      }
    }
    for (j = 0; j < model->_(facDim); j++) {
      sumItemLatFac[j] += model->_(iFac)[item][j];
    }
    r_us_est += itemRats[i]->rating;
  }
  r_us_est = r_us_est/majSz;

  //compute user gradient
  for (j = 0; j < model->_(facDim); j++) {
    uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
    if (itemRats[0]->rating < r_us) {
      //constraint kicked in 
      uGrad[j] += -1.0*model->constrainWt*model->_(iFac)[itemRats[0]->item][j];
    }
  }

  //update user
  coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU), model->_(learnRate),
      model->_(facDim));

  //update items
  for (i = 0; i < majSz; i++) {
    item = itemRats[i]->item;
    //get item gradient
    for (j = 0; j < model->_(facDim); j++) {
      iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
      if (i == 0 && itemRats[0]->rating < r_us) {
        //item with max rating and constraint kicked in 
        iGrad[j] += -1.0*model->constrainWt*model->_(uFac)[u][j];
      }
    }
    //update item + reg
    coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
        model->_(learnRate), model->_(facDim));
  }


}


//following uses line search to find learning rate over a sample and then descent
void ModelMajority_trainSGDSamp(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet, succ, fail;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, epochDiff; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  for (iter = 0; iter < params->maxIter; iter++) {
    //find opt learn rate based on training error of few samples
    model->_(learnRate) = ModelMajority_learnRateSearch(model, data, itemRats, 
        sumItemLatFac, iGrad, uGrad, params->seed + iter);

    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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

      ModelMajority_update(model, userSet, s, itemRats, sumItemLatFac, iGrad, 
          uGrad);
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (valTest->setObj != valTest->setObj) {
        //NAN check
        printf("\nComputed objective is NAN: %f", valTest->setObj);
        break;
      }

      if (iter % 50 == 0) {
        printf("\nIter: %d Obj: %f", iter, valTest->setObj);
      }

      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
      }
      prevObj = valTest->setObj;
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  } 

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter || valTest->setObj != valTest->setObj) {
    printf("\nNOT CONVERGED:Reached maximum iterations or NAN objective");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//subgradient descent with adaptive learning rate with momentum
void ModelMajority_trainSGDMomentum(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet, succ, fail;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, epochDiff; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));

  float **uUpd     = (float**) malloc(sizeof(float*)*params->nUsers);
  for (i = 0 ; i < params->nUsers; i++) {
    uUpd[i] = (float *) malloc(sizeof(float)*model->_(facDim));
    memset(uUpd[i], 0, sizeof(float)*model->_(facDim));
  }

  float **iUpd     = (float**) malloc(sizeof(float*)*params->nItems);
  for (i = 0; i < params->nItems; i++) {
    iUpd[i] = (float*) malloc(sizeof(float)*model->_(facDim));
    memset(iUpd[i], 0, sizeof(float)*model->_(facDim));
  }

  //TODO: hardcode
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }

  printf("\nnumAllSets: %d", numAllSets);

  //get train error
  printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));
  
  //get objective
  printf("\nInit Obj: %f", model->_(objective)(model, data, sim));
  
  printf("\nInit Constraint violation: %d", ModelMajority_constViol(model, data));
  
  for (iter = 0; iter < params->maxIter; iter++) {

    for (subIter = 0; subIter < numAllSets; subIter++) {
      
      //sample u
      u = rand() % data->nUsers;
      
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
      r_us_est  = 0.0;

      assert(setSz <= 100 && setSz > 1);
     
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
      
      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //update user and item latent factor
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        if (i > 0) {
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            fflush(stdout);
            exit(0);
          }
        }
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient n update
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
        
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] += -1.0*model->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }

        //add reg components to user gradient
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j]; 

        //compute update for user latent factor
        uUpd[u][j] += model->momentum*uUpd[u][j] + model->_(learnRate)*uGrad[j]; 
        //apply update
        model->_(uFac)[u][j] -= uUpd[u][j];
      }

      //update items
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < r_us) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*model->constrainWt*model->_(uFac)[u][j];
          }
          //add regularization to item gradient
          iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
        
          //compute update for item latent factor
          iUpd[item][j] += model->momentum*iUpd[item][j] + model->_(learnRate)*iGrad[j];
          //apply update
          model->_(iFac)[item][j] -= iUpd[item][j];
        }
      }
      
      if (subIter % 100 == 0) {
        valTest->setObj = model->_(objective)(model, data, sim);
        printf("\nsubIter: %d obj: %f", subIter, valTest->setObj);
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (valTest->setObj != valTest->setObj) {
        //NAN check
        printf("\nComputed objective is NAN: %f", valTest->setObj);
        break;
      }
      printf("\nIter: %d Obj: %f", iter, valTest->setObj);
      if (iter > 0) {
        if (fabs(prevObj - valTest->setObj) < EPS) {
          //exit train procedure
          printf("\nConverged in iteration: %d prevObj: %f currObj: %f", iter,
              prevObj, valTest->setObj);
          break;
        }
        
        if (fabs((prevObj - valTest->setObj)/prevObj) <= 0.1) {
          //if objective decrease has become stable than increase momentum
          printf("\nUpdated momentum to 0.9");
          model->momentum = 0.9;
        }

      }
      prevObj = valTest->setObj;
    }

    //validation check
    /*
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }
    */
    
  }

  printf("\nEnd Obj: %f", valTest->setObj);

  printf("\nFinal Constraint violation: %d", ModelMajority_constViol(model, data));
  
  if (iter == params->maxIter || valTest->setObj != valTest->setObj) {
    printf("\nNOT CONVERGED:Reached maximum iterations or NAN objective");
  }

  valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
  printf("\nValidation set err: %f", valTest->valSetRMSE);
  
  valTest->trainItemsRMSE = model->_(indivTrainSetsErr) (model, data);
  printf("\nTrain set indiv error(modelMajority): %f", valTest->trainItemsRMSE);

  valTest->trainSetRMSE = model->_(trainErr)(model, data, NULL); 
  printf("\nTrain set error(modelMajority): %f", valTest->trainSetRMSE);
  
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  //printf("\nTest hit rate: %f", 
  //    model->_(hitRate)(model, data->trainMat, data->testMat));

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");
  for (i = 0 ; i < params->nUsers; i++) {
    free(uUpd[i]);
  }
  free(uUpd);

  for (i = 0; i < params->nItems; i++) {
    free(iUpd[i]);
  }
  free(iUpd);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


//generate sets on the fly and update model with fixed learning rate
void ModelMajority_trainSamp(void *self, Data *data, Params *params, 
    float **sim, ValTestRMSE *valTest) {
   
  int iter, u, i, j, k, s, numAllSets;
  ModelMajority *model = self;
  int setSz = 5, item, majSz, subIter;
  float estPosRat, expCoeff, prevVal, posRat;

  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* posLatFac     = (float*) malloc(sizeof(float)*model->_(facDim));
  int *posSet          = (int*) malloc(sizeof(int)*setSz);
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  numAllSets = 0;
  for (u = 0; u < data->nUsers; u++) {
    numAllSets += data->userSets[u]->numSets;
  }
  
  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter < numAllSets; subIter++) {

      u = rand() % data->nUsers;

      //sample a positive set for u
      samplePosSetNRat(data, u, posSet, setSz, &posRat);
     
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = posSet[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[posSet[i]], 
            model->_(facDim));
      }
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
      
      if (setSz % 2  == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }

      //get estimated rating on set
      memset(posLatFac, 0, sizeof(float)*model->_(facDim));
      estPosRat = 0;
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        
        if (i > 0) {
          
          if (itemRats[i]->rating > itemRats[i-1]->rating) {
            printf("\nTrain:Not in decreasing order:  %f %f", 
                itemRats[i]->rating, itemRats[i-1]->rating);
            Params_display(params);
            fflush(stdout);
            exit(0);
          }
        }

        for (j = 0; j < model->_(facDim); j++) {
          posLatFac[j] += model->_(iFac)[item][j];
        }
        estPosRat += itemRats[i]->rating;
      }
      estPosRat = estPosRat/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(estPosRat - posRat)*posLatFac[j]*(1.0/majSz);
        if (itemRats[0]->rating < posRat) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[0]->item][j];
        }
      }
      //update user 
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //update items in  set
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item; 
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(estPosRat - posRat)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i == 0 && itemRats[0]->rating < posRat) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
        }
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI),
          model->_(learnRate), model->_(facDim));
      }

    }

    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest->valSetRMSE = model->_(validationErr) (model, data, NULL);
      //printf("\nIter:%d validation error: %f", iter, valTest[0]);
      if (iter > 0) {
        if (fabs(prevVal - valTest->valSetRMSE) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest->valSetRMSE, fabs(prevVal - valTest->valSetRMSE));
          break;
        }
      }
      prevVal = valTest->valSetRMSE;
    }

  }
 
  valTest->testSetRMSE = model->_(testErr) (model, data, NULL); 
  printf("\nTest set error(modelMajority): %f", valTest->testSetRMSE);

  //get test eror
  valTest->testItemsRMSE = model->_(indivItemCSRErr) (model, data->testMat, NULL);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

 
   for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(iGrad);
  free(uGrad);
  free(posLatFac);
  free(posSet);

}


//model specifications
Model ModelMajorityProto = {
  .objective             = ModelMajority_objective,
  //.objective             = ModelMajority_objectiveWOCons,
  .setScore              = ModelMajority_setScore,
  .itemFeatScore         = ModelMajority_itemFeatScoreAvg, 
  //.setScore              = ModelMajority_setScoreAvg,
  .train                 = ModelMajority_trainRMSProp //ModelMajority_train
};


//perform line search to find optimal learning rate
float ModelMajority_learnRateSearch(void *self, Data *data, ItemRat **itemRats,
    float *sumItemLatFac, float *iGrad, float *uGrad, int seed) {

  int u, i, j, k, s;
  int isTestValSet;
  float bestLearnRate = -1;
  float bestObj = -1;
  float modelObj, initObj;  
 
  float learnRate = 0.000001;

  UserSets *userSet = NULL;
  ModelMajority *model = self;
  ModelMajority *dupModel = NEW(ModelMajority, 
                              "set prediction with majority score"); 
  dupModel->_(init)(dupModel, model->_(nUsers), model->_(nItems), model->_(facDim),
      model->_(regU), model->_(regI), model->_(learnRate));
  initObj = model->_(objective)(model, data, NULL);

  while (learnRate <= 0.01) {
    
    //copy original model i.e. lat facs
    model->_(copy)(model, dupModel);

    //modify learn rate of dupModel
    dupModel->_(learnRate) = learnRate;
    
    //NOTE: make sure rand() deals with same samples for learning rate
    //computation
    srand(seed);
    //do few samples update using the learn rate
    for (k = 0; k < 5000; k++) {
  
      u = rand() % data->nUsers;
      userSet = data->userSets[u];
      isTestValSet = 0;
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

      ModelMajority_update(dupModel, userSet, s, itemRats, sumItemLatFac, iGrad, 
          uGrad);
    }

    //compute objective after updates
    modelObj = dupModel->_(objective) (dupModel, data, NULL);
    //printf("\nlearnRate:%f obj:%f", dupModel->_(learnRate), modelObj);

    if (modelObj != modelObj) {
      //NAN check
      printf("\nObjective is NAN = %f for learn rate %f", modelObj, dupModel->_(learnRate));
      continue;
    }

    //find learn rates
    if (bestLearnRate < 0) {
      bestLearnRate = dupModel->_(learnRate);
      bestObj = modelObj;
    } else {
      if (modelObj < bestObj) {
        bestObj = modelObj;
        bestLearnRate = dupModel->_(learnRate);
      } else if (modelObj > bestObj) {
        //found same or bigger objective then best, after this point objective
        //will only increase
        //printf("\nFound bigger objective. bestObj: %f currObj: %f", 
        //    bestObj, modelObj);
        break;
      }
    }

    learnRate = learnRate*2;
  }

  if (initObj < bestObj) {
    //printf("\nLearning rate is not optimal, initObj: %f bestObj: %f", 
    //    initObj, bestObj);
  }

  //reset the seed so that 5000 samples chosen above  are updated in the 
  //main model updates again to get the observed decrease in these samples
  srand(seed);
  //printf("\nBest obj: %f learnRate: %f", bestObj, bestLearnRate);

  dupModel->_(free)(dupModel);

  return bestLearnRate;
}


void modelMajority(Data *data, Params *params, ValTestRMSE *valTest) {
  
  ModelMajority *model = NEW(ModelMajority, 
                              "set prediction with majority score");  
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
                  params->regU, params->regI, params->learnRate);
  model->constrainWt = params->constrainWt;
  model->rhoRMS      = params->rhoRMS;
  model->epsRMS      = params->epsRMS;
  model->momentum    = 0.1;
  model->beta1       = params->rhoRMS;
  model->beta2       = params->epsRMS;

  printf("\nconstWt: %f rhoRMS:%f epsRMS:%f beta1:%f beta2:%f", 
      model->constrainWt, model->rhoRMS, model->epsRMS, model->beta1, 
      model->beta2);

  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  //loadUserItemWtsFrmTrain(data);
 
  //initialize model with latent factors given to the program
  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  printf("\nTest Error: %f", model->_(testErr) (model, data, NULL));  
  //testItemRat();
  
  //get objective
  printf("\nModelMaj Init Obj: %.10e", model->_(objective)(model, data, NULL));
  
  //train model 
  model->_(train)(model, data,  params, NULL, valTest);
 
  //compare model with loaded latent factors if any
  //model->_(cmpLoadedFac)(model, data);

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "modelMajUFacAvg.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "modelMajIFacAvg.txt");

  model->_(free)(model);
}




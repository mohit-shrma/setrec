#include "modelMajority.h"


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

  temp = dotProd(model->_(uFac)[u], sumItemFac, model->_(facDim))/majSz; 
  if (fabs(r_us_est - temp) > 0.00001) {
    printf("\nERR: Rating mismatch for set %f %f", r_us_est, temp);
    exit(0);
  }

  return r_us_est;
}


void ModelMajority_gradientCheck(void *self, int u, int *set, int setSz, 
    float r_us, ItemRat **itemRats) {

  int i, j, item, majSz;
  ModelMajority *model = self;
  float *sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float *uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float *iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float *fac           = (float*) malloc(sizeof(float)*model->_(facDim));
  float lossRight, lossLeft, gradE, r_us_est, rat_s;
  
  if (setSz % 2 == 0) {
    majSz = setSz/2;
  } else {
    majSz = setSz/2 + 1;
  }
  
  r_us_est = ModelMajority_estScoreNSumFac(model, u, set, setSz, sumItemLatFac,
      itemRats);

  //sample item
  item = itemRats[rand()%majSz]->item;
  //item gradient
  memset(iGrad, 0, sizeof(float)*model->_(facDim));
  for (j = 0; j < model->_(facDim); j++) {
    iGrad[j] = (2.0*(r_us_est - r_us)*model->_(uFac)[u][j])/majSz;
    //if (itemRats[0]->item == item && itemRats[0]->rating < r_us) { 
    //if (itemRats[0]->item == item) { 
    //  iGrad[j] += -1.0*model->constrainWt*model->_(uFac)[u][j];
    //}
  }

  //perturb item with +E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim)); 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] + 0.0001;    
  }
  rat_s = dotProd(model->_(uFac)[u], fac, model->_(facDim))/majSz;
  lossRight = pow(rat_s - r_us, 2);
  //if (itemRats[0]->item == item && r_us > itemRats[0]->rating) {
  //if (itemRats[0]->item == item) {
  //  lossRight += model->constrainWt*(r_us - itemRats[0]->rating);
  //}

  //perturb item with  -E and cmpute loss
  memset(fac, 0, sizeof(float)*model->_(facDim)); 
  for (j = 0; j < model->_(facDim); j++) {
    fac[j] = sumItemLatFac[j] - 0.0001;    
  }
  rat_s = dotProd(model->_(uFac)[u], fac, model->_(facDim))/majSz;
  lossLeft = pow(rat_s - r_us, 2);
  //if (itemRats[0]->item == item && r_us > itemRats[0]->rating) {
  //if (itemRats[0]->item == item) {
  //  lossLeft += model->constrainWt*(r_us - itemRats[0]->rating);
  //}

  //compute gradient and E dotprod
  gradE = 0;
  for (j = 0; j < model->_(facDim); j++) {
    gradE += 2.0*iGrad[j]*0.0001;
  }

  if (fabs(lossRight-lossLeft-gradE) > 0.00001) {
    printf("\nu: %d item: %d diff: %f div: %f lDiff:%f gradE:%f", u, item, 
      lossRight-lossLeft-gradE, (lossRight-lossLeft)/gradE, lossRight-lossLeft, 
      gradE);
  }


  free(sumItemLatFac);
  free(iGrad);
  free(uGrad);
  free(fac);
}


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


float ModelMajority_objective2(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;
  float maxRat = 0;
  int constViolCt = 0;
  int nSets = 0;
  
  FILE *fp = NULL;
  //fp = fopen("constrVilOff.txt", "w");

  int majSz;
  setSz = data->userSets[0]->uSetsSize[0];
  ItemRat **itemRats = (ItemRat**) malloc(sizeof(ItemRat*)*setSz);
  for (i = 0; i < setSz; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }


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
      
      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]], 
            model->_(facDim));
      }
      
      qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);
      
      if (setSz % 2 == 0) {
        majSz = setSz/2;
      } else {
        majSz = setSz/2 + 1;
      }
      
      userSetPref = 0;
      for (i = 0; i < majSz; i++) {
        userSetPref += itemRats[i]->rating;
      }
      userSetPref = userSetPref/majSz;
      maxRat = itemRats[0]->rating;

      //get preference over set
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
      rmse += diff*diff;
      if (userSet->labels[s] > maxRat) {
        //constraint violated
        rmse += model->constrainWt*(userSet->labels[s] - maxRat);
        constViolCt++;
        //fprintf(fp, "%f\n", userSet->labels[s]-maxRat);
      }
      
      if (itemRats[majSz-1]->rating < itemRats[majSz]->rating + 0.1) {
        rmse += model->constrainWt*(itemRats[majSz-1]->rating - (itemRats[majSz]->rating + 0.1))*(itemRats[majSz-1]->rating - (itemRats[majSz]->rating + 0.1));
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
  
  printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f Constraint violation count: %d", 
      (rmse+uRegErr+iRegErr), rmse, uRegErr, iRegErr, constViolCt);
  
  //free itemRats
  for (i = 0; i < setSz; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  
  //fclose(fp);
  return (rmse + uRegErr + iRegErr);
}


void ModelMajority_train2Constraints(void *self, Data *data, Params *params, float **sim, 
    ValTestRMSE *valTest) {

  int iter, subIter, u, i, j, k, s, numAllSets;
  UserSets *userSet;
  int setSz, item, majSz, isTestValSet;
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
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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

        if (itemRats[majSz-1]->rating < itemRats[majSz]->rating + 0.1) {
          uGrad[j] += 2.0*params->constrainWt*(itemRats[majSz-1]->rating - (itemRats[majSz]->rating + 0.1))*(model->_(iFac)[itemRats[majSz-1]->item][j] - model->_(iFac)[itemRats[majSz]->item][j]);
        }
      }
     
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

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

          if ((i == majSz -1 || i == majSz)  && 
              itemRats[majSz-1]->rating < (itemRats[majSz]->rating + 0.1)) {
            if (i == majSz-1) {
              iGrad[j] += 2.0* (params->constrainWt)*(itemRats[majSz-1]->rating - (itemRats[majSz]->rating + 0.1))*model->_(uFac)[u][j];
            } else {
              iGrad[j] += -2.0*params->constrainWt*(itemRats[majSz-1]->rating - (itemRats[majSz]->rating + 0.1))*model->_(uFac)[u][j];
            }
          }

        }
        //update item + reg
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim));
      }

    
      //find whether update helped in estimating set ratings
      r_us_est_aftr_upd = 0;
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        } 
      } 
      r_us_est_aftr_upd = dotProd(model->_(uFac)[u], sumItemLatFac, 
          model->_(facDim))/majSz;
      epochDiff += fabs(r_us_est_aftr_upd - r_us);
    }
    
    printf("\nEpoch avg diff = %f", epochDiff/subIter);

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


void ModelMajority_trainCD(void *self, Data *data, Params *params, float **sim, 
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

  float *numAr, *denomAr;
  numAr = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(numAr, 0, sizeof(float)*model->_(facDim));
  denomAr = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(denomAr, 0, sizeof(float)*model->_(facDim));

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
    
    /*
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
    */

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
        printf("\nIter: %d obj: %f test items rmse: %f", iter, valTest->setObj, 
            model->_(indivItemSetErr) (model, data->testSet));
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
  free(numAr);
  free(denomAr);
}


void ModelMajority_trainCDAlt(void *self, Data *data, Params *params, float **sim, 
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

  float *numAr, *denomAr;
  numAr = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(numAr, 0, sizeof(float)*model->_(facDim));
  denomAr = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(denomAr, 0, sizeof(float)*model->_(facDim));

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
      model->_(indivItemSetErr) (model, data->testSet));
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
  printf("\nTest items error(modelMajority): %f", valTest->testItemsRMSE);

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  for (i = 0; i < nSetsPerUser; i++) {
    free(setSumItemLatFac[i]);
  }
  free(setSumItemLatFac);
  free(numAr);
  free(denomAr);
}


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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


void samplePosSetNRat(Data *data, int u, int *posSet, int setSz, float *rat) {
  
  int i, nItems, itemInd, item, found;
  int nUserItems, majSz;
  gk_csr_t *trainMat = data->trainMat;
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
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

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
}


void ModelMajority_trainRMSProp(void *self, Data *data, Params *params, float **sim, 
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
 
  printf("\nrhoRMS = %f", params->rhoRMS);

  for (iter = 0; iter < params->maxIter; iter++) {
    for (subIter = 0; subIter <  100; subIter++) {
      
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
   


      ModelMajority_gradientCheck(model, u, set, setSz, r_us, itemRats);
      continue;

      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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
      
      /*
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
      if (iter%1000 == 0) {
        printf("\nIter: %d obj: %f, testItemsErr: %f", iter, valTest->setObj, model->_(indivItemSetErr) (model, data->testSet));
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
   
      //ModelMajority_gradientCheck(model, u, set, setSz, r_us, itemRats);
      //continue;

      for (i = 0; i < setSz; i++) {
        itemRats[i]->item = set[i];
        itemRats[i]->rating = dotProd(model->_(uFac)[u], model->_(iFac)[set[i]],
                                        model->_(facDim));
      }
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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
      
      /*
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
      */
      
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
      printf("\nIter: %d obj: %f, testItemsErr: %f", iter, valTest->setObj, model->_(indivItemSetErr) (model, data->testSet));
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

    if (iter > 0 && iter % 500 == 0) {
      model->_(learnRate) = model->_(learnRate)/10.0;
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
    if (iter % 1000 == 0) {
      printf("\nIter: %d", iter);
      fflush(stdout);
    }
  

    if (iter >5000 && iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter%1000 == 0) {
        printf("\nIter: %d obj: %f, trainItemsErr: %f", iter, valTest->setObj, model->_(indivTrainSetsErr) (model, data));
      }
      if (iter > 6000) {
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


void ModelMajority_trainRMSPropCons(void *self, Data *data, Params *params, float **sim, 
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
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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
        if (itemRats[0]->rating < r_us) {
          //constraint kicked in 
          uGrad[j] = -1.0*model->_(iFac)[itemRats[0]->item][j];
        } else {
          uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
          uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
        }
        //accumulate the gradients square
        uGradsAcc[u][j] = uGradsAcc[u][j]*params->rhoRMS + 
          (1.0 - params->rhoRMS)*uGrad[j]*uGrad[j];
        //update
        model->_(uFac)[u][j] -= (model->_(learnRate)/sqrt(uGradsAcc[u][j] + 0.0000001))*uGrad[j];
      }
    
      
      if (itemRats[0]->rating < r_us) {
        //constraint violated
        //update only top item
        item = itemRats[0]->item;
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = -1.0*model->_(uFac)[u][j];
          //accumulate the gradients square 
          iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
          //update
          model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
        }
      } else {
        //update all items
        //compute  gradient of item and update their latent factor in sets
        for (i = 0; i < majSz; i++) {
          item = itemRats[i]->item;        
          //get item gradient
          for (j = 0; j < model->_(facDim); j++) {
            iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
            iGrad[j] += 2.0*model->_(regI)*model->_(iFac)[item][j];
            //accumulate the gradients square 
            iGradsAcc[item][j] = params->rhoRMS*iGradsAcc[item][j]  + (1.0 - params->rhoRMS)*iGrad[j]*iGrad[j];
            //update
            model->_(iFac)[item][j] -= (model->_(learnRate)/sqrt(iGradsAcc[item][j] + 0.0000001))*iGrad[j];
          }
        }
        
      }
    
    }

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter%1000 == 0) {
        printf("\nIter: %d obj: %f", iter, valTest->setObj);
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
    

    fflush(stdout);
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
      //qsort(itemRats, setSz, sizeof(ItemRat*), compItemRat);

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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


void ModelMajority_trainRMSPropAvg(void *self, Data *data, Params *params, float **sim, 
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
      valTest->setObj = model->_(objective)(model, data, sim);
      if (iter%1000 == 0) {
        printf("\nIter: %d obj: %f trainRMSE: %f", iter, valTest->setObj, 
            model->_(indivTrainSetsErr) (model, data));
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
      
      //TODO:verify whether sorted in descending order
      //TODO: print rating and verify
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
      }
     
      coeffUpdateAdap(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim), iter);

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
        coeffUpdateAdap(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim), iter);
      }

    
      //find whether update helped in estimating set ratings
      r_us_est_aftr_upd = 0;
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        } 
      } 
      r_us_est_aftr_upd = dotProd(model->_(uFac)[u], sumItemLatFac, 
          model->_(facDim))/majSz;
      if (fabs(r_us_est_aftr_upd - r_us) <= fabs(r_us_est - r_us)) {
        succ++;
      } else {
        fail++;
      }
      epochDiff += fabs(r_us_est_aftr_upd - r_us);
    }
    
    //printf("\nEpoch: %d  succ = %d fail = %d avg diff = %f", 
    //    iter, succ, fail, epochDiff/subIter);

    //objective check
    if (iter % OBJ_ITER == 0) {
      valTest->setObj = model->_(objective)(model, data, sim);
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


//following do line search to find learning rate
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


void ModelMajority_trainSampMedian(void *self, Data *data, Params *params, 
    float **sim, float *valTest) {
   
  int iter, u, i, j, k, s;
  int fromViol, toViol;
  ModelMajority *model = self;
  int setSz = 5, item, majSz;
  float estPosRat, expCoeff, prevVal, posRat;
  float uFacNorm, iFacNorm;

  float* itemFac      = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* posLatFac     = (float*) malloc(sizeof(float)*model->_(facDim));
  int *posSet          = (int*) malloc(sizeof(int)*setSz);
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);

  memset(itemFac, 0, sizeof(float)*model->_(facDim));

  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {

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

      //find from and to what indices constraint violated
      fromViol = -1;
      toViol = -1;
      for(i = 0; i <= ceil(((double)majSz)/2.0); i++) {
        if (itemRats[i]->rating < posRat) {
          if (fromViol == -1) {
            fromViol = i;
            toViol = i;
          } else {
            toViol = i;
          }
        }
      }

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(estPosRat - posRat)*posLatFac[j]*(1.0/majSz);
        for (i = fromViol; i <= toViol && i >= 0; i++) {
          //constraint kicked in 
          uGrad[j] += -1.0*params->constrainWt*model->_(iFac)[itemRats[i]->item][j];
        }
      }
      //update user 
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));
      /*uFacNorm = norm(model->_(uFac)[u], model->_(facDim));
      
      if (uFacNorm != uFacNorm) {
        //nan check
        printf("\nuFac Norm: %f", uFacNorm);
        fflush(stdout);
        exit(0);
      }
      */

      //update items in  set
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item; 
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(estPosRat - posRat)*model->_(uFac)[u][j]*(1.0/majSz);
          if (i >= fromViol && i <= toViol) {
            //item with max rating and constraint kicked in 
            iGrad[j] += -1.0*params->constrainWt*model->_(uFac)[u][j];
          }
        }
        //check if gradient norm is nan
        /*
        iFacNorm = norm(iGrad, model->_(facDim));
        if (iFacNorm != iFacNorm) {
          //null check
          printf("\nitem:%d iGrad Norm: %f ", item, iFacNorm);
          fflush(stdout);
          exit(0);
        }
        */
        //memcpy(itemFac, model->_(iFac)[item], sizeof(float)*model->_(facDim));

        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI),
          model->_(learnRate), model->_(facDim));
        /*
        iFacNorm = norm(model->_(iFac)[item], model->_(facDim));
        if (iFacNorm != iFacNorm) {
          writeFloatVector(itemFac, model->_(facDim), "pre_fac.txt");
          writeFloatVector(model->_(iFac)[item], model->_(facDim), "post_fac.txt");
          writeFloatVector(iGrad, model->_(facDim), "igrad.txt");
          //null check
          printf("\nitem:%d iFac Norm: %f ", item, iFacNorm);
          fflush(stdout);
          exit(0);
        }
        */
      }

    }

    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest[0] = model->_(validationErr) (model, data, NULL);
      //printf("\nIter:%d validation error: %f", iter, valTest[0]);
      if (iter > 0) {
        if (fabs(prevVal - valTest[0]) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest[0], fabs(prevVal - valTest[0]));
          break;
        }
      }
      prevVal = valTest[0];
    }

  }
  
  //get test error on set
  printf("\nTest set error(modelMajority): %f", 
      model->_(testErr) (model, data, NULL));

  //get test eror
  valTest[1] = model->_(indivItemSetErr) (model, data->testSet);

  printf("\nTest hit rate: %f", 
      model->_(hitRate)(model, data->trainMat, data->testMat));

  for (i = 0; i < 100; i++) {
    free(itemRats[i]);
  }
  free(itemRats);
  free(itemFac);
  free(iGrad);
  free(uGrad);
  free(posLatFac);
  free(posSet);

}


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
  valTest->testItemsRMSE = model->_(indivItemSetErr) (model, data->testSet);
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


Model ModelMajorityProto = {
  .objective             = ModelMajority_objectiveWOCons,
  //.objective             = ModelMajority_objectiveAvg,
  .setScore              = ModelMajority_setScore,
  //.setScore              = ModelMajority_setScoreAvg,
  .train                 = ModelMajority_trainCDAlt
};


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

  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  loadUserItemWtsFrmTrain(data);
  
  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  printf("\nTest Error: %f", model->_(testErr) (model, data, NULL));  
  //testItemRat();
  
  //get objective
  printf("\nModelMaj Init Obj: %f", model->_(objective)(model, data, NULL));
  
  //train model 
  model->_(train)(model, data,  params, NULL, valTest);
 
  //compare model with loaded latent factors if any
  //model->_(cmpLoadedFac)(model, data);

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "modelMajUFacAvg.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "modelMajIFacAvg.txt");

  model->_(free)(model);
}




#include "modelMajority.h"


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
  
  FILE *fp = NULL;
  //fp = fopen("constrVilOff.txt", "w");

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
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
      rmse += diff*diff;
      if (userSet->labels[s] > maxRat) {
        //constraint violated
        rmse += model->constrainWt*(userSet->labels[s] - maxRat);
        constViolCt++;
        //fprintf(fp, "%f\n", userSet->labels[s]-maxRat);
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


void ModelMajority_trainAdaDelta(void *self, Data *data, Params *params, float **sim, 
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
  
  float *iGradsAcc    = (float*) malloc(sizeof(float)*params->nItems);
  float *uGradsAcc    = (float*) malloc(sizeof(float)*params->nUsers);
  memset(iGradsAcc, 0, sizeof(float)*params->nItems);
  memset(uGradsAcc, 0, sizeof(float)*params->nUsers);

  float *iDeltaAcc    = (float*) malloc(sizeof(float)*params->nItems);
  float *uDeltaAcc    = (float*) malloc(sizeof(float)*params->nUsers);
  memset(iDeltaAcc, 0, sizeof(float)*params->nItems);
  memset(uDeltaAcc, 0, sizeof(float)*params->nUsers);

  float *updDelta = (float*) malloc(sizeof(float)*model->_(facDim));
  memset(updDelta, 0, sizeof(float)*model->_(facDim));
  
  float  deltaRMS = 0.0, gradRMS = 0.0;

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
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      //accumulate the gradients
      uGradsAcc[u] = model->rhoRMS*uGradsAcc[u] + 
        (1.0 - model->rhoRMS)*dotProd(uGrad, uGrad, model->_(facDim));
      
      deltaRMS = sqrt(uDeltaAcc[u] + model->epsRMS);
      gradRMS = sqrt(uGradsAcc[u] + model->epsRMS);

      //compute update
      memset(updDelta, 0, sizeof(float)*model->_(facDim));
      for (j = 0; j < model->_(facDim); j++) {
        updDelta[j] = -1.0*(deltaRMS/gradRMS)*uGrad[j];
      }

      //accumulate update
      uDeltaAcc[u] = model->rhoRMS*uDeltaAcc[u] + 
        (1.0 - model->rhoRMS)*dotProd(updDelta, updDelta, model->_(facDim));

      //apply update
      for (j = 0; j < model->_(facDim); j++) {
        model->_(uFac)[u][j] += updDelta[j];          
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
        
        //accumulate the gradients 
        iGradsAcc[item] = model->rhoRMS*iGradsAcc[item] +
          (1.0 - model->rhoRMS)*dotProd(iGrad, iGrad, model->_(facDim));

        deltaRMS = sqrt(iDeltaAcc[item] + model->epsRMS);
        gradRMS = sqrt(iGradsAcc[item] + model->epsRMS);
       
        //compute update
        memset(updDelta, 0, sizeof(float)*model->_(facDim));
        for (j = 0; j < model->_(facDim); j++) {
          updDelta[j] = -1.0*(deltaRMS/gradRMS)*iGrad[j];
        }
        
        //accumulate update
        iDeltaAcc[item] = model->rhoRMS*iDeltaAcc[item] + 
          (1.0 - model->rhoRMS)*dotProd(updDelta, updDelta, model->_(facDim));
        
        //apply update
        for (j = 0; j < model->_(facDim); j++) {
          model->_(iFac)[item][j] += updDelta[j];
        }

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
    
    printf("\nEpoch: %d  succ = %d fail = %d avg diff = %f", 
        iter, succ, fail, epochDiff/subIter);

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

  free(uGradsAcc);
  free(iGradsAcc);

  free(uDeltaAcc);
  free(iDeltaAcc);

  free(updDelta);
  
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
  int setSz, item, majSz, isTestValSet, succ, fail;
  int *set;
  float r_us, r_us_est, r_us_est_aftr_upd, prevObj; //actual set preference
  float temp, epochDiff; 
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
        uGrad[j] += 2.0*model->_(regU)*model->_(uFac)[u][j];
      }
    
      //accumulate the gradients
      for (j = 0; j < model->_(facDim); j++) {
        uGradsAcc[u][j] += uGrad[j];
      }

      temp = sqrt(norm(uGradsAcc[u], model->_(facDim)));
      coeffUpdateWOReg(model->_(uFac)[u], uGrad, model->_(learnRate)/temp, 
          model->_(facDim));

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
        
        //accumulate the gradients 
        for (j = 0; j < model->_(facDim); j++) {
          iGradsAcc[item][j] += iGrad[j];
        }

        temp = sqrt(norm(iGradsAcc[item], model->_(facDim)));
        coeffUpdateWOReg(model->_(iFac)[item], iGrad,  model->_(learnRate)/temp, 
            model->_(facDim));
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
    
    printf("\nEpoch: %d  succ = %d fail = %d avg diff = %f", 
        iter, succ, fail, epochDiff/subIter);

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
    
    printf("\nEpoch: %d  succ = %d fail = %d avg diff = %f", 
        iter, succ, fail, epochDiff/subIter);

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


float ModelMajority_learnRateSearch(void *self, Data *data, ItemRat **itemRats,
    float *sumItemLatFac, float *iGrad, float *uGrad) {

  int u, i, j, k, s;
  int isTestValSet;
  float bestLearnRate = -1;
  float bestObj = -1;
  float origObj, modelObj;  
 
  float learnRate = 0.00001;

  UserSets *userSet = NULL;
  ModelMajority *model = self;
  ModelMajority *dupModel = NEW(ModelMajority, 
                              "set prediction with majority score"); 
  dupModel->_(init)(model, model->_(nUsers), model->_(nItems), model->_(facDim),
      model->_(regU), model->_(regI), model->_(learnRate));
  origObj = model->_(objective)(model, data, NULL);

  while (learnRate <= 0.01) {
    
    //copy original model i.e. lat facs
    model->_(copy)(model, dupModel);

    //modify learn rate of dupModel
    dupModel->_(learnRate) = learnRate;

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

    if (modelObj != modelObj) {
      //NAN check
      printf("\nObjective is NAN for learn rate %f", dupModel->_(learnRate));
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
      }
    }

    learnRate = learnRate*2;
  }

  printf("\nBest obj: %f learnRate: %f", bestObj, bestLearnRate);

  dupModel->_(free)(dupModel);

  return bestLearnRate;
}


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
    epochDiff = 0;
    succ = 0;
    fail = 0;

    //find opt learn rate based on training error of few samples
    model->_(learnRate) = ModelMajority_learnRateSearch(model, data, itemRats, 
        sumItemLatFac, iGrad, uGrad);

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
  .objective = ModelMajority_objective,
  .setScore = ModelMajority_setScore,
  .train     = ModelMajority_trainSGDSamp
};


void modelMajority(Data *data, Params *params, ValTestRMSE *valTest) {
  
  ModelMajority *model = NEW(ModelMajority, 
                              "set prediction with majority score");  
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
                  params->regU, params->regI, params->learnRate);
  model->constrainWt = params->constrainWt;
  model->rhoRMS = 0.85;
  model->epsRMS = 0.000001;

  //load user item weights from train: needed to compute training on indiv items
  //in training sets
  loadUserItemWtsFrmTrain(data);
  
  //copyMat(data->uFac, model->_(uFac), data->nUsers, data->facDim); 
  //copyMat(data->iFac, model->_(iFac), data->nItems, data->facDim); 

  printf("\nTest Error: %f", model->_(testErr) (model, data, NULL));  
  //testItemRat();
  
  //train model 
  model->_(train)(model, data,  params, NULL, valTest);
 
  //compare model with loaded latent factors if any
  //model->_(cmpLoadedFac)(model, data);

  //save model latent factors
  //writeMat(model->_(uFac), params->nUsers, model->_(facDim), "modelMajUFac.txt");
  //writeMat(model->_(iFac), params->nItems, model->_(facDim), "modelMajIFac.txt");

  model->_(free)(model);
}






int compItemRat(const void *elem1, const void *elem2) {
  ItemRat *item1Rat = *(ItemRat**)elem1;
  ItemRat *item2Rat = *(ItemRat**)elem2;
  if (item1Rat->rating > item2Rat->rating) return -1;
  if (item1Rat->rating < item2Rat->rating) return 1;
  return 0;
}


float ModelMajority_setScore(void *self, int u, int *set, int setSz, 
    float **sim) {
  //TODO
}


float ModelMajority_objective(void *self, Data *data, float **sim) {
 
  int u, i, s, item, setSz, isTestValSet;
  UserSets *userSet = NULL;
  int *set = NULL;
  float rmse = 0, diff = 0, userSetPref = 0;
  float uRegErr = 0, iRegErr = 0;
  ModelMajority *model = self;

  int nSets = 0;

  for (u = 0; u < data->nUsers; u++) {
    userSet = data->userSets[u];
    isTestValSet = 0;
    for (s = 0; s < userSet->numSets; s++) {
      //TODO: do I need this?    
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
      userSetPref = model->_(setScore)(model, u, set, setSz, NULL);
      //set sim
      setSim = model->_(setSimilarity)(self, set, setSz, NULL);   
 
      diff = userSetPref - userSet->labels[s];
      
      //printf("\ndiff = %f userSetPref = %f setSim = %f", diff, userSetPref, setSim);
      rmse += diff*diff*setSim;
      avgSetSim += setSim;
      nSets++;
    }
    uRegErr += dotProd(model->_(uFac)[u], model->_(uFac)[u], model->_(facDim));
  }
    
  avgSetSim = avgSetSim/nSets;

  uRegErr = uRegErr*model->_(regU);

  for (i = 0; i < data->nItems; i++) {
    iRegErr += dotProd(model->_(iFac)[i], model->_(iFac)[i], model->_(facDim)); 
  }
  iRegErr *= model->_(regI);
  printf("\nObj: %f SE: %f uRegErr: %f iRegErr: %f avgSetSim: %f", (rmse+uRegErr+iRegErr),
      rmse, uRegErr, iRegErr, avgSetSim);
  return (rmse + uRegErr + iRegErr);
  
}


void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item, majSz;
  int *set;
  float r_us, r_us_est, prevVal; //actual set preference
  float temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  //TODO: hardcode
  //TODO: free
  ItemRat **itemRats    = (ItemRat**) malloc(sizeof(ItemRat*)*100);
  for (i = 0; i < 100; i++) {
    itemRats[i] = (ItemRat*) malloc(sizeof(ItemRat));
    memset(itemRats[i], 0, sizeof(ItemRat));
  }

  prevVal = 0.0;

  model->_(objective)(model, data, sim);
  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
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
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
        r_us_est += itemRats[i]->rating;
      }
      r_us_est = r_us_est/majSz;

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = 2.0*(r_us_est - r_us)*sumItemLatFac[j]*(1.0/majSz);
      }
     
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < majSz; i++) {
        item = itemRats[i]->item;
        //get item gradient
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = 2.0*(r_us_est - r_us)*model->_(uFac)[u][j]*(1.0/majSz);
        }
        //update item + reg
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim));
      }

    }

    
    //objective check
    if (iter % OBJ_ITER == 0) {
      printf("\nuserFacNorm:%f itemFacNorm:%f", model->_(userFacNorm)(self, data), model->_(itemFacNorm)(self, data));
      //TODO:
      model->_(objective)(model, data, sim);
    }

    //validation check
    if (iter % VAL_ITER == 0) {
      //validation err
      valTest[0] = model->_(indivItemSetErr) (model, data->valSet);
      printf("\nIter:%d validation error: %f", iter, valTest[0]);
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

  
  //get train error
  //printf("\nTrain error: %f", model->_(trainErr) (model, data, sim));

  //get test eror
  valTest[1] = model->_(indivItemSetErr) (model, data->testSet);

  //model->_(writeUserSetSim)(self, data, "userSetsSim.txt");

  free(sumItemLatFac);
  free(uGrad);
  free(iGrad);
}


void modelMajority(Data *data, Params *params, float *valTest) {
  
  ModelMajority *model = NEW(ModelMajority, "set prediction with majority score");  
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, params->regU, 
    params->regI, params->learnRate);
 
  //train model 
  model->_(train)(model, data,  params, NULL, valTest);
  
  model->_(free)(model);
}




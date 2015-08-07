


void ModelMajority_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {

  int iter, u, i, j, k, s;
  UserSets *userSet;
  int isTestValSet, setSz, item;
  int *set;
  float r_us, prevVal; //actual set preference
  float avgSetSim, temp; 
  ModelMajority *model = self;
  float* sumItemLatFac = (float*) malloc(sizeof(float)*model->_(facDim));
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  
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
      
      //dont update no. of items if set is 1
      if (setSz == 1) {
        continue;
      }

      avgSetSim = model->_(setSimilarity)(self, set, setSz, sim);
      //avgSetSim = model->_(setSimilarity)(self, set, setSz, NULL);
      
      //update user and item latent factor
     
      memset(sumItemLatFac, 0, sizeof(float)*model->_(facDim));
      for (i = 0; i < setSz; i++) {
        item = set[i];
        for (j = 0; j < model->_(facDim); j++) {
          sumItemLatFac[j] += model->_(iFac)[item][j];
        }
      }
      
      //TODO
      //compute user gradient
      computeUGrad(model, u, set, setSz, avgSetSim, r_us, sumItemLatFac, uGrad);
      //update user + reg
      //printf("\nb4 u = %d norm uFac = %f uGrad = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)), norm(uGrad, model->_(facDim)));
      
      coefficientNormUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //printf("\naftr u = %d norm uFac = %f", 
      //    u, norm(model->_(uFac)[u], model->_(facDim)));
      //compute  gradient of item and update their latent factor in sets
      for (i = 0; i < setSz; i++) {
        item = set[i];
        //TODO
        //get item gradient
        computeIGrad(model, u, item, set, setSz, avgSetSim, r_us, sumItemLatFac,
           iGrad);
        //update item + reg
        //printf("\nitem = %d norm iGrad = %f", item, norm(iGrad, model->_(facDim)));
        coefficientNormUpdate(model->_(iFac)[item], iGrad, model->_(regI), 
            model->_(learnRate), model->_(facDim));
      }

    }

    //update sim
    model->_(updateSim)(model, sim);
    
    //objective check
    if (iter % OBJ_ITER == 0) {
      printf("\nuserFacNorm:%f itemFacNorm:%f", model->_(userFacNorm)(self, data), model->_(itemFacNorm)(self, data));

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




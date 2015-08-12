#include "modelBPR.h"

void ModelBPR_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {
 
  int iter, nUserItems, posItem, negItem, found;
  int u, i, j, k, s;
  ModelBPR *model = self;
  float r_uij, expCoeff, prevVal;
  gk_csr_t *trainMat   = data->trainMat;
  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
 
  printf("\nValidation error: %f", model->_(hitRate)(model, 
        data->trainMat, data->valMat));

  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
      nUserItems = trainMat->rowptr[u+1] - trainMat->rowptr[u];

      //sample a positive item
      k = rand()%nUserItems; 
      posItem = trainMat->rowind[trainMat->rowptr[u] + k];

      //sample a negative item
      negItem = -1;
      while (negItem == -1) {
        k = rand() % trainMat->ncols;
        //ignore if item k is rated by user
        found = 0;
        for (i = trainMat->rowptr[u]; i < trainMat->rowptr[u+1]; i++) {
          if (k == trainMat->rowind[i]) {
            found = 1;
            break;
          }
        }
        if (found) {
          continue;
        }
        negItem = k;
      }

      r_uij = dotProd(model->_(uFac)[u], model->_(iFac)[posItem], model->_(facDim)) -
              dotProd(model->_(uFac)[u], model->_(iFac)[negItem], model->_(facDim));
      expCoeff = -1.0/(1.0 + exp(r_uij));
      
      //printf("\nexpCoeff= %f", expCoeff);

      //compute user gradient
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = expCoeff*(model->_(iFac)[posItem][j] - 
            model->_(iFac)[negItem][j]);
      }

      //update user
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //compute pos item gradient
      for (j = 0; j < model->_(facDim); j++) {
        iGrad[j] = expCoeff*model->_(uFac)[u][j];
      }

      //update pos item
      coeffUpdate(model->_(iFac)[posItem], iGrad, model->_(regI),
        model->_(learnRate), model->_(facDim));

      //compute neg item gradient
      for (j = 0; j < model->_(facDim); j++) {
        iGrad[j] = -1.0*expCoeff*model->_(uFac)[u][j];
      }
      
      //update neg item
      coeffUpdate(model->_(iFac)[negItem], iGrad, model->_(regI),
        model->_(learnRate), model->_(facDim));

    }
    
    //compute validation 
    if (iter % VAL_ITER == 0) {
      valTest[0] = model->_(hitRate)(model, data->trainMat, data->valMat);
      //printf("\nIter: %d val err: %f val rmse Err: %f", iter, valTest[0], model->_(indivItemSetErr) (model, data->valSet));
      if (iter % 100 == 0) {
        if (iter > 0 && fabs(prevVal - valTest[0]) < EPS) {
          //exit the model train procedure
          printf("\nConverged in iteration: %d prevVal: %f currVal: %f diff: %f", iter,
              prevVal, valTest[0], fabs(prevVal - valTest[0]));
          break;
        }
        prevVal = valTest[0];
      }
    }
  
  }

  //get test hitrate
  valTest[1] = model->_(hitRate)(model, data->trainMat, data->testMat);
  printf("\nTest hit rate: %f", valTest[1]);
  
  free(iGrad);
  free(uGrad);
}


Model ModelBPRProto = {
  .train = ModelBPR_train
};

void modelBPR(Data *data, Params *params, float *valTest) {

  ModelBPR *model = NEW(ModelBPR, "bpr model for individual items");
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
                  params->regU, params->regI, params->learnRate);
  //train model
  model->_(train)(model, data, params, NULL, valTest);
  model->_(free)(model);
}



#include "modelCofiSet.h"

//TODO: add threshold on rating value for sampling
void samplePosSet(Data *data, int u, int *posSet, int setSz) {
  
  int i, nItems, itemInd, item, found;
  int nUserItems;
  gk_csr_t *trainMat = data->trainMat;

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
    nItems++;
  }

}


void sampleNegSet(Data *data, int u, int *negSet, int setSz) {

  int ii, nItems, itemInd, item, found;
  gk_csr_t *trainMat = data->trainMat;

  nItems = 0;

  while (nItems < setSz) {
    found = 0;
    item = rand()%trainMat->ncols;
    for (ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      if (item == trainMat->rowind[ii]) {
        found = 1;
        break;
      }
    }
    if (found) {
      continue;
    }
    
    negSet[nItems] = item;
    nItems++;
  }

}


void ModelCofi_train(void *self, Data *data, Params *params, float **sim, 
    float *valTest) {
 
  int iter, u, i, j, k, s;
  ModelCofi *model = self;
  int setSz = 5, item;
  float r_upa, expCoeff, prevVal;

  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* diffLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  float* posLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  float* negLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  int *posSet          = (int*) malloc(sizeof(int)*setSz);
  int *negSet          = (int*) malloc(sizeof(int)*setSz);
  
  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
      memset(diffLatFac, 0, sizeof(float)*model->_(facDim));
      memset(posLatFac, 0, sizeof(float)*model->_(facDim));
      memset(negLatFac, 0, sizeof(float)*model->_(facDim));

      //sample a positive set for u
      samplePosSet(data, u, posSet, setSz);
      
      //sample a negative set for u
      sampleNegSet(data, u, negSet, setSz);

      //sum lat fac from pos set
      for (i = 0; i < setSz; i++) {
        for (j = 0; j < model->_(facDim); j++) {
          posLatFac[j] += model->_(iFac)[posSet[i]][j];
        }
      }
      for (j = 0; j < model->_(facDim); j++) {
        posLatFac[j] = posLatFac[j] / (setSz*1.0);
      }


      //sum lat fac from neg set
      for (i = 0; i < setSz; i++) {
        for (j = 0; j < model->_(facDim); j++) {
          negLatFac[j] += model->_(iFac)[negSet[i]][j];
        }
      }
      for (j = 0; j < model->_(facDim); j++) {
        negLatFac[j] = negLatFac[j] / (setSz*1.0);
      }

      r_upa = dotProd(model->_(uFac)[u], posLatFac, model->_(facDim)) - 
                dotProd(model->_(uFac)[u], negLatFac, model->_(facDim));
      expCoeff = -1.0/(1.0 + exp(r_upa));

      //update user
      for (j = 0; j < model->_(facDim); j++) {
        uGrad[j] = expCoeff*(posLatFac[j] - negLatFac[j]);
      }
      coeffUpdate(model->_(uFac)[u], uGrad, model->_(regU),
          model->_(learnRate), model->_(facDim));

      //update items in positive set
      for (i = 0; i < setSz; i++) {
        item = posSet[i]; 
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = expCoeff*(model->_(uFac)[u][j]/(1.0*setSz));
        }
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI),
          model->_(learnRate), model->_(facDim));
      }

      //update items in negative set
      for (i = 0; i < setSz; i++) {
        item = negSet[i]; 
        for (j = 0; j < model->_(facDim); j++) {
          iGrad[j] = -1.0*expCoeff*(model->_(uFac)[u][j]/(1.0*setSz));
        }
        coeffUpdate(model->_(iFac)[item], iGrad, model->_(regI),
          model->_(learnRate), model->_(facDim));
      }
      
    }

    //compute validation 
    if (iter % VAL_ITER == 0) {
      valTest[0] = model->_(hitRate)(model, data->trainMat, data->valMat);
      //printf("\nIter: %d val err: %f val rmse Err: %f", iter, valTest[0], model->_(indivItemSetErr) (model, data->valSet));
      if (iter % 100 == 0) {
        if (iter > 0  && fabs(prevVal - valTest[0]) < EPS) {
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
  free(diffLatFac);
  free(posLatFac);
  free(negLatFac);
  free(posSet);
  free(negSet);
}


Model ModelCofiProto = {
  .train = ModelCofi_train
};


void modelCofi(Data *data, Params *params, float *valTest) {
  
  ModelCofi *model = NEW(ModelCofi, "cofiset algo implementation");
  model->_(init)(model, params->nUsers, params->nItems, params->facDim, 
                  params->regU, params->regI, params->learnRate);
  model->_(train)(model, data, params, NULL, valTest);
  model->_(free)(model);
}





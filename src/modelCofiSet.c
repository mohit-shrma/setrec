#include "modelCofiSet.h"

//TODO: add threshold on rating value for sampling
void samplePosSet(Data *data, int *posSet, int setSz) {
  
  int i, nItems, itemInd, item, found;
  float rating;
  UserSets *userSet = NULL;
 
  nItems = 0;
  userSet = data->userSets[u];
  while (nItems < setSz) {
    found = 0;
    itemInd = rand()%userSet->nUserItems;
    item = userSet->itemWtSets[itemInd]->item;  
    rating = userSet->itemWtSets[itemInd]->wt;  
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


void sampleNegSet(Data *data, int *negSet, int setSz) {

  int i, nItems, itemInd, item, found;
  float rating;
  UserSets *userSet = NULL;
  nItems = 0;

  userSet = data->userSets[u];
  while (nItems < setSz) {
    found = 0;
    item = rand()%data->nItems;
    for (i = 0; i < userSet->nUserItems; i++) {
      if (item == userSet->itemWtSets[i]->item) {
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
  UserSets *userSet = NULL;
  ModelCofi *model = self;
  int setSz = 5, item;
  float r_upa, expCoeff;

  float* iGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* uGrad         = (float*) malloc(sizeof(float)*model->_(facDim));
  float* diffLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  float* posLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  float* negLatFac    = (float*) malloc(sizeof(float)*model->_(facDim));
  int *posSet          = (int*) malloc(sizeof(int)*setSz);
  int *negSet          = (int*) malloc(sizeof(int)*setSz);
  
  for (iter = 0; iter < params->maxIter; iter++) {
    for (u = 0; u < data->nUsers; u++) {
      userSet = data->userSets[u];
      memset(diffLatFac, 0, sizeof(float)*model->_(facDim));
      memset(posLatFac, 0, sizeof(float)*model->_(facDim));
      memset(negLatFac, 0, sizeof(float)*model->_(facDim));

      //sample a positive set for u
      samplePosSet(data, posSet, setSz);
      
      //sample a negative set for u
      sampleNegSet(data, negSet, setSz);

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
  }

  
  free(iGrad);
  free(uGrad);
  free(diffLatFac);
  free(posLatFac);
  free(negLatFac);
  free(posSet);
  free(negSet)
}


void modelCofi(Data *data, Params *params, float *valTest) {
  
  loadUserItemWtsFrmTrain(data); 

}





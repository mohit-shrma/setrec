#ifndef _MODEL_H_
#define _MODEL_H_

#include "datastruct.h"
#include "util.h"
#include <stdio.h>
#include <string.h>

#define EPS 0.000001
#define OBJ_ITER 1
#define VAL_CONV 1
#define OBJ_CONV 1
#define VAL_ITER 1

typedef struct {
  int nUsers;
  int nItems;
  
  //user and item latent factors
  float **uFac;
  float **iFac;
  
  //user and item regularization
  float regU;
  float regI;

  //size of latent factors
  int facDim;

  float learnRate;

  char *description; //identify which model it is

  //TODO: pointer to common methods in base model
  void (*init) (void *self, int nUsers, int nItems, int facDim, float regU,
      float regI, float learnRate);
  void (*describe) (void *self);
  void (*updateSim) (void *self, float **sim);
  float (*objective) (void *self, Data *data, float **sim);
  float (*setScore) (void *self, int user, int *set, int setSz, float **sim);
  float (*validationErr) (void *self, Data *data, float **sim);
  float (*hingeValidationErr) (void *self, Data *data, float **sim);
  float (*validationClassLoss) (void *self, Data *data, float **sim);
  float (*testErr) (void *self, Data *data, float **sim);
  float (*hingeTestErr) (void *self, Data *data, float **sim);
  float (*testClassLoss) (void *self, Data *data, float **sim);
  float (*trainErr) (void *self, Data *data, float **sim);
  float (*trainClassLoss) (void *self, Data *data, float **sim);
  float (*userFacNorm) (void *self, Data *data);
  float (*itemFacNorm) (void *self, Data *data);
  float (*setSimilarity) (void *self, int *set, int setSz, float **sim);  
  void (*writeUserSetSim) (void *self, Data *data, float **sim, char *fName); 
  float (*indivItemSetErr) (void *self, RatingSet *ratSet);
  float (*indivTrainSetsErr) (void *self, Data *data);
  float (*hitRate) (void *self, gk_csr_t *trainMat, gk_csr_t *testMat);
  void (*train) (void *self, Data *data, Params *params, float **Sim, ValTestRMSE *valTest);
  void (*free) (void *self);
  void (*reset) (void *self);
  
} Model;

void Model_init(void *self, int nUsers, int nItems, int facDim, float regU, 
    float regI, float learnRate);
void Model_reset(void *self);
void Model_free(void *self);
void Model_describe(void *self);
void Model_updateSim(void *self, float **sim);
float Model_objective(void *self, Data *data, float **sim);
float Model_setScore(void *self, int user, int *set, int setSz, float **sim);
float Model_setSimilarity(void *self, int *set, int setSz, float **sim);
void Model_writeUserSetSim(void *self, Data *data, float **sim, char *fName);
void Model_train(void *self, Data *data, Params *params, float **Sim, ValTestRMSE *valTest);
float Model_validationErr(void *self, Data *data, float **sim);
float Model_hingeValErr(void *self, Data *data, float **sim);
float Model_validationClassLoss(void *self, Data *data, float **sim);
float Model_testErr(void *self, Data *data, float **sim);
float Model_hingeTestErr(void *self, Data *data, float **sim);
float Model_testClassLoss(void *self, Data *data, float **sim);
float Model_trainErr(void *self, Data *data, float **sim);
float Model_trainClassLoss(void *self, Data *data, float **sim);
float Model_userFacNorm(void *self, Data *data);
float Model_itemFacNorm(void *Self, Data *data);
float Model_indivItemSetErr(void *self, RatingSet *ratSet);
float Model_hitRate(void *self, gk_csr_t *trainMat, gk_csr_t *testMat);
float Model_indivTrainSetsErr(void *self, Data *data);
void *Model_new(size_t size, Model proto, char *description);

#define NEW(T, N) Model_new(sizeof(T), T##Proto, N)
#define _(N) proto.N

#endif

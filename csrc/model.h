#ifndef _MODEL_H_
#define _MODEL_H_

#include "datastruct.h"
#include "util.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <gsl/gsl_statistics.h>
#include <omp.h>

#define EPS 0.000001
#define OBJ_ITER 500
#define VAL_CONV 1
#define OBJ_CONV 1
#define VAL_ITER 100
#define MAX_SET_SZ 5
#define MAX_RAT 5.0

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
  float (*validationClass01Loss) (void *self, Data *data, float **sim);
  float (*testErr) (void *self, Data *data, float **sim);
  float (*hingeTestErr) (void *self, Data *data, float **sim);
  float (*testClassLoss) (void *self, Data *data, float **sim);
  float (*testClass01Loss) (void *self, Data *data, float **sim);
  float (*trainErr) (void *self, Data *data, float **sim);
  float (*trainClassLoss) (void *self, Data *data, float **sim);
  float (*trainClass01Loss) (void *self, Data *data, float **sim);
  float (*userFacNorm) (void *self, Data *data);
  float (*itemFacNorm) (void *self, Data *data);
  float (*setSimilarity) (void *self, int *set, int setSz, float **sim);  
  void (*writeUserSetSim) (void *self, Data *data, float **sim, char *fName); 
  float (*indivTrainSetsErr) (void *self, Data *data);
  float (*indivItemCSRErr) (void *self, gk_csr_t *mat, char *opName); 
  float (*coldHitRate) (void *self, UserSets **userSets, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N);
  float (*coldHitRateTr) (void *self, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N);
  float (*coldHitRateTrPar) (void *self, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N);
  
  float (*indivTrainSetsScaledErr) (void *self, Data *data, float maxRat);
  float (*getMaxEstTrainRat) (void *self, Data *data);
  float (*indivItemCSRScaledErr) (void *self, gk_csr_t *mat, float maxRat, 
    char *opFName);
  float (*subMatKnownRankErr) (void *self, float **uFac, float **iFac, 
    int facDim, int uStart, int uEnd, int iStart, int iEnd); 
  float (*itemFeatScore) (void *self, int u, int item, gk_csr_t *featMat);
  float (*hitRate) (void *self, gk_csr_t *trainMat, gk_csr_t *testMat);
  float (*hitRateOrigTopN) (void *self, gk_csr_t *trainMat, float **origUFac, 
      float **origIFac, int N);
  double (*spearmanRankCorrN) (void *self, gk_csr_t *testMat, int N);
  float (*cmpLoadedFac) (void *self, Data *data);
  void (*train) (void *self, Data *data, Params *params, float **Sim, ValTestRMSE *valTest);
  int  (*isTerminateModel) (void *self, void *bestM, int iter, int *bestIter, 
      float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data); 
  int  (*isTerminateClassModel) (void *self, void *bestM, int iter, int *bestIter, 
      float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data);
  int (*isTerminateColdModel) (void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data, 
    int *testItemIds, int nTestItems);
  void (*free) (void *self);
  void (*reset) (void *self);
  void (*copy) (void *self, void *dest);  
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
float Model_validationClass01Loss(void *self, Data *data, float **sim);
float Model_testErr(void *self, Data *data, float **sim);
float Model_hingeTestErr(void *self, Data *data, float **sim);
float Model_testClassLoss(void *self, Data *data, float **sim);
float Model_testClass01Loss(void *self, Data *data, float **sim);
float Model_trainErr(void *self, Data *data, float **sim);
float Model_trainClassLoss(void *self, Data *data, float **sim);
float Model_trainClass01Loss(void *self, Data *data, float **sim);
float Model_userFacNorm(void *self, Data *data);
float Model_itemFacNorm(void *Self, Data *data);
float Model_itemFeatScore(void *self, int u, int item, gk_csr_t *featMat);
float Model_coldHitRate(void *self, UserSets **userSets, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N); 
float Model_coldHitRateTr(void *self, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N); 
float Model_coldHitRateTrPar(void *self, gk_csr_t *trainMat, gk_csr_t *testMat, 
    gk_csr_t *itemFeatMat, int *testItemIds, int nTestItems, int N); 
float Model_subMatKnownRankErr(void *self, float **uFac, float **iFac, 
    int facDim, int uStart, int uEnd, int iStart, int iEnd);
float Model_hitRate(void *self, gk_csr_t *trainMat, gk_csr_t *testMat);
float Model_hitRateOrigTopN(void *self, gk_csr_t *trainMat, float **origUFac, 
    float **origIFac, int N);
double Model_spearmanRankCorrN(void *self, gk_csr_t *testMat, int N);
float Model_indivTrainSetsScaledErr(void *self, Data *data, float maxRat);
float Model_getMaxEstTrainRat(void *self, Data *data);
float Model_indivItemCSRScaledErr(void *self, gk_csr_t *mat, float maxRat, 
    char *opFName);
float Model_indivTrainSetsErr(void *self, Data *data);
float Model_indivItemCSRErr(void *self, gk_csr_t *mat, char *opName);
float Model_cmpLoadedFac(void *self, Data *data);
void  Model_copy (void *self, void *dest);
int Model_isTerminateModel(void *self, void *bestM, int iter, int *bestIter, float *bestObj, 
    float *prevObj, ValTestRMSE *valTest, Data *data);
int Model_isTerminateClassModel(void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data);
int Model_isTerminateColdModel(void *self, void *bestM, int iter, int *bestIter, 
    float *bestObj, float *prevObj, ValTestRMSE *valTest, Data *data, 
    int *testItemIds, int nTestItems);
void *Model_new(size_t size, Model proto, const char *description);

#define NEW(T, N) Model_new(sizeof(T), T##Proto, N)
#define _(N) proto.N

#endif

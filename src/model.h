#ifndef _MODEL_H_
#define _MODEL_H_

#include "datastruct.h"
#include "util.h"
#include <stdio.h>
#include <string.h>

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
  float (*objective) (void *self, Data *data);
  float (*setScore) (void *self, int user, int *set, int setSz, float **sim);
  float (*validationErr) (void *self, Data *data, float **sim);
  float (*testErr) (void *self, Data *data, float **sim);
  void  (*train) (void *self, Data *data, Params *params, float **Sim);
  void (*free) (void *self);
} Model;

void Model_init(void *self, int nUsers, int nItems, int facDim, float regU, float regI, float learnRate);
void Model_free(void *self);
void Model_describe(void *self);
void Model_updateSim(void *self, float **sim);
float Model_objective(void *self, Data *data);
float Model_setScore(void *self, int user, int *set, int setSz, float **sim);
void Model_train(void *self, Data *data, Params *params, float **Sim);
float Model_validationErr(void *self, Data *data, float **sim);
float Model_testErr(void *self, Data *data, float **sim);
void *Model_new(size_t size, Model proto, char *description);

#define NEW(T, N) Model_new(sizeof(T), T##Proto, N)
#define _(N) proto.N

#endif

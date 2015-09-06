#ifndef _DATASTRUCT_H
#define _DATASTRUCT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "GKlib.h"
#include "set.h"


typedef struct {
  float valItemsRMSE;
  float testItemsRMSE;
  float trainItemsRMSE;
  float valSetRMSE;
  float testSetRMSE;
  float trainSetRMSE;
  float setObj;
} ValTestRMSE;


typedef struct {
  float corr;
  float misClassLoss;
} Metric;


typedef struct {
  char *user_set_file;
  char *ext_setSim_file;
  char *val_set_file;
  int val_set_size;
  char *test_set_file;
  int test_set_size;
  char *train_set_file;
  int train_set_size;
  char *train_mat_file;
  char *val_mat_file;
  char *test_mat_file;
  char *uFacFileName;
  char *iFacFileName;
  int nUsers;
  int nItems;
  int facDim;
  float regU;
  float regI;
  float learnRate;
  int useSim;
  int maxIter;
  int seed;
  float constrainWt;
  float rhoRMS;
  float epsRMS;
} Params;

void Params_display(Params *params);


typedef struct {
  //index of item in parent array
  int ind;

  //itemId 
  int item; 
  
  //user score on item 
  float wt; 
  
  //indices of set where item appears in user history 
  int *itemSets; 
  
  //no. of set where item appears or size of above array 
  int szItemSets;

} ItemWtSets;


typedef struct {
  
  int userId;
 
  //number of labeled sets in user history
  int numSets;
  
  //array of user sets in history
  int **uSets;
  
  //size of sets in user history
  int *uSetsSize;

  //array of set labels 
  float *labels;

  //validation set indices and no. of validation sets
  int *valSets;
  int szValSet;

  //test set indices and no. of test sets
  int *testSets;
  int szTestSet;

  //items preferred by user
  int nUserItems; 

  ItemWtSets **itemWtSets;

  //true for item which occur in only test or validation set
  int *testValItems;
} UserSets;


void UserSets_init(UserSets *self, int user, int numSets, int nItems,
    int nUserItems);
void UserSets_reset(UserSets * const self, int nItems);
void UserSets_free(UserSets *self);
void UserSets_initWt(UserSets *self);
void UserSets_updWt_avgItemSim(UserSets *self, float **sim);
void UserSets_updWt_avgItemPairSim(UserSets *self, float **sim);
void UserSets_sortItems(UserSets *self);
void UserSets_dispWt(UserSets *self);
void UserSets_writeWt(UserSets *self, char *fileName);
ItemWtSets* UserSets_search(UserSets *self, int item);


typedef struct {

  //item x users
  int **itemUsers;

  int nItems;
  int nUsers;

} ItemSets;

void ItemSets_init(ItemSets *itemSets, Data *data);
void ItemSets_free(ItemSets *self);

typedef struct {
  int user;
  int item;
  float rat;
} Rating;

typedef struct {
  Rating **rats;
  int size;
} RatingSet;

void RatingSet_init(RatingSet *self, int size);
void RatingSet_free(RatingSet *self);


typedef struct {
  int nUsers;
  int nItems;

  UserSets **userSets;
  
  RatingSet *testSet;
  RatingSet *valSet;  
  RatingSet *trainSet;
  
  gk_csr_t *trainMat;
  gk_csr_t *valMat;
  gk_csr_t *testMat;

  int facDim;
  float **uFac;
  float **iFac;

  ItemSets *itemSets;
} Data;

void Data_init(Data *self, int nUsers, int nItems);
void Data_free(Data *self);
void Data_reset(Data *self, int nUsers, int nItems);
void Data_jaccSim(Data *self, float **sim);

void loadUserItemWtsFrmTrain(Data *data);


typedef struct {
  int item;
  float rating;
} ItemRat;

int compItemRat(const void *elem1, const void *elem2);

#endif



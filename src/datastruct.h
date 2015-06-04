#ifndef _DATASTRUCT_H
#define _DATASTRUCT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


typedef struct {
  float corr;
  float misClassLoss;
} Metric;


typedef struct {
  char *user_set_file;
  int nUsers;
  int nItems;
  int facDim;
  float regU;
  float regI;
  float learnRate;
  int useSim;
  int maxIter;
  int seed;
} Params;


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
void UserSets_free(UserSets *self);
void UserSets_initWt(UserSets *self);
void UserSets_updWt(UserSets *self, float **sim);
void UserSets_sortItems(UserSets *self);
void UserSets_dispWt(UserSets *self);
ItemWtSets* UserSets_search(UserSets *self, int item);


typedef struct {
  
  int nUsers;
  int nItems;
  
  UserSets **userSets;

} Data;

void Data_init(Data *self, int nUsers, int nItems);
void Data_free(Data *self);


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
} Model;

void Model_init(Model *self, int nUsers, int nItems, int facDim, float regU, float regI, float learnRate);
void Model_free(Model *self);

#endif



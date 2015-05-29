#ifndef _DATASTRUCT_H
#define _DATASTRUCT_H

#include <stdlib.h>
#include <string.h>

typedef struct {
  char *user_set_file;
  int nUsers;
  int nItems;
} Params;


typedef struct {
  
  //indices of set to which item belongs
  int* setInd;
  
  //number of sets to which item belongs
  int numSets; 

} ItemSets;

void ItemSets_init(ItemSets *self, int sz);
void ItemSets_free(ItemSets *self);

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


  //items preferred by user
  int *items;
  int nUserItems;

  //TODO: creation item to sets mapping
  //map of items to user sets  indices
  //will be null for items which dont occur 
  //itemSets[i1] = [1,3,5] #uSets[1], uSets[3], uSets[5]
  int** itemSets;
  
  //size of individual item sets in above arr
  int *itemSetsSize;

} UserSets;

void UserSets_init(UserSets *self, int user, int numSets, int nItems,
    int nUserItems);
void UserSets_free(UserSets *self, int nItems);

typedef struct {
  
  int nUsers;
  int nItems;
  
  UserSets *userSets;

} Data;

void Data_init(Data *self, int nUsers, int nItems);
void Data_free(Data *self);

#endif



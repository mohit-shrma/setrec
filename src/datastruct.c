#include "datastruct.h"


void UserSets_init(UserSets * const self, int user, int numSets, int nItems,
  int nUserItems) {
  
  int i;
  
  self->userId = user;
  self->numSets = numSets;
  self->nUserItems = nUserItems;

  self->uSets = (int **) malloc(sizeof(int*)*numSets);
  memset(self->uSets, 0, sizeof(int*)*numSets);

  self->labels = (float*) malloc(sizeof(float)*numSets);
  memset(self->labels, 0, sizeof(float)*numSets);
  
  self->uSetsSize = (int*) malloc(sizeof(int) * numSets);
  memset(self->uSetsSize, 0, sizeof(int)*numSets);
  
  //map of item to sets, NOTE: this exists for all items
  self->itemSets = (int **) malloc(sizeof(int*)*nItems);
  memset(self->itemSets, 0, sizeof(int*)*nItems);
  self->itemSetsSize = (int *) malloc(sizeof(int)*nItems);
  memset(self->itemSetsSize, 0, sizeof(int)*nItems);
  
  self->items = (int*) malloc(sizeof(int)*nUserItems);
  memset(self->items, 0, sizeof(int)*nUserItems);
}


void UserSets_free(UserSets * const self) {
  int i, item;
 
  //free arrays of arrays
  for (i = 0; i < self->numSets; i++) {
    free(self->uSets[i]);
  }
  free(self->uSets);

  for (i = 0; i < self->nUserItems; i++) {
    item = self->items[i];
    free(self->itemSets[item]);
  }
  free(self->itemSets);
  
  free(self->uSetsSize);
  free(self->labels);
  free(self->items);
  free(self->itemSetsSize);
  
}


void Data_init(Data *self, int nUsers, int nItems) {
  
  int i;
  self->nUsers = nUsers;
  self->nItems = nItems;

  self->userSets = (UserSets**) malloc(sizeof(UserSets*)*nUsers);
  for (i = 0; i < nUsers; i++) {
    self->userSets[i] = (UserSets*) malloc(sizeof(UserSets));
  }
  
}


void Data_free(Data *self) {
  int i;

  for (i = 0; i < self->nUsers; i++) {
    UserSets_free(self->userSets[i]);
  }
  free(self->userSets);
  
  free(self);
}


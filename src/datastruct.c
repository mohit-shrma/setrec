#include "datastruct.h"


void UserSets_init(UserSets *self, int user, int numSets, int nItems,
  int nUserItems) {
  
  int i;
  
  self->userId = user;
  self->numSets = numSets;
  
  self->uSets = (int **) malloc(sizeof(int*)*numSets);
  memset(self->uSets, 0, sizeof(int*)*numSets);

  self->labels = (float*) malloc(sizeof(float)*numSets);
  memset(self->labels, 0, sizeof(float)*numSets);
  
  self->uSetsSize = (int*) malloc(sizeof(int) * numSets);
  memset(self->uSetsSize, 0, sizeof(int)*numSets);

  self->itemSets = (int **) malloc(sizeof(int*)*nItems);
  memset(self->itemSets, 0, sizeof(int*)*nItems);

  self->itemSetsSize = (int *) malloc(sizeof(int)*nItems);
  memset(self->itemSetsSize, 0, sizeof(int)*nItems);
  
  self->items = (int*) malloc(sizeof(int)*nUserItems);
  memset(self->items, 0, sizeof(int)*nUserItems);
}


void UserSets_free(UserSets *self, int nItems) {
  int i;
  
  for (i = 0; i < self->numSets; i++) {
    free(self->uSets[i]);
  }

  for (i = 0; i < self->nUserItems; i++) {
    free(self->itemSets[self->items[i]]);
  }
  
  free(self->itemSets);
  free(self->itemSetsSize);
  free(self->labels);
  free(self->items);
  free(self->uSetsSize);
}


void Data_init(Data *self, int nUsers, int nItems) {
  int i;
  self->nUsers = nUsers;
  self->nItems = nItems;

  self->userSets = (UserSets*) malloc(sizeof(UserSets)*nUsers);
  memset(self->userSets, 0, sizeof(UserSets)*nUsers);

}


void Data_free(Data *self) {
  int i;
  UserSets *userSets;

  for (i = 0; i < self->nUsers; i++) {
    UserSets_free(&(self->userSets[i]), self->nItems);
  }
  
  free(self->userSets);
}


#ifndef _DATASTRUCT_H
#define _DATASTRUCT_H

#include <map>

using namespace std;

//map of all items to their id
map<int, string> idArtist;
map<string, int> artistId;


typedef struct {
  
  //indices of set to which item belongs
  int* setInd;
  
  //number of sets to which item belongs
  int numSets; 

} ItemSets;

void ItemSets_init(ItemSets *self, int sz);
void ItemSets_free(ItemSets *self);

typedef struct {
  char *userId;
 
  //number of labeled sets in user history
  int numSets;

  //array of user sets
  int **uSets;

  //array of set labels 
  float *labels;

  //map of items to user sets arr
  map<int, ItemSets> itemSets; 
} UserSets;

void UserSets_init(UserSets *self, char *userId, int numSets);
void UserSets_free(UserSets *self);

#endif



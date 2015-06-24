#ifndef _SET_H_
#define _SET_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

typedef struct {
  int *bVec; 
  int bVecSz; //size of above array
  //no. of elements in set
  //NOTE: based on no. of elements above size will be computed
  int nElem; 
} Set;

void Set_init(Set *self, int nElem);
void Set_reset(Set *self);
void Set_free(Set *self);
void Set_addElem(Set *self, int elem);
void Set_delElem(Set *self, int elem);
void Set_union(Set *uni, Set *a, Set *b);
void Set_intersection(Set *inters, Set *a, Set *b);
void Set_display(Set *self);
int Set_numElem(Set *self);

#endif

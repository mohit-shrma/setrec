#include "set.h"


void Set_init(Set *self, int nElem) {
  
  self->nElem = nElem;

  //determine the size of int arr
  self->bVecSz = nElem/(sizeof(int)*8);
  if (nElem%(sizeof(int)*8) > 0) {
    self->bVecSz += 1;
  }
  
  self->bVec = (int *) malloc(sizeof(int)*self->bVecSz);
  memset(self->bVec, 0, sizeof(int)*self->bVecSz);
}


void Set_reset(Set *self) {
  memset(self->bVec, 0, sizeof(int)*self->bVecSz);
}


void Set_free(Set *self) {
  free(self->bVec);
  free(self);
}


void Set_addElem(Set *self, int elem) {
  int ind, pos;
  ind = elem/(sizeof(int)*8);
  pos = elem % (sizeof(int)*8);
  self->bVec[ind] = self->bVec[ind] | (1 << pos); 
}


void Set_union(Set *uni, Set *a, Set *b) {
  int i;
  assert(a->nElem == b->nElem);
  assert(a->nElem == uni->nElem);
  for (i = 0; i < uni->bVecSz; i++) {
    uni->bVec[i] = a->bVec[i] | b->bVec[i];
  }
}


void Set_intersection(Set *inters, Set *a, Set *b) {
  int i;
  assert(a->nElem == b->nElem);
  assert(a->nElem == inters->nElem);
  for (i = 0; i < inters->bVecSz; i++) {
    inters->bVec[i] = a->bVec[i] & b->bVec[i];
  }
}


/*
 * copied from somewhere on web
 * TODO: verify before using it in production
 */

int numberOfSetBits(int i)
{
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

//TODO: what about signed bit 
//count no. of bits set using brian kernighans algo
int numBitsSet(int v) {
  int c; // c accumulates the total bits set in v
  for (c = 0; v; c++)
  {
      v &= v - 1; // clear the least significant bit set
  }
  return c;
}


int Set_numElem(Set *self) {
  int i, nElem;
  nElem = 0;
  for (i = 0; i < self->bVecSz; i++) {
    nElem += numberOfSetBits(self->bVec[i]);
  }
  return nElem;
}

//naive implementation to display items of set
void Set_display(Set *self) {
  int i, j;
  
  printf("\n");
  for (i = 0; i < self->bVecSz; i++) {
    for (j = 0; j < sizeof(int)*8; j++) {
      if (self->bVec[i] & (1 << j)) {
        printf("%d ", i*32 + j);
      }
    }
  }

}



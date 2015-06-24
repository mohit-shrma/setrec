#include <stdio.h>
#include "set.h"

int main(int argc, char **argv) {
  
  Set *a = (Set*) malloc(sizeof(Set));
  Set *b = (Set*) malloc(sizeof(Set));

  Set *c = (Set*) malloc(sizeof(Set));

  Set_init(a, 70);
  Set_init(b, 70);
  Set_init(c, 70);

  Set_addElem(a, 2);
  Set_addElem(a, 4);
  Set_addElem(a, 45);
  Set_addElem(a, 62);
  Set_addElem(a, 35);
  Set_addElem(a, 100);

  Set_addElem(b, 2);
  Set_addElem(b, 1);
  Set_addElem(b, 45);
  Set_addElem(b, 33);
  Set_addElem(b, 62);
  Set_addElem(b, 69);
  
  printf("\nsize of int: %d", sizeof(int));

  printf("\n%d Elements of a: ", Set_numElem(a));
  Set_display(a);

  printf("\n%d Elements of b: ", Set_numElem(b));
  Set_display(b);

  Set_union(c, a, b);
  printf("\n%d Elements of c: ", Set_numElem(c));
  Set_display(c);
  Set_reset(c);

  Set_intersection(c, a, b); 
  printf("\n%d Elements of c: ", Set_numElem(c));
  Set_display(c);

  Set_free(a);
  Set_free(b);
  Set_free(c);
  return 0;
}



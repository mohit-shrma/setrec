#include <stdio.h>
#include <omp.h>


int main(int argc, char *argv[]) {

  int i, result;

  result = 0;
#pragma omp parallel for private(i) reduction(+:result)
for (i = 0; i < 100000; i++) {
  result += 1;
}

  printf("\nresult = %d", result);

  return 0;
}

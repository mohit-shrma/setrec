#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_statistics_double.h>
#include <stdio.h>
//gcc testSpearman.c -g -Wall -lgsl -lgslcblas
//gcc -c -g -Wall -I/export/scratch/mohit/lib/gsl/include testSpearman.c
//gcc -L/export/scratch/mohit/lib/gsl/lib testSpearman.o -lgsl -lgslcblas -lm
//export LD_LIBRARY_PATH=/export/scratch/mohit/lib/gsl/lib

int main(int argc, char *argv[]) {
 
  double a[] = {1,2,3,4,5};
  double b[] = {3,4,5,4,7};
  double w[] = {0,0,0,0,0,
                0,0,0,0,0};
  double pearson = gsl_stats_correlation(a, 1, b, 1, 5);
  double spearman = gsl_stats_spearman(a, 1, b, 1, 5, w);
  printf("\npearson = %f spearman = %f", pearson, spearman);

  return 0;
}


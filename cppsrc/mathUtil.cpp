#include "mathUtil.h"

float sigmoid(float x, float k) {
  float expVal = exp((double) -(k*x));
  float fSigm = 1.0/(1.0 + expVal);
  return fSigm;
}


double spearmanRankCorrN(std::vector<float> x, std::vector<float> y, int N) {

  double *actualRat     = (double*) malloc(sizeof(double)*N);
  memset(actualRat, 0, sizeof(double)*N);
  double *predRat       = (double*) malloc(sizeof(double)*N);
  memset(predRat, 0, sizeof(double)*N);
  double *spearmanWork  = (double*) malloc(sizeof(double)*2*N);
  double uSpearMan;
  for (int i = 0; i < N && i < (int)x.size(); i++) {
    actualRat[i] = x[i];
    predRat[i] = y[i];
  }
    
  uSpearMan = gsl_stats_spearman(actualRat, 1, predRat, 1, N, spearmanWork);

  free(actualRat);
  free(predRat);
  free(spearmanWork);
  
  return uSpearMan;
}




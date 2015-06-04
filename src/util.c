#include "util.h"

float dotProd(float *u, float *v, int sz) {
  float prod = 0;
  int i;
  for (i = 0; i < sz; i++) {
    prod += u[i]*v[i];
  }
  return prod;
}


float pearsonCorr(float *x, float *y, int n) {
  
  int i;
  float xMean = 0, yMean = 0;
  float xyMeanDev = 0, xMeanDevSqr = 0, yMeanDevSqr = 0;
  float corr;

  for (i = 0; i < n; i++) {
    xMean += x[i];
    yMean += y[i];
  }
  xMean = xMean/n;
  yMean = yMean/n;

  for (i = 0; i < n; i++) {
    xyMeanDev += (x[i] - xMean)*(y[i] - yMean);
    xMeanDevSqr += (x[i] - xMean)*(x[i] - xMean);
    yMeanDevSqr += (y[i] - yMean)*(y[i] - yMean);
  }
  
  corr = xyMeanDev/(sqrt(xMeanDevSqr)*sqrt(yMeanDevSqr));
  return corr;
}


void writeIntVector(int *vec, int n, char *fileName) {
  FILE *fp = NULL;
  int i;
  fp = fopen(fileName, "w");
 
  if (fp == NULL) {
    printf("\nErr: cant open file");
  } else {
    for (i = 0; i < n; i++) {
      fprintf(fp, "%d\n", vec[i]);
    }
  }

  fclose(fp);
 }


void writeFloatVector(float *vec, int n, char *fileName) {
  FILE *fp = NULL;
  int i;
  fp = fopen(fileName, "w");
 
  if (fp == NULL) {
    printf("\nErr: cant open file");
  } else {
    for (i = 0; i < n; i++) {
      fprintf(fp, "%f\n", vec[i]);
    }
  }

  fclose(fp);
}



void writeMat(float **mat, int nrows, int ncols, char *fileName) {
  FILE *fp = NULL;
  int i, j;
  fp = fopen(fileName, "w");
  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) {
      fprintf(fp, "%f ", mat[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}



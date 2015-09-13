#ifndef _UTIL_H_
#define _UTIL_H_

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
float sigmoid(float x, float k);
void coeffUpdateWOReg(float *fac, float *grad, float stepSize, int facDim);
void coefficientNormUpdate(float *fac, float *grad, float reg, float learnRate, int facDim);
void coeffUpdate(float *fac, float *grad, float reg, float learnRate, int facDim);
void coeffUpdateAdap(float *fac, float *grad, float reg, float learnRate, 
    int facDim, int t);
float dotProd(float *u, float *v, int sz);
float norm(float *v, int sz);
float matNorm(float **mat, int nrows, int ncols);
float pearsonCorr(float *x, float *y, int n);
void writeIntVector(int *vec, int n, char *fileName);
void writeFloatVector(float *vec, int n, char *fileName);
void writeMat(float **mat, int nrows, int ncols, char *fileName);
double generateGaussianNoise(double mu, double sigma);
void writeUpperMat(float **mat, int nrows, int ncols, char *fileName);
void copyMat(float **fromMat, float **toMat, int nrows, int ncols);
#endif

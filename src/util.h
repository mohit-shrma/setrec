#ifndef _UTIL_H_
#define _UTIL_H_

#include <math.h>
#include <stdio.h>

float dotProd(float *u, float *v, int sz);
float pearsonCorr(float *x, float *y, int n);
void writeIntVector(int *vec, int n, char *fileName);
void writeFloatVector(float *vec, int n, char *fileName);

#endif

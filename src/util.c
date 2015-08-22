#include "util.h"


void coeffUpdate(float *fac, float *grad, float reg, float learnRate, int facDim) {

  int k;
  float facNorm;

  for (k = 0; k < facDim; k++) {
    fac[k] -= learnRate*(grad[k] + 2.0*reg*fac[k]); 
  }

}


void coefficientNormUpdate(float *fac, float *grad, float reg, float learnRate, int facDim) {

  int k;
  float facNorm;

  for (k = 0; k < facDim; k++) {
    fac[k] -= learnRate*(grad[k] + reg*fac[k]); 
  }

  //normalize factor
  facNorm = norm(fac, facDim);
  for (k = 0; k < facDim; k++) {
    fac[k] = fac[k]/facNorm;
  }

}


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

  if (fp == NULL) {
    printf("\nCan't open file %s", fileName);
    exit(0);
  }
  
  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) {
      fprintf(fp, "%f ", mat[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}


void copyMat(float **fromMat, float **toMat, int nrows, int ncols) {
  
  int i, j;
  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) {
      toMat[i][j] = fromMat[i][j];
    }
    //memcpy(toMat[i], fromMat[i], sizeof(float)*ncols);
  }

}


void writeUpperMat(float **mat, int nrows, int ncols, char *fileName) {
  FILE *fp = NULL;
  int i, j;
  fp = fopen(fileName, "w");
  for (i = 0; i < nrows; i++) {
    for (j = i+1; j < ncols; j++) {
      fprintf(fp, "\n%d %d %f", i, j, mat[i][j]);
    }
  }
  fclose(fp);
}


float norm(float *v, int sz) {
  int i;
  float norm = 0;
  for (i = 0; i < sz; i++) {
    norm += v[i]*v[i];
  }
  return sqrt(norm);
}


float matNorm(float **mat, int nrows, int ncols) {
  int i;
  float matNorm = 0;
  for (i = 0; i < nrows; i++) {
    matNorm += norm(mat[i], ncols);
  }
  return matNorm/nrows;
}


/*
 * Following implement Box-Muller transform to generate samples from gaussian
 * distributions.
 * refer wikipedia page
 */
double generateGaussianNoise(double mu, double sigma) {
  
  const double epsilon = -DBL_MAX;
  const double two_pi = 2.0*3.14159265358979323846;

  static double z0, z1;
  static int generate = 1;
  generate = !generate;

  if (!generate) {
    return z1 * sigma + mu;
  }

  double u1, u2;
  do {
    u1 = rand() * (1.0 / RAND_MAX);
    u2 = rand() * (1.0 / RAND_MAX);
  } while (u1 <= epsilon);

  z0 = sqrt(-2.0 * log(u1)) * cos(two_pi *u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(two_pi *u2);

  return z0 * sigma + mu;
}


#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <vector>
#include <gsl/gsl_statistics.h>
#include <cstdlib>
#include <cmath>

float sigmoid(float x, float k);
double spearManRankCorrN(std::vector<float> x, std::vector<float> y, int N);

#endif




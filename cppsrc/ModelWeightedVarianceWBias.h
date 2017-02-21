#ifndef _MODEL_WEIGHTED_VARIANCE_W_BIAS_H_
#define _MODEL_WEIGHTED_VARIANCE_W_BIAS_H_

#include "ModelWeightedVariance.h"

class ModelWeightedVarianceWBias: public ModelWeightedVariance {
  public:
    ModelWeightedVarianceWBias(const Params& params): ModelWeightedVariance(params) {}
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif

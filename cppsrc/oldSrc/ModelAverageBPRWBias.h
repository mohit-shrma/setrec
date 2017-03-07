#ifndef _MODEL_AVERAGE_BPR_W_BIAS_H_
#define _MODEL_AVERAGE_BPR_W_BIAS_H_

#include "ModelAverage.h"

class ModelAverageBPRWBias: public ModelAverage {
  public:
    ModelAverageBPRWBias(const Params& params):ModelAverage(params) {}
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, 
        Model& bestModel);
};

#endif


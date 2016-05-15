#ifndef _MODEL_BPR_AVG_H_
#define _MODEL_BPR_AVG_H_

#include "ModelAverage.h"

class ModelAverageBPR: public ModelAverage {
  
  public:
    ModelAverageBPR(const Params& params):ModelAverage(params) {}
    virtual void train(const Data& data, const Params& params, 
        Model& bestModel);
};

#endif

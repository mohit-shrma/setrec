#ifndef _MODEL_HINGE_AVG_H_
#define _MODEL_HINGE_AVG_H_

#include "ModelAverage.h"

class ModelAverageHinge: public ModelAverage {
  
  public:
    ModelAverageHinge(const Params& params):ModelAverage(params) {}
    virtual void train(const Data& data, const Params& params, 
        Model& bestModel);
};


#endif

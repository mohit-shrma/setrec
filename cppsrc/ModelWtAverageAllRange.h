#ifndef _MODEL_WT_AVG_ALL_RANGE_H_
#define _MODEL_WT_AVG_ALL_RANGE_H_

#include "Model.h"
#include "alglib/stdafx.h"
#include "alglib/optimization.h"

class ModelWtAverageAllRange: public Model {
  public:
    ModelWtAverageAllRange(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};


#endif


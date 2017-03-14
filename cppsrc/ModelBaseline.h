#ifndef _MODEL_BASELINE_H_
#define _MODEL_BASELINE_H_

#include <map>

#include "Model.h"

class ModelBaseline: public Model {
  
  public:
    ModelBaseline(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif

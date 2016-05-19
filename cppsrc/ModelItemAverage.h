#ifndef _MODEL_ITEM_AVERAGE_H_
#define _MODEL_ITEM_AVERAGE_H_

#include "Model.h"

class ModelItemAverage: public Model {
  public:
    ModelItemAverage(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif


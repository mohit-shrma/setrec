#ifndef _MODEL_AVERAGE_GBIAS_H_
#define _MODEL_AVERAGE_GBIAS_H_

#include "Model.h"

class ModelAverageWGBias: public Model {
  public:
    ModelAverageWGBias(const Params& params):Model(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual float estItemRating(int user, int item);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif



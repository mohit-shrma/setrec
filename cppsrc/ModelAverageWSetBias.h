#ifndef _MODEL_AVERAGE_WSET_BIAS_H_
#define _MODEL_AVERAGE_WSET_BIAS_H_

#include "ModelAverageWBias.h"


class ModelAverageWSetBias: public ModelAverageWBias {
  public:
    ModelAverageWSetBias(const Params& params):ModelAverageWBias(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
};

#endif

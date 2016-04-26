#ifndef _MODEL_AVERAGE_CONST_H_
#define _MODEL_AVERAGE_CONST_H_

#include "ModelAverage.h"
#include <tuple>

class ModelAverageWCons : public ModelAverage {
  public:
    float constWt;
    ModelAverageWCons(const Params& params)
      :ModelAverage(params), constWt(params.constWt) {}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    std::pair<float, float> setRatingNMaxRat(int user, std::vector<int>& items);
};

#endif

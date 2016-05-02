#ifndef _MODEL_AVERAGE_WBIAS_CONST_H_
#define _MODEL_AVERAGE_WBIAS_CONST_H_

#include "ModelAverageWBias.h"

class ModelAverageWBiasConst: public ModelAverageWBias {
  public:
    float constWt;
    ModelAverageWBiasConst(const Params& params)
      :ModelAverageWBias(params), constWt(params.constWt){}
    virtual void train(const Data& data, const Params& params, 
        Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    std::pair<float, float> setRatingNMaxRat(int user, std::vector<int>& items);
};

#endif


#ifndef _MODEL_MAJORITY_CONST_H_
#define _MODEL_MAJORITY_CONST_H_

#include "ModelMajority.h"

class ModelMajorityWCons: public ModelMajority {
  
  public:
    float constWt;
    ModelMajorityWCons(const Params& params)
      :ModelMajority(params), constWt(params.constWt) {}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    std::pair<float, float> setRatingNMaxRat(int user, std::vector<int>& items);
};

#endif

#ifndef _MODEL_AVG_SIGMOID_BIAS_H_
#define _MODEL_AVG_SIGMOID_BIAS_H_

#include "ModelAverageWBias.h"

class ModelAverageSigmoidWBias: public ModelAverageWBias {
  public:
    ModelAverageSigmoidWBias(const Params& params):ModelAverageWBias(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
}; 

#endif


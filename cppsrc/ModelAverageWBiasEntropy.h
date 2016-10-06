#ifndef _MODEL_AVERAGE_WBIAS_ENTROPY_H_
#define _MODEL_AVERAGE_WBIAS_ENTROPY_H_

#include "ModelAverageWBias.h"

class ModelAverageWBiasEntropy: public ModelAverageWBias {
  public:
    ModelAverageWBiasEntropy(const Params& params):ModelAverageWBias(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);

};


#endif

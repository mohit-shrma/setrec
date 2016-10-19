#ifndef _MODEL_WEIGHTED_VARIANCE_H_
#define _MODEL_WEIGHTED_VARIANCE_H_

#include "Model.h"

class ModelWeightedVariance: public Model {
  public:
    ModelWeightedVariance(const Params& params): Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    float objective(const std::vector<UserSets>& uSets);
    float objective(const std::vector<UserSets>& uSets, gk_csr_t* mat);
  };

#endif



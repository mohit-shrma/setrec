#ifndef _MODEL_AVERAGE_WBIAS_H_
#define _MODEL_AVERAGE_WBIAS_H_

#include "Model.h"


class ModelAverageWBias: public Model {
  public:
    ModelAverageWBias(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, 
        gk_csr_t *mat);
};

#endif

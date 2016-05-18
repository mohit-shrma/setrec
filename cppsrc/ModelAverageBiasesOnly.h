#ifndef _MODEL_AVERAGE_BIASES_ONLY_H_
#define _MODEL_AVERAGE_BIASES_ONLY_H_

#include "Model.h"

class ModelAverageBiasesOnly: public Model {
  
  public:
    ModelAverageBiasesOnly(const Params& params):Model(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual float estItemRating(int user, int item);  
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
}

#endif


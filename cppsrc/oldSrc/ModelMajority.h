#ifndef _MODEL_MAJORITY_H_
#define _MODEL_MAJORITY_H_

#include <cmath>

#include "Model.h"
#include "util.h"

class ModelMajority: public Model {

  public:
    ModelMajority(const Params& params):Model(params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    float estSetRating(int user, std::vector<int>& items, 
        Eigen::VectorXf& sumFac);
};


#endif

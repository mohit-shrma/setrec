#ifndef _MODEL_FM_H_
#define _MODEL_FM_H_

#include "Model.h"

class ModelFM: public Model {
  public:
    ModelFM(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    float estSetRating(int user, std::vector<int>& items, 
      Eigen::VectorXf& sumFactors) ;
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
};

#endif

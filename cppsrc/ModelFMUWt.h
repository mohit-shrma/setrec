#ifndef _MODEL_FM_U_WT_H_
#define _MODEL_FM_U_WT_H_

#include "Model.h"

class ModelFMUWt: public Model {
  public:
    ModelFMUWt(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    float estSetRating(int user, std::vector<int>& items, 
      Eigen::VectorXf& sumItemFactors, float& avgItemsPairwiseSim);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
};

#endif
